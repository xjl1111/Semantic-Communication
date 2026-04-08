"""
A(src) 正交维度排列组合测试

5 个正交维度（互不影响，可自由组合）：
  D1. Prompt          — 文本前缀，影响语言模型续写方向
  D2. 解码约束         — bad_words 禁止特定 token
  D3. 锐化强度         — 图像预处理，影响 ViT 视觉特征提取
  D4. SR 目标尺寸      — 上采样倍率
  D5. 后处理           — caption 输出后的文本修正

额外单独测试的新方法：
  M1. 对比度增强 (PIL.ImageEnhance)
  M2. 中心裁剪（SR 后取中心区域，去除边缘插值伪影）
  M3. 多次推理投票（beam + sampling 混合，多数票决定）
  M4. 自适应 prompt（先无 prompt 生成，再根据结果二次推理）
  M5. 短文本约束 (max_new_tokens=10, 强制简短输出)

运行：
    .\.venv\Scripts\python.exe VLM_CSC\exp\fig7\test_combo_asrc.py
"""
from __future__ import annotations

import gc
import re
import sys
import time
import itertools
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageEnhance

_ROOT = Path(__file__).resolve().parents[3]
_VLM  = _ROOT / "VLM_CSC"
sys.path.insert(0, str(_VLM / "experiments"))

BLIP_DIR  = _VLM / "data" / "assets" / "downloaded_models" / "blip"
TEST_DIR  = _ROOT / "data" / "datasets" / "catsvsdogs" / "test"
DEVICE    = "cuda"
MAX_PER_CLASS = 250

# ── 关键词正则 ────────────────────────────────────────────────────────────────
_CAT_RE = re.compile(r"\b(cat|cats|kitten|kittens|kitty|feline)\b", re.I)
_DOG_RE = re.compile(r"\b(dog|dogs|puppy|puppies|pup|canine|hound)\b", re.I)


def classify(text: str) -> str:
    c, d = bool(_CAT_RE.search(text)), bool(_DOG_RE.search(text))
    if c and d: return "both"
    if c: return "cat"
    if d: return "dog"
    return "neither"


def a_src_exclusive(label: int, text: str) -> bool:
    c, d = bool(_CAT_RE.search(text)), bool(_DOG_RE.search(text))
    return (c and not d) if label == 0 else (d and not c)


def first_animal(text: str) -> str | None:
    cat_m, dog_m = _CAT_RE.search(text), _DOG_RE.search(text)
    if cat_m and dog_m:
        return "cat" if cat_m.start() < dog_m.start() else "dog"
    if cat_m: return "cat"
    if dog_m: return "dog"
    return None


def a_src_first_animal(label: int, text: str) -> bool:
    fa = first_animal(text)
    if fa is None: return False
    return (fa == "cat") if label == 0 else (fa == "dog")


# ── 加载 ──────────────────────────────────────────────────────────────────────
def load_images(max_per_class: int):
    items = []
    for label, cls in enumerate(["cat", "dog"]):
        d = TEST_DIR / cls
        files = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        if max_per_class > 0: files = files[:max_per_class]
        for f in files:
            items.append((Image.open(f).convert("RGB"), label, f.name))
    return items


def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(str(BLIP_DIR), use_fast=False, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(str(BLIP_DIR), local_files_only=True)
    model.to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad = False
    return proc, model


# ── 图像预处理 ────────────────────────────────────────────────────────────────
def preprocess_image(pil: Image.Image, *,
                     target_size: int = 256,
                     resample=Image.LANCZOS,
                     sharpen_percent: int = 150,
                     sharpen_radius: float = 2.0,
                     sharpen_threshold: int = 3,
                     contrast_factor: float = 1.0,
                     brightness_factor: float = 1.0,
                     color_factor: float = 1.0,
                     center_crop_ratio: float = 1.0,
                     gaussian_blur: float = 0.0) -> Image.Image:
    """通用预处理管线。"""
    w, h = pil.size
    # SR upscale
    if w < target_size or h < target_size:
        scale = max(target_size / w, target_size / h)
        pil = pil.resize((int(w * scale), int(h * scale)), resample)
    # 高斯平滑（去插值伪影）
    if gaussian_blur > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))
    # 锐化
    if sharpen_percent > 0:
        pil = pil.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius, percent=sharpen_percent, threshold=sharpen_threshold))
    # 对比度 / 亮度 / 饱和度增强
    if contrast_factor != 1.0:
        pil = ImageEnhance.Contrast(pil).enhance(contrast_factor)
    if brightness_factor != 1.0:
        pil = ImageEnhance.Brightness(pil).enhance(brightness_factor)
    if color_factor != 1.0:
        pil = ImageEnhance.Color(pil).enhance(color_factor)
    # 中心裁剪（去除边缘插值伪影）
    if center_crop_ratio < 1.0:
        w2, h2 = pil.size
        cw, ch = int(w2 * center_crop_ratio), int(h2 * center_crop_ratio)
        left, top = (w2 - cw) // 2, (h2 - ch) // 2
        pil = pil.crop((left, top, left + cw, top + ch))
    return pil


# ── caption 生成 ──────────────────────────────────────────────────────────────
@torch.inference_mode()
def gen_caption(proc, model, pil: Image.Image, *,
                prompt: str | None = "a photo of a",
                num_beams: int = 4,
                do_sample: bool = False,
                top_p: float = 0.9,
                temperature: float = 1.0,
                length_penalty: float = 1.0,
                bad_words: list[str] | None = None,
                max_new_tokens: int = 64) -> str:
    inputs = proc(images=pil, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    kw = dict(max_new_tokens=max_new_tokens, num_beams=num_beams,
              length_penalty=length_penalty)
    if do_sample:
        kw.update(do_sample=True, top_p=top_p, temperature=temperature, num_beams=1)
    if bad_words:
        kw["bad_words_ids"] = [proc.tokenizer.encode(w, add_special_tokens=False) for w in bad_words]
    ids = model.generate(**inputs, **kw)
    return proc.decode(ids[0], skip_special_tokens=True).strip()


@torch.inference_mode()
def gen_caption_vote(proc, model, pil: Image.Image, *,
                     prompt: str | None, bad_words: list[str] | None,
                     n_runs: int = 5) -> str:
    """多次推理投票：1 次 beam + (n-1) 次 sampling，对类别投票。"""
    captions = []
    # beam search
    captions.append(gen_caption(proc, model, pil, prompt=prompt, num_beams=4, bad_words=bad_words))
    # sampling runs
    for _ in range(n_runs - 1):
        captions.append(gen_caption(proc, model, pil, prompt=prompt, do_sample=True,
                                    top_p=0.9, temperature=0.8, bad_words=bad_words))
    # 投票
    votes = Counter()
    for c in captions:
        fa = first_animal(c)
        if fa: votes[fa] += 1
    if not votes:
        return captions[0]  # 全部 neither，返回 beam 结果
    winner = votes.most_common(1)[0][0]
    # 找一条包含 winner 的 caption 返回
    for c in captions:
        if (winner == "cat" and _CAT_RE.search(c)) or (winner == "dog" and _DOG_RE.search(c)):
            return c
    return captions[0]


@torch.inference_mode()
def gen_caption_adaptive(proc, model, pil: Image.Image, *,
                         bad_words: list[str] | None) -> str:
    """自适应二次推理：先无 prompt 获取粗略描述，若含动物词则用其构造精确 prompt 再推理。"""
    # 第一轮：无 prompt
    c1 = gen_caption(proc, model, pil, prompt=None, num_beams=4, bad_words=bad_words)
    fa = first_animal(c1)
    if fa:
        # 已有明确动物词，用精确 prompt 二次确认
        refined_prompt = f"a photo of a {fa}"
        c2 = gen_caption(proc, model, pil, prompt=refined_prompt, num_beams=4, bad_words=bad_words)
        return c2
    # 没识别出来，用通用 prompt 再试一次
    c2 = gen_caption(proc, model, pil, prompt="this is a", num_beams=4, bad_words=bad_words)
    return c2


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: 正交维度排列组合
# ══════════════════════════════════════════════════════════════════════════════

# D1: Prompt
D1_PROMPT = {
    "photo_of_a":       "a photo of a",
    "this_is_a":        "this is a",
    "animal_a":         "a photo of an animal, a",
}

# D2: 解码约束
D2_CONSTRAINT = {
    "none":      [],
    "ban_and":   ["and"],
}

# D3: 锐化
D3_SHARPEN = {
    "normal":   dict(sharpen_percent=150, sharpen_radius=2.0, sharpen_threshold=3),
    "strong":   dict(sharpen_percent=250, sharpen_radius=2.0, sharpen_threshold=2),
}

# D4: 后处理
D4_POSTPROC = {
    "exclusive":    False,   # 排他性匹配
    "first_animal": True,    # 取首个动物词
}

# 选择组合（不跑全部 3×2×2×2=24，跑关键组合）
ORTHO_COMBOS = list(itertools.product(
    D1_PROMPT.items(),
    D2_CONSTRAINT.items(),
    D3_SHARPEN.items(),
    D4_POSTPROC.items(),
))

# ══════════════════════════════════════════════════════════════════════════════
# Part 2: 新方法（超越 prompt / 解码的维度）
# ══════════════════════════════════════════════════════════════════════════════

NEW_METHODS = [
    # --- 对比度增强 ---
    ("对比度增强 1.3x",
     dict(contrast_factor=1.3), "a photo of a", [], 4, False, False, False),
    ("对比度增强 1.5x",
     dict(contrast_factor=1.5), "a photo of a", [], 4, False, False, False),
    ("对比度+饱和度 1.3×1.2",
     dict(contrast_factor=1.3, color_factor=1.2), "a photo of a", [], 4, False, False, False),

    # --- 中心裁剪 ---
    ("中心裁剪 90%",
     dict(center_crop_ratio=0.9), "a photo of a", [], 4, False, False, False),
    ("中心裁剪 80%",
     dict(center_crop_ratio=0.8), "a photo of a", [], 4, False, False, False),

    # --- SR 尺寸 ---
    ("SR→384",
     dict(target_size=384), "a photo of a", [], 4, False, False, False),
    ("SR→512",
     dict(target_size=512), "a photo of a", [], 4, False, False, False),

    # --- 短文本约束 ---
    ("短文本 max_tokens=10",
     dict(), "a photo of a", [], 4, False, False, True),
    ("短文本 max_tokens=15",
     dict(), "a photo of a", [], 4, False, False, "15"),

    # --- 投票 ---
    ("5 次投票",
     dict(), "a photo of a", [], 4, True, False, False),
    ("5 次投票 + ban'and'",
     dict(), "a photo of a", ["and"], 4, True, False, False),

    # --- 自适应 prompt ---
    ("自适应二次推理",
     dict(), None, [], 4, False, True, False),
    ("自适应二次推理 + ban'and'",
     dict(), None, ["and"], 4, False, True, False),

    # --- 强锐化 + 对比度 组合 ---
    ("强锐化+对比度1.3",
     dict(sharpen_percent=250, sharpen_radius=2.0, sharpen_threshold=2, contrast_factor=1.3),
     "a photo of a", [], 4, False, False, False),
    ("强锐化+对比度1.3+ban'and'",
     dict(sharpen_percent=250, sharpen_radius=2.0, sharpen_threshold=2, contrast_factor=1.3),
     "a photo of a", ["and"], 4, False, False, False),

    # --- 高斯模糊 + 强锐化（先平滑去噪再锐化） ---
    ("blur0.5+强锐化",
     dict(gaussian_blur=0.5, sharpen_percent=250, sharpen_radius=2.0, sharpen_threshold=2),
     "a photo of a", [], 4, False, False, False),
]


def eval_one(proc, model, items, *, img_kw, prompt, bad_words, num_beams,
             use_vote, use_adaptive, use_first_animal, short_tokens):
    """评估单个方案。"""
    n = len(items)
    correct_excl = 0
    correct_fa   = 0
    kind_cnt = Counter()

    max_tok = 64
    if short_tokens is True:
        max_tok = 10
    elif short_tokens == "15":
        max_tok = 15

    for img, label, _ in items:
        pil = preprocess_image(img, **img_kw)

        if use_vote:
            caption = gen_caption_vote(proc, model, pil, prompt=prompt,
                                       bad_words=bad_words or None, n_runs=5)
        elif use_adaptive:
            caption = gen_caption_adaptive(proc, model, pil,
                                           bad_words=bad_words or None)
        else:
            caption = gen_caption(proc, model, pil, prompt=prompt,
                                  num_beams=num_beams,
                                  bad_words=bad_words or None,
                                  max_new_tokens=max_tok)

        k = classify(caption)
        kind_cnt[k] += 1
        if a_src_exclusive(label, caption):
            correct_excl += 1
        if a_src_first_animal(label, caption):
            correct_fa += 1

    a_excl = correct_excl / n * 100
    a_fa   = correct_fa / n * 100
    both   = kind_cnt.get("both", 0) / n * 100
    miss   = kind_cnt.get("neither", 0) / n * 100
    return a_excl, a_fa, both, miss


def main():
    print(f"[test_combo_asrc] 加载图片... ({MAX_PER_CLASS}/class)")
    items = load_images(MAX_PER_CLASS)
    print(f"  共 {len(items)} 张（猫 {sum(1 for _,l,_ in items if l==0)}，"
          f"狗 {sum(1 for _,l,_ in items if l==1)}）\n")

    print("[加载 BLIP-base]...")
    proc, model = load_blip()
    print("[BLIP 就绪]\n")

    all_results = []  # (name, a_excl, a_fa, both%, miss%)

    # ════════════════════════════════════════════════════════════════════════
    # Part 1: 正交排列组合 (24 组)
    # ════════════════════════════════════════════════════════════════════════
    total_ortho = len(ORTHO_COMBOS)
    print(f"═══ Part 1: 正交排列组合 ({total_ortho} 组) ═══\n")

    for idx, ((p_name, prompt), (c_name, bw), (s_name, s_kw), (pp_name, use_fa)) in enumerate(ORTHO_COMBOS, 1):
        name = f"P:{p_name} C:{c_name} S:{s_name} PP:{pp_name}"
        short_name = f"{p_name}|{c_name}|{s_name}|{pp_name}"
        t0 = time.time()
        print(f"  [{idx:2d}/{total_ortho}] {short_name}...", end=" ", flush=True)

        img_kw = dict(target_size=256, **s_kw)
        a_excl, a_fa, both, miss = eval_one(
            proc, model, items,
            img_kw=img_kw, prompt=prompt, bad_words=bw or None,
            num_beams=4, use_vote=False, use_adaptive=False,
            use_first_animal=use_fa, short_tokens=False)

        metric = a_fa if use_fa else a_excl
        dt = time.time() - t0
        print(f"A={metric:.1f}% excl={a_excl:.1f}% fa={a_fa:.1f}% both={both:.1f}% miss={miss:.1f}%  [{dt:.0f}s]")
        all_results.append((short_name, a_excl, a_fa, both, miss, use_fa))

    # ════════════════════════════════════════════════════════════════════════
    # Part 2: 新方法
    # ════════════════════════════════════════════════════════════════════════
    total_new = len(NEW_METHODS)
    print(f"\n═══ Part 2: 新方法 ({total_new} 组) ═══\n")

    for idx, (name, img_kw_extra, prompt, bw, beams, use_vote, use_adaptive, short_tok) in enumerate(NEW_METHODS, 1):
        t0 = time.time()
        print(f"  [{idx:2d}/{total_new}] {name}...", end=" ", flush=True)

        img_kw = dict(target_size=256)
        img_kw.update(img_kw_extra)

        a_excl, a_fa, both, miss = eval_one(
            proc, model, items,
            img_kw=img_kw, prompt=prompt, bad_words=bw or None,
            num_beams=beams, use_vote=use_vote, use_adaptive=use_adaptive,
            use_first_animal=False, short_tokens=short_tok)

        dt = time.time() - t0
        print(f"A={a_excl:.1f}% fa={a_fa:.1f}% both={both:.1f}% miss={miss:.1f}%  [{dt:.0f}s]")
        all_results.append((name, a_excl, a_fa, both, miss, False))

    # ════════════════════════════════════════════════════════════════════════
    # 汇总排行
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  ★ A(src) 综合排行榜")
    print(f"{'='*90}")
    print(f"  {'#':>3}  {'方法':<50} {'A(排他)':>8} {'A(首词)':>8} {'both%':>6} {'miss%':>6}")
    print(f"  {'-'*82}")

    # 排序：use_fa=True 的按 a_fa，否则按 a_excl
    ranked = sorted(all_results, key=lambda r: (r[2] if r[5] else r[1]), reverse=True)
    for rank, (name, a_excl, a_fa, both, miss, use_fa) in enumerate(ranked, 1):
        best = a_fa if use_fa else a_excl
        fa_s = f"{a_fa:.1f}%" if use_fa else f"({a_fa:.1f}%)"
        m = " ★" if rank <= 5 else ""
        print(f"  {rank:>3}  {name:<50} {a_excl:>7.1f}% {fa_s:>8} {both:>5.1f}% {miss:>5.1f}%{m}")

    # Top-5 详情
    print(f"\n  ═══ Top-5 详情 ═══")
    for rank, (name, a_excl, a_fa, both, miss, use_fa) in enumerate(ranked[:5], 1):
        best = a_fa if use_fa else a_excl
        print(f"  #{rank} {name}")
        print(f"      A(排他)={a_excl:.1f}%  A(首词)={a_fa:.1f}%  both={both:.1f}%  miss={miss:.1f}%")

    print(f"{'='*90}")


if __name__ == "__main__":
    main()
