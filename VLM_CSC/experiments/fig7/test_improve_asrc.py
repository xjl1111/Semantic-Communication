"""
A(src) 改进方法穷举测试 — 不修改模型权重，不影响实验公平性

测试维度（全部基于原始 BLIP-base，无微调）：
  1. Prompt 工程 — 7 种不同提示词
  2. 解码策略   — greedy / beam-4 / beam-8 / nucleus sampling
  3. 约束解码   — 禁止 "and" 防止列举
  4. 长度惩罚   — length_penalty 调整
  5. 后处理     — 双动物时取第一个提到的动物词
  6. 插值方法   — LANCZOS vs BICUBIC vs BILINEAR
  7. 锐化参数   — 多组对比
  8. 高斯平滑   — SR 后加模糊去除插值伪影

运行：
    .\.venv\Scripts\python.exe VLM_CSC\exp\fig7\test_improve_asrc.py
"""
from __future__ import annotations

import gc
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageFilter

# ── 路径 ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
_VLM  = _ROOT / "VLM_CSC"
sys.path.insert(0, str(_VLM / "experiments"))

BLIP_DIR  = _VLM / "data" / "assets" / "downloaded_models" / "blip"
TEST_DIR  = _ROOT / "data" / "datasets" / "catsvsdogs" / "test"
DEVICE    = "cuda"
MAX_PER_CLASS = 250  # 每类最多测几张

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
    """后处理：返回文本中第一个出现的动物词（cat/dog）。"""
    cat_m = _CAT_RE.search(text)
    dog_m = _DOG_RE.search(text)
    if cat_m and dog_m:
        return "cat" if cat_m.start() < dog_m.start() else "dog"
    if cat_m: return "cat"
    if dog_m: return "dog"
    return None


def a_src_first_animal(label: int, text: str) -> bool:
    """后处理版 A(src)：双动物 caption 取第一个提到的。"""
    fa = first_animal(text)
    if fa is None: return False
    return (fa == "cat") if label == 0 else (fa == "dog")


# ── 加载图片 ──────────────────────────────────────────────────────────────────
def load_images(max_per_class: int):
    items = []
    for label, cls in enumerate(["cat", "dog"]):
        d = TEST_DIR / cls
        files = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        if max_per_class > 0:
            files = files[:max_per_class]
        for f in files:
            items.append((Image.open(f).convert("RGB"), label, f.name))
    return items


# ── BLIP 加载（只加载一次） ──────────────────────────────────────────────────
def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor
    processor = BlipProcessor.from_pretrained(str(BLIP_DIR), use_fast=False, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(str(BLIP_DIR), local_files_only=True)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model


# ── SR 预处理函数 ─────────────────────────────────────────────────────────────
def sr_preprocess(pil: Image.Image,
                  target_size: int = 256,
                  resample=Image.LANCZOS,
                  sharpen_radius: float = 2.0,
                  sharpen_percent: int = 150,
                  sharpen_threshold: int = 3,
                  gaussian_blur: float = 0.0) -> Image.Image:
    """SR 上采样 + 锐化（+ 可选高斯平滑）。"""
    w, h = pil.size
    scale = max(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    up = pil.resize((new_w, new_h), resample)
    if gaussian_blur > 0:
        up = up.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))
    if sharpen_percent > 0:
        up = up.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius, percent=sharpen_percent, threshold=sharpen_threshold))
    return up


# ── 生成 caption ──────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_caption(processor, model, pil: Image.Image,
                     prompt: str | None = "a photo of a",
                     num_beams: int = 4,
                     do_sample: bool = False,
                     top_p: float = 0.9,
                     temperature: float = 1.0,
                     length_penalty: float = 1.0,
                     bad_words: list[str] | None = None,
                     max_new_tokens: int = 64) -> str:
    inputs = processor(images=pil, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        length_penalty=length_penalty,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, top_p=top_p, temperature=temperature, num_beams=1)
    if bad_words:
        bad_ids = [processor.tokenizer.encode(w, add_special_tokens=False) for w in bad_words]
        gen_kwargs["bad_words_ids"] = bad_ids

    output_ids = model.generate(**inputs, **gen_kwargs)
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


# ── 测试方案定义 ──────────────────────────────────────────────────────────────
# 每个方案: (名称, sr_kwargs, gen_kwargs, use_first_animal_postprocess)

METHODS = [
    # ===== Baseline =====
    ("Baseline (beam4, prompt='a photo of a')",
     dict(), dict(prompt="a photo of a", num_beams=4), False),

    # ===== 1. Prompt 工程 =====
    ("Prompt: 'this is a'",
     dict(), dict(prompt="this is a", num_beams=4), False),

    ("Prompt: 'a picture of a'",
     dict(), dict(prompt="a picture of a", num_beams=4), False),

    ("Prompt: 'a close up photo of a'",
     dict(), dict(prompt="a close up photo of a", num_beams=4), False),

    ("Prompt: 'an image of a'",
     dict(), dict(prompt="an image of a", num_beams=4), False),

    ("Prompt: 'the animal in the image is a'",
     dict(), dict(prompt="the animal in the image is a", num_beams=4), False),

    ("Prompt: None (无prompt)",
     dict(), dict(prompt=None, num_beams=4), False),

    # ===== 2. 解码策略 =====
    ("Decode: greedy (beam=1)",
     dict(), dict(prompt="a photo of a", num_beams=1), False),

    ("Decode: beam=8",
     dict(), dict(prompt="a photo of a", num_beams=8), False),

    ("Decode: nucleus p=0.9 t=0.7",
     dict(), dict(prompt="a photo of a", do_sample=True, top_p=0.9, temperature=0.7), False),

    ("Decode: nucleus p=0.5 t=0.5",
     dict(), dict(prompt="a photo of a", do_sample=True, top_p=0.5, temperature=0.5), False),

    # ===== 3. 约束解码 — 禁止 "and" =====
    ("BanWord: 'and'",
     dict(), dict(prompt="a photo of a", num_beams=4, bad_words=["and"]), False),

    ("BanWord: 'and' + ','",
     dict(), dict(prompt="a photo of a", num_beams=4, bad_words=["and", ","]), False),

    # ===== 4. 长度惩罚 =====
    ("LenPenalty=0.6 (偏短)",
     dict(), dict(prompt="a photo of a", num_beams=4, length_penalty=0.6), False),

    ("LenPenalty=2.0 (偏长)",
     dict(), dict(prompt="a photo of a", num_beams=4, length_penalty=2.0), False),

    # ===== 5. 后处理 — 双动物取首个 =====
    ("FirstAnimal 后处理",
     dict(), dict(prompt="a photo of a", num_beams=4), True),

    ("FirstAnimal + 旧prompt",
     dict(), dict(prompt="a photo of an animal, a", num_beams=4), True),

    # ===== 6. 插值方法 =====
    ("Resample: BICUBIC",
     dict(resample=Image.BICUBIC), dict(prompt="a photo of a", num_beams=4), False),

    ("Resample: BILINEAR",
     dict(resample=Image.BILINEAR), dict(prompt="a photo of a", num_beams=4), False),

    # ===== 7. 锐化参数 =====
    ("Sharpen: 强(r=2,p=250,t=2)",
     dict(sharpen_radius=2.0, sharpen_percent=250, sharpen_threshold=2),
     dict(prompt="a photo of a", num_beams=4), False),

    ("Sharpen: 无锐化",
     dict(sharpen_percent=0),
     dict(prompt="a photo of a", num_beams=4), False),

    ("Sharpen: 轻(r=1,p=80,t=5)",
     dict(sharpen_radius=1.0, sharpen_percent=80, sharpen_threshold=5),
     dict(prompt="a photo of a", num_beams=4), False),

    # ===== 8. 高斯平滑去插值伪影 =====
    ("GaussBlur=0.5 after SR",
     dict(gaussian_blur=0.5),
     dict(prompt="a photo of a", num_beams=4), False),

    ("GaussBlur=1.0 after SR",
     dict(gaussian_blur=1.0),
     dict(prompt="a photo of a", num_beams=4), False),

    # ===== 9. 组合策略 =====
    ("Combo: 'this is a' + ban'and' + FirstAnimal",
     dict(), dict(prompt="this is a", num_beams=4, bad_words=["and"]), True),

    ("Combo: 'a photo of a' + ban'and' + lenP=0.6",
     dict(), dict(prompt="a photo of a", num_beams=4, bad_words=["and"], length_penalty=0.6), False),

    ("Combo: BICUBIC + 'a photo of a' + ban'and'",
     dict(resample=Image.BICUBIC),
     dict(prompt="a photo of a", num_beams=4, bad_words=["and"]), False),

    ("Combo: beam8 + ban'and' + FirstAnimal",
     dict(), dict(prompt="a photo of a", num_beams=8, bad_words=["and"]), True),
]


def run_method(processor, model, items, sr_kw, gen_kw, use_first_animal):
    """对所有图片跑一个方案，返回统计。"""
    n = len(items)
    correct_excl = 0
    correct_fa   = 0
    kind_cnt = Counter()

    for img, label, fname in items:
        pil = sr_preprocess(img, **sr_kw)
        caption = generate_caption(processor, model, pil, **gen_kw)
        k = classify(caption)
        kind_cnt[k] += 1
        if a_src_exclusive(label, caption):
            correct_excl += 1
        if use_first_animal:
            if a_src_first_animal(label, caption):
                correct_fa += 1

    a_excl = correct_excl / n * 100
    both_pct = kind_cnt.get("both", 0) / n * 100
    neither_pct = kind_cnt.get("neither", 0) / n * 100
    a_fa = correct_fa / n * 100 if use_first_animal else a_excl
    return a_excl, a_fa, both_pct, neither_pct


def main():
    print(f"[test_improve_asrc] 加载图片... ({MAX_PER_CLASS}/class)")
    items = load_images(MAX_PER_CLASS)
    n_cat = sum(1 for _, l, _ in items if l == 0)
    n_dog = sum(1 for _, l, _ in items if l == 1)
    print(f"  共 {len(items)} 张（猫 {n_cat}，狗 {n_dog}）\n")

    print("[加载 BLIP-base]...")
    processor, model = load_blip()
    print("[BLIP 就绪]\n")

    results = []
    total = len(METHODS)
    for idx, (name, sr_kw, gen_kw, use_fa) in enumerate(METHODS, 1):
        t0 = time.time()
        print(f"[{idx:2d}/{total}] {name}...", end=" ", flush=True)
        a_excl, a_fa, both_pct, neither_pct = run_method(
            processor, model, items, sr_kw, gen_kw, use_fa)
        dt = time.time() - t0
        metric = a_fa if use_fa else a_excl
        print(f"A={metric:.1f}%  both={both_pct:.1f}%  miss={neither_pct:.1f}%  [{dt:.0f}s]")
        results.append((name, a_excl, a_fa, both_pct, neither_pct, use_fa))

    # ── 排行榜 ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  ★ A(src) 排行榜（排他性匹配）")
    print(f"{'='*80}")
    print(f"  {'#':>3}  {'方法':<42} {'A(排他)':>8} {'A(首词)':>8} {'both%':>6} {'miss%':>6}")
    print(f"  {'-'*74}")

    # 按 A(首词 if 后处理 else 排他) 降序排列
    ranked = sorted(results, key=lambda r: (r[2] if r[5] else r[1]), reverse=True)
    for rank, (name, a_excl, a_fa, both_pct, neither_pct, use_fa) in enumerate(ranked, 1):
        best_a = a_fa if use_fa else a_excl
        fa_col = f"{a_fa:.1f}%" if use_fa else "  -  "
        marker = " ★" if rank <= 3 else ""
        print(f"  {rank:>3}  {name:<42} {a_excl:>7.1f}% {fa_col:>8} {both_pct:>5.1f}% {neither_pct:>5.1f}%{marker}")

    print(f"{'='*80}")

    # baseline 对比
    base_a = results[0][1]
    print(f"\n  Baseline = {base_a:.1f}%")
    print(f"  最佳方案相比 Baseline 的增益:")
    best = ranked[0]
    best_a = best[2] if best[5] else best[1]
    print(f"    {ranked[0][0]}  → {best_a:.1f}%  (Δ={best_a - base_a:+.1f}%)")


if __name__ == "__main__":
    main()
