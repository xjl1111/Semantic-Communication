"""
八组对比测试：SR 上采样 × 模型 × 提示词

   模型      提示词          SR     组号
  原始BLIP   旧prompt       无SR    ①
  原始BLIP   旧prompt       有SR    ②
  原始BLIP   新prompt       无SR    ③
  原始BLIP   新prompt       有SR    ④
  微调BLIP   旧prompt       无SR    ⑤
  微调BLIP   旧prompt       有SR    ⑥
  微调BLIP   新prompt       无SR    ⑦
  微调BLIP   新prompt       有SR    ⑧

运行：
    .\.venv\Scripts\python.exe VLM_CSC\exp\fig7\test_old_prompt_asrc.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]          # Semantic-Communication/
_VLM  = _ROOT / "VLM_CSC"
sys.path.insert(0, str(_VLM / "model"))
sys.path.insert(0, str(_VLM / "exp"))

from PIL import Image

# ── 配置 ──────────────────────────────────────────────────────────────────────
BLIP_DIR_ORIG = _VLM / "data" / "assets" / "downloaded_models" / "blip"
BLIP_DIR_FT   = _VLM / "data" / "experiments" / "fig7" / "finetuned_blip"
TEST_DIR      = _ROOT / "data" / "datasets" / "catsvsdogs" / "test"
DEVICE        = "cuda"           # 改 "cpu" 若无 GPU
MAX_PER_CLASS = 250              # 每类最多测几张（-1 = 全部）

# 八组 (标签, blip_dir, prompt, caption_mode)
# caption_mode="prompt"    → 无SR（原始32×32直接送BLIP）
# caption_mode="sr_prompt" → 有SR（LANCZOS 32→256 再送BLIP）
COMBOS = [
    ("① 原始BLIP+旧prompt 无SR", BLIP_DIR_ORIG, "a photo of an animal, a", "prompt"),
    ("② 原始BLIP+旧prompt 有SR", BLIP_DIR_ORIG, "a photo of an animal, a", "sr_prompt"),
    ("③ 原始BLIP+新prompt 无SR", BLIP_DIR_ORIG, "a photo of a",            "prompt"),
    ("④ 原始BLIP+新prompt 有SR", BLIP_DIR_ORIG, "a photo of a",            "sr_prompt"),
    ("⑤ 微调BLIP+旧prompt 无SR", BLIP_DIR_FT,   "a photo of an animal, a", "prompt"),
    ("⑥ 微调BLIP+旧prompt 有SR", BLIP_DIR_FT,   "a photo of an animal, a", "sr_prompt"),
    ("⑦ 微调BLIP+新prompt 无SR", BLIP_DIR_FT,   "a photo of a",            "prompt"),
    ("⑧ 微调BLIP+新prompt 有SR", BLIP_DIR_FT,   "a photo of a",            "sr_prompt"),
]

# ── 关键词正则 ────────────────────────────────────────────────────────────────
_CAT_RE = re.compile(r"\b(cat|cats|kitten|kittens|kitty|feline)\b", re.IGNORECASE)
_DOG_RE = re.compile(r"\b(dog|dogs|puppy|puppies|pup|canine|hound)\b", re.IGNORECASE)


def _classify(text: str) -> str:
    """返回 'cat' / 'dog' / 'both' / 'neither'"""
    has_cat = bool(_CAT_RE.search(text))
    has_dog = bool(_DOG_RE.search(text))
    if has_cat and has_dog:
        return "both"
    if has_cat:
        return "cat"
    if has_dog:
        return "dog"
    return "neither"


def _a_src_exclusive(label: int, text: str) -> bool:
    """排他性 A(src)：正确动物存在 且 错误动物不存在。"""
    has_cat = bool(_CAT_RE.search(text))
    has_dog = bool(_DOG_RE.search(text))
    if label == 0:   # 图片是猫
        return has_cat and not has_dog
    else:            # 图片是狗
        return has_dog and not has_cat


def load_images(test_dir: Path, max_per_class: int):
    """返回 [(pil_image, label, filename), ...]，0=cat 1=dog。"""
    items = []
    for label, cls in enumerate(["cat", "dog"]):
        cls_dir = test_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"测试集目录不存在: {cls_dir}")
        files = sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.jpeg"))
        if max_per_class > 0:
            files = files[:max_per_class]
        for f in files:
            img = Image.open(f).convert("RGB")
            items.append((img, label, f.name))
    return items


def run_test(sender, items, prompt_name: str):
    """对所有图片生成 caption，统计 A(src) 和各类别分布。"""
    results = []
    total = len(items)
    for i, (img, label, fname) in enumerate(items, 1):
        if i % 50 == 0 or i == total:
            print(f"  [{prompt_name}] {i}/{total}...", flush=True)
        caption = sender.forward(img)
        kind    = _classify(caption)
        correct = _a_src_exclusive(label, caption)
        results.append({
            "label": label,
            "fname": fname,
            "caption": caption,
            "kind": kind,
            "correct": correct,
        })
    return results


def print_stats(results, prompt_name: str):
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    a_src = n_correct / n * 100

    # 按类别统计
    cats = [r for r in results if r["label"] == 0]
    dogs = [r for r in results if r["label"] == 1]
    cat_acc = sum(r["correct"] for r in cats) / len(cats) * 100 if cats else 0
    dog_acc = sum(r["correct"] for r in dogs) / len(dogs) * 100 if dogs else 0

    # caption 类别分布
    from collections import Counter
    kind_cnt = Counter(r["kind"] for r in results)
    both_cnt = kind_cnt.get("both", 0)
    neither_cnt = kind_cnt.get("neither", 0)

    # 失败分析
    fail_both    = sum(1 for r in results if not r["correct"] and r["kind"] == "both")
    fail_wrong   = sum(1 for r in results if not r["correct"] and (
        (r["label"] == 0 and r["kind"] == "dog") or
        (r["label"] == 1 and r["kind"] == "cat")))
    fail_neither = sum(1 for r in results if not r["correct"] and r["kind"] == "neither")

    print(f"\n{'='*60}")
    print(f"  提示词: {prompt_name}")
    print(f"{'='*60}")
    print(f"  A(src) [排他]   = {a_src:.1f}%  ({n_correct}/{n})")
    print(f"  Cat 准确率      = {cat_acc:.1f}%  ({sum(r['correct'] for r in cats)}/{len(cats)})")
    print(f"  Dog 准确率      = {dog_acc:.1f}%  ({sum(r['correct'] for r in dogs)}/{len(dogs)})")
    print(f"  --- Caption 分布 ---")
    print(f"  cat only        = {kind_cnt.get('cat',0):4d}  ({kind_cnt.get('cat',0)/n*100:.1f}%)")
    print(f"  dog only        = {kind_cnt.get('dog',0):4d}  ({kind_cnt.get('dog',0)/n*100:.1f}%)")
    print(f"  both (cat+dog)  = {both_cnt:4d}  ({both_cnt/n*100:.1f}%)  ← 混合 caption")
    print(f"  neither         = {neither_cnt:4d}  ({neither_cnt/n*100:.1f}%)")
    print(f"  --- 失败原因 ---")
    print(f"  BOTH  (混合)    = {fail_both:4d}  ({fail_both/n*100:.1f}%)")
    print(f"  WRONG (全错词)  = {fail_wrong:4d}  ({fail_wrong/n*100:.1f}%)")
    print(f"  MISS  (无关键词)= {fail_neither:4d}  ({fail_neither/n*100:.1f}%)")

    # 典型 BOTH 样本
    both_samples = [r for r in results if r["kind"] == "both"][:5]
    if both_samples:
        print(f"\n  典型 BOTH 样本 (最多5条):")
        for r in both_samples:
            lbl = "cat" if r["label"] == 0 else "dog"
            print(f"    [{lbl}] {r['caption']}")

    return a_src, both_cnt / n * 100


def main():
    print(f"[test_old_prompt_asrc] 加载图片... 测试集: {TEST_DIR}")
    items = load_images(TEST_DIR, MAX_PER_CLASS)
    print(f"  共 {len(items)} 张（猫 {sum(1 for _,l,_ in items if l==0)}，狗 {sum(1 for _,l,_ in items if l==1)}）")

    # 延迟导入
    try:
        from VLM_CSC import SenderCKB_BLIP  # type: ignore
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("VLM_CSC", _VLM / "model" / "VLM-CSC.py")
        mod  = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
        SenderCKB_BLIP = mod.SenderCKB_BLIP

    summary = {}
    for combo_name, blip_dir, prompt_text, caption_mode in COMBOS:
        model_tag = "微调" if "finetuned" in str(blip_dir) else "原始"
        sr_tag    = "有SR" if caption_mode == "sr_prompt" else "无SR"
        print(f"\n[加载 BLIP] {model_tag}模型  模式={caption_mode}({sr_tag})  prompt={prompt_text!r}")
        print(f"           目录: {blip_dir}")
        sender = SenderCKB_BLIP(
            blip_dir=str(blip_dir),
            use_real_ckb=True,
            device=DEVICE,
            caption_mode=caption_mode,
            caption_prompt=prompt_text,
        )
        results = run_test(sender, items, combo_name)
        a_src, both_pct = print_stats(results, combo_name)
        summary[combo_name] = (a_src, both_pct)

        # 释放模型显存
        del sender
        try:
            import torch, gc
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    print(f"\n{'='*70}")
    print("  ★ 八组汇总对比")
    print(f"  {'组合':<34} {'A(src)':>8}  {'混合caption%':>12}")
    print(f"  {'-'*58}")
    for name, (a, b) in summary.items():
        print(f"  {name:<34} {a:>7.1f}%  {b:>11.1f}%")
    # SR 效果小结
    print(f"\n  --- SR 效果小结 ---")
    keys = list(summary.keys())
    pairs = [(keys[i], keys[i+1]) for i in range(0, len(keys), 2)]  # 无SR/有SR 成对
    for no_sr_key, sr_key in pairs:
        diff = summary[sr_key][0] - summary[no_sr_key][0]
        sign = "+" if diff >= 0 else ""
        base = no_sr_key.replace(" 无SR", "")
        print(f"  {base:<30}  SR增益: {sign}{diff:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
