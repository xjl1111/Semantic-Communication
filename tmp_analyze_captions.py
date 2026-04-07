import json
import os
from collections import Counter

base = r"d:\code\pycode\Semantic-Communication\VLM_CSC\data\experiments\fig8\caption_cache\with_med\blip"

for task in ["task_1_cifar", "task_2_birds", "task_3_catsvsdogs"]:
    path = os.path.join(base, task, "blip_captions.json")
    if not os.path.exists(path):
        print(f"[NOT FOUND] {path}")
        continue
    d = json.load(open(path, "r", encoding="utf-8"))
    vals = list(d.values())
    print(f"\n{'='*60}")
    print(f"Dataset: {task}  |  Total captions: {len(vals)}")
    print(f"{'='*60}")
    
    # Show first 10 captions
    for i, v in enumerate(vals[:10]):
        print(f"  [{i}] {v}")
    
    # Analyze average length
    lengths = [len(str(v).split()) for v in vals]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"\n  Avg word count: {avg_len:.1f}")
    print(f"  Min/Max words: {min(lengths)} / {max(lengths)}")
    
    # Unique captions
    unique = len(set(str(v) for v in vals))
    print(f"  Unique captions: {unique} / {len(vals)} ({100*unique/len(vals):.1f}%)")
    
    # Most common words
    all_words = []
    for v in vals:
        all_words.extend(str(v).lower().split())
    word_freq = Counter(all_words)
    print(f"  Top 15 words: {word_freq.most_common(15)}")
