"""Analyze fig7 eval details CSV to debug low D(recon_clf)."""
import csv
import collections
import os

f = r"D:\code\pycode\Semantic-Communication\VLM_CSC\data\experiments\fig7\fig_fig7_eval_details.csv"
if not os.path.exists(f):
    print("NOT FOUND:", f)
    exit()

rows = list(csv.DictReader(open(f, encoding="utf-8")))
print(f"Total rows: {len(rows)}")
snr_list = sorted(set(r["snr_db"] for r in rows), key=float)

for snr in snr_list:
    sub = [r for r in rows if r["snr_db"] == snr]
    labels = [int(r["label"]) for r in sub]
    pred_r = [int(r["pred_reconstructed"]) for r in sub]
    pred_o = [int(r["pred_original"]) for r in sub]

    d_correct = sum(1 for l, p in zip(labels, pred_r) if l == p)
    c_correct = sum(1 for l, p in zip(labels, pred_o) if l == p)

    pred_r_dist = dict(collections.Counter(pred_r))
    label_dist = dict(collections.Counter(labels))

    cat_as_cat = sum(1 for l, p in zip(labels, pred_r) if l == 0 and p == 0)
    cat_as_dog = sum(1 for l, p in zip(labels, pred_r) if l == 0 and p == 1)
    dog_as_dog = sum(1 for l, p in zip(labels, pred_r) if l == 1 and p == 1)
    dog_as_cat = sum(1 for l, p in zip(labels, pred_r) if l == 1 and p == 0)

    cat_acc = cat_as_cat / (cat_as_cat + cat_as_dog) if (cat_as_cat + cat_as_dog) > 0 else 0.0
    dog_acc = dog_as_dog / (dog_as_dog + dog_as_cat) if (dog_as_dog + dog_as_cat) > 0 else 0.0

    print(f"\nSNR={snr}dB  N={len(sub)}")
    print(f"  label dist      : {label_dist}  (0=cat, 1=dog)")
    print(f"  pred_original   : {dict(collections.Counter(pred_o))}  C={c_correct/len(sub):.1%}")
    print(f"  pred_reconstruct: {pred_r_dist}  D={d_correct/len(sub):.1%}")
    print(f"  Cat->Cat={cat_as_cat} Cat->Dog={cat_as_dog}  cat_acc={cat_acc:.1%}")
    print(f"  Dog->Dog={dog_as_dog} Dog->Cat={dog_as_cat}  dog_acc={dog_acc:.1%}")

print("\n=== Sample rows at SNR=0 ===")
for r in [x for x in rows if x["snr_db"] == "0.0"][:5]:
    lab = "cat" if int(r["label"]) == 0 else "dog"
    pred = "cat" if int(r["pred_reconstructed"]) == 0 else "dog"
    print(f"  [{lab}] src=[{r['source_text']}]")
    print(f"        rec=[{r['recovered_text']}]  pred_r={pred}")
    print()

print("=== Sample rows at SNR=10 ===")
for r in [x for x in rows if x["snr_db"] == "10.0"][:5]:
    lab = "cat" if int(r["label"]) == 0 else "dog"
    pred = "cat" if int(r["pred_reconstructed"]) == 0 else "dog"
    print(f"  [{lab}] src=[{r['source_text']}]")
    print(f"        rec=[{r['recovered_text']}]  pred_r={pred}")
    print()
