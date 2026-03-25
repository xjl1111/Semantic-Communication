# repro_report.md

## 复现范围

- 目标论文：Visual Language Model-Based Cross-Modal Semantic Communication Systems (VLM-CSC)
- 目标图：Fig.7, Fig.8, Fig.9, Fig.10

## 论文事实 vs 实现选择

- 论文明确内容：见 `paper_facts.md`
- 合理实现选择：见 `assumptions.md`

## 当前执行模式

- 当前已完成 Step 1~10 的工程实现与脚本连通验证。
- Fig.7~Fig.10 当前在 **proxy 模式** 下执行，用于先打通训练/评估/绘图/日志链路。
- proxy 模式不是论文最终数值复现，真实数值复现需接入完整数据与预训练权重并进行长时训练。

## 复现结果（当前版本）

- Fig.7：已执行，产物位于 `outputs/fig7/`。
	- `results.csv`
	- `curve.png`
	- `logs/run_meta.json`
	- `experiment_note.md`
- Fig.8：已执行，产物位于 `outputs/fig8/`。
	- `results.csv`
	- `med_off_bleu1_map.png`
	- `med_off_bleu2_map.png`
	- `med_on_bleu1_map.png`
	- `med_on_bleu2_map.png`
	- `logs/run_meta.json`
	- `experiment_note.md`
- Fig.9：已执行，产物位于 `outputs/fig9/`。
	- `results.csv`
	- `curve.png`
	- `logs/run_meta.json`
	- `experiment_note.md`
- Fig.10：已执行，产物位于 `outputs/fig10/`。
	- `results.csv`
	- `comparison.png`
	- `comparison.pdf`
	- `semantic_alignment.png`
	- `logs/run_meta.json`
	- `experiment_note.md`

## 趋势一致性检查

- Fig.7：在 proxy 曲线中，BLIP 整体高于 LEMON 与 RAM。
- Fig.8：在 proxy map 中，MED on 相比 MED off 的遗忘程度更低。
- Fig.9：在 proxy 曲线中，NAM-on 整体高于固定 SNR 训练的 NAM-off 曲线。
- Fig.10：已输出 VLM-CSC/JSCC/WITT 多指标对比表与语义对齐可视化。

## 差异分析

- 趋势复现情况：当前为工程级趋势复现（proxy）。
- 数值偏差情况：与论文绝对数值不可直接对齐。
- 原因定位：
	1) 尚未加载论文对应完整预训练权重与数据流程；
	2) 尚未进行长周期真实训练；
	3) baseline 超参数仍为合理实现选择。

## 后续真实复现建议

- 接入真实 CATSvsDOGS、BIRDS、CIFAR 数据流与缓存。
- 接入 BLIP/SD/LEMON/RAM 可用权重，并在统一 seed 下做 3 次重复实验。
- 依照 `configs/` 逐图执行真实训练，覆盖 proxy 结果。
- 在更新后的 `repro_report.md` 中补充论文图趋势与数值偏差定量对比。
