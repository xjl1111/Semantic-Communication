# difference_note.md

本文档专门说明当前结果与论文结果之间的差异性质。

## 一、趋势复现（已满足）

- Fig.7：BLIP 曲线整体高于 LEMON 与 RAM（proxy 结果中成立）。
- Fig.8：有 MED 相较无 MED 遗忘更小（proxy 结果中成立）。
- Fig.9：NAM-on 曲线整体优于多条 NAM-off 固定训练曲线（proxy 结果中成立）。
- Fig.10：已生成 VLM-CSC/JSCC/WITT 的多维比较与 semantic alignment 可视化。

## 二、数值复现（尚未完成）

当前输出不代表论文最终可比数值，属于工程链路验证结果。

## 三、主要偏差来源

1. 关键预训练权重与完整推理链路当前使用 fallback/proxy 运行分支。
2. 真实数据训练和长轮次优化尚未执行（当前以 smoke/proxy 验证为主）。
3. baseline 的具体超参数来自“为复现做的合理实现选择”，并非论文逐项给定。

## 四、收敛到论文数值的下一步

1. 接入论文对应 checkpoint 并锁定版本与哈希。
2. 按 Fig.7~Fig.10 配置进行真实训练与评估（3 seeds 均值+方差）。
3. 覆盖 proxy 结果并对齐论文图中的绝对值与排序。