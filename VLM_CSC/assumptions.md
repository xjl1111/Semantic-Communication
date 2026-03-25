# assumptions.md

本文档仅记录论文未明确给出的实现选择，均标记为：**为复现做的合理实现选择**。

## 模型与权重

- BLIP checkpoint 名称未明确：默认使用 BLIP-base image captioning（参数量尽量接近 129MB）。
- SD checkpoint 名称未明确：默认使用 Stable Diffusion v1.5 fp16（参数量尽量接近 1.99GB）。

## 输入与文本

- 图像分辨率默认设为 224x224。
- patch size、tokenizer max length 论文未给出：默认 caption max length=32。
- tokenizer padding 默认 right padding。
- 训练/推理实现中 Transformer 内部位置编码最大长度默认 512。

## Transformer 实现细节

- semantic encoder/decoder 的 dropout、FFN expansion、positional encoding 未明确。
- 默认：dropout=0.1，FFN hidden dim=4*d_model，learned positional embedding。

## 信道与调制

- channel encoder 输出维度、调制方式、功率归一化细节未明确。
- 默认：输出维度与语义维度对齐（初始 C=128）；实值信道实现；按最后维平均功率归一化。

## 损失函数

- 论文提到 minimizing mutual information，但未给可计算形式。
- 默认 mutual_info_proxy：`MSE(decoded_feat, original_feat) + beta * power_penalty`。

## Rayleigh 实现

- 论文未给复基带细节：默认 i.i.d. 实值 Rayleigh 系数实现。

## SSQ 分类器

- 论文未给固定分类器：默认 ResNet18 作为评估器。

## baseline 细节

- JSCC/WITT 的精确配置未给：将采用公开实现的标准配置并统一比较定义。

## 实验运行形态

- Fig.7 在完整 CATSvsDOGS 数据与 LEMON/RAM 可用权重缺失时，先运行 proxy 模式以打通结果产物链路（`results.csv`、`curve.png`、`logs`、`experiment_note.md`）。
- proxy 结果仅用于工程验证，不作为论文最终数值对齐结论。
- Fig.8 在真实 continual training 未开启前，先运行 proxy 模式生成 BLEU-1/2 的 MED on/off 四张 map 与 `results.csv`，用于验证实验输出链路。
- Fig.9 在真实 NAM on/off 训练未完成前，先运行 proxy 模式生成 BLEU-SNR 曲线与 `results.csv`，仅用于工程链路验证。
- Fig.10 在真实 VLM-CSC/JSCC/WITT 全流程训练前，先运行 proxy 模式生成 SSQ、compression ratio、trainable parameters 与 semantic alignment 对比产物。

## 训练超参数

- epoch、batch size、optimizer、scheduler、seed 未明确。
- 默认：AdamW，lr=1e-4，batch=16（显存不足 8 + 累积），3 seeds。
- Stage C 早停默认 `patience=3`，属于为复现做的合理实现选择。

## 关键冲突记录

### NAM 维度冲突
- 论文给出主干维度 128 与 NAM 层宽 56/128/56/56 存在维度耦合不完整。
- 默认采用方案 A：SNR 分支 `1->56->128->56`，特征分支 `d_model->56`，gate 后 `56->d_model` 回投。

### MED 语义描述与公式冲突
- 文字描述偏向“选显著不同样本”，公式为 `R > λ` 才转移。
- 默认按公式实现 `R > λ`，并提供 `transfer_if='greater'|'smaller'` 开关。

## 待补充

- 具体 checkpoint 下载链接与哈希。
- baseline 公开实现引用。
- Fig.7-10 经验超参网格。
- 若环境未安装 `transformers` 或无权重访问，BLIP 封装允许 fallback 模式用于 shape/smoke 测试（不用于最终论文指标）。
- 若环境未安装 `diffusers` 或无权重访问，SD 封装允许 deterministic fallback 图像生成用于 shape/smoke 测试（不用于最终论文指标）。
