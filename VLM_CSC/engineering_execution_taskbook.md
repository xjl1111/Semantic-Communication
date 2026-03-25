# VLM-CSC 工程落地任务单（严格基于事实包）

本文档仅基于 `paper_fact_pack_and_dev_taskbook.txt`，不引入额外论文设定。

---

## 0. 先纠错：对事实包与任务书的检查

### 0.1 结论
- 未发现“必须中止实现”的硬冲突。
- 发现 3 个需要显式标注的不确定点（见第 5 节），可继续工程实现。

### 0.2 需特别声明的点
1. **MED 转移规则语义**
   - [论文明确写出] 公式采用 $R(s_i^{stm})>\lambda$ 触发转移。
   - [不确定点] 该规则与“选择差异大样本”的文字直觉可能不一致。
2. **Stage A 的 MI 目标**
   - [论文明确写出] Stage A 目标是 minimizing mutual information。
   - [不确定点] 未给出可直接编码的 MI 估计公式。
3. **Fig.7 的 LEMON**
   - [论文明确写出] 需要 BLIP/LEMON/RAM 对比。
   - [不确定点] 若无可验证 checkpoint，只能输出 `LEMON unresolved`，不得静默替代。

---

## 1. 完整工程实现方案

### 1.1 主链路（必须优先完成）
- [论文明确写出] 数据流：`image -> BLIP -> text -> semantic encoder -> channel encoder -> channel -> channel decoder -> semantic decoder -> text_hat -> SD -> image_hat`。
- [论文明确写出] BLIP、SD 为预训练模块，不作为 CSC 主训练对象。
- [为复现做的合理实现选择] 通过 text cache 将 BLIP 产物离线化，训练期不重复 caption。

### 1.2 三阶段训练
- [论文明确写出] Stage A：训练 channel encoder/decoder + NAM，使用 MED。
- [论文明确写出] Stage B：训练 semantic encoder/decoder + NAM，使用 CE，使用 MED。
- [论文明确写出] Stage C：A/B 交替迭代至收敛。
- [为复现做的合理实现选择] 提供 early stopping、单次交替周期定义、checkpoint 策略。

### 1.3 评估主线
- [论文明确写出] Fig.8/9/10 必须可输出结果表；Fig.7 涉及 BLIP/LEMON/RAM。
- [为复现做的合理实现选择] 先打通真实主链路 + Stage A/B/C，再做 Fig.8/9，再处理 Fig.7/10 unresolved baseline。

---

## 2. 模块划分、文件结构、函数接口（含 shape）

## 2.1 配置与文档
- `configs/default.yaml`：统一训练/信道/模型超参。
- `unresolved.md`：集中记录不确定点。
- `reproduction_report.md`：复现状态与结果边界。

## 2.2 数据与缓存
- `data/datasets.py`
  - `CIFARDataset.__getitem__(idx) -> Dict`
  - `BirdsDataset.__getitem__(idx) -> Dict`
  - `CatsVsDogsDataset.__getitem__(idx) -> Dict`
  - 输出统一：
    - `image`: `[3,H,W]`
    - `label`: `int`
    - `raw_text`: `str|None`
    - `sample_id`: `str`
    - `dataset_name`: `str`
- `data/cache.py`
  - `append_caption_cache(jsonl_path, item)`
  - `write_caption_cache(jsonl_path, items)`
  - `read_caption_cache(jsonl_path) -> List[CaptionCacheItem]`

## 2.3 发送端/接收端 CKB
- `models/kb_blip.py`
  - `load_model() -> None`
  - `generate_caption(image:[B,3,H,W]) -> List[str]`
  - `tokenize(captions:List[str]) -> {input_ids:[B,L], attention_mask:[B,L]}`
  - `generate_caption_tokens(image:[B,3,H,W]) -> Dict`
  - [论文明确写出] BLIP 做 image-to-text。
  - [为复现做的合理实现选择] checkpoint 名称、`max_length`、tokenizer 细节。
- `models/kb_sd.py`
  - `load_pipeline() -> None`
  - `reconstruct_from_text(texts:List[str]) -> Tensor[B,3,H,W]`
  - `reconstruct_image(text:str) -> PIL.Image`（单样本接口）
  - `expose_components() -> Dict(text_encoder/unet/vae)`
  - [论文明确写出] SD 做 text-to-image。
  - [为复现做的合理实现选择] 采样器、推理步数、guidance、图像尺寸。

## 2.4 semantic / channel / NAM / MED
- `models/semantic_codec.py`
  - `SemanticEncoder.forward(token_ids:[B,L], snr:[B,1]) -> [B,L,128]`
  - `SemanticDecoder.forward(channel_features:[B,L,128], target_ids:[B,L], snr:[B,1]) -> logits[B,L,V]`
  - `greedy_decode(...) -> ids[B,L]`
  - [论文明确写出] 3 层、8 heads、dim=128，与 NAM 交替。
  - [为复现做的合理实现选择] positional embedding、dropout、shifted teacher forcing。
- `models/channel_codec.py`
  - `ChannelEncoder.forward(semantic_features:[B,L,128], snr:[B,1]) -> symbols[B,L,C]`
  - `ChannelDecoder.forward(received_symbols:[B,L,C], snr:[B,1]) -> [B,L,128]`
  - `power_normalize(symbols) -> symbols`
  - [论文明确写出] hidden=256,128 + NAM；decoder 对称反向。
  - [为复现做的合理实现选择] `C` 的取值与归一化方案。
- `models/nam.py`
  - `forward(r:[B,1]|[B], G:[B,D]|[B,L,D]) -> A(same shape)`
  - [论文明确写出] FF 宽度 56,128,56,56，按公式做投影与缩放。
  - [为复现做的合理实现选择] 当 `G=[B,L,D]` 时广播策略。
- `models/med.py`
  - `add_to_stm(sample, feature)`
  - `is_stm_full() -> bool`
  - `compute_rbf(stm_feat, ltm_feat) -> float`
  - `select_representatives() -> List[idx]`
  - `transfer_to_ltm() -> int`
  - `sample_train_batch(...) -> Dict`
  - [论文明确写出] STM=500, $\tau=10$, $\lambda=0.05$。
  - [为复现做的合理实现选择] LTM 是否上限、STM/LTM 混合采样比。

## 2.5 信道与损失
- `channels/awgn.py`
  - `awgn_channel(symbols:[B,L,C], snr_db:float, training_mode:bool, seed:int) -> [B,L,C]`
- `channels/rayleigh.py`
  - `rayleigh_channel(symbols:[B,L,C], snr_db:float, training_mode:bool, seed:int) -> [B,L,C]`
- `losses/text_ce.py`
  - `text_cross_entropy(logits:[B,L,V], targets:[B,L], mask:[B,L]) -> scalar`
- `losses/mutual_info_proxy.py`
  - `channel_proxy_loss(decoded_feat, original_feat, encoded_symbols) -> scalar`
  - [为复现做的合理实现选择] surrogate loss，必须在文档声明“不等同论文原式”。

## 2.6 训练与评估入口
- `trainers/trainer_channel.py`：Stage A。
- `trainers/trainer_semantic.py`：Stage B。
- `trainers/trainer_joint.py`：Stage C。
- `scripts/prepare_text_cache.py`：生成 caption cache。
- `scripts/train_all.py`：统一触发 Stage A/B/C（formal/smoke 模式）。
- `scripts/run_fig7.py`~`scripts/run_fig10.py`：图实验入口。
- `eval/eval_fig7.py`~`eval/eval_fig10.py`：指标计算与结果落盘。

---

## 3. 公式到代码实现含义

1. 发送端映射
$$
y = C_{\beta}(S_{\alpha}(K_{\theta}(x),\mu),\mu)
$$
- 代码含义：`kb_blip.generate_caption_tokens` + `semantic_encoder` + `channel_encoder`。

2. 信道映射
$$
\hat{y}=h\cdot y+n
$$
- 代码含义：`awgn_channel` 或 `rayleigh_channel`。

3. 接收端重建
$$
\hat{x}=K^{-1}_{\theta'}(S^{-1}_{\delta}(C^{-1}_{\gamma}(\hat{y},\mu),\mu))
$$
- 代码含义：`channel_decoder` -> `semantic_decoder` -> `kb_sd.reconstruct_from_text`。

4. CE 目标
$$
L_{CE}(s,\hat{s})
$$
- 代码含义：`text_cross_entropy(logits, targets, mask)`，目标 token 与恢复 token 对齐。

5. NAM 校准
$$
v'=ReLU(W_{n2}ReLU(W_{n1}r+b_{n1})+b_{n2}),\quad
v=Sigmoid(W_{n3}v'+b_{n3})
$$
$$
e=W_{n4}G+b_{n4},\quad K=Sigmoid(e\cdot v),\quad A_i=K_i\cdot G_i
$$
- 代码含义：SNR 分支输出门控向量，对特征逐元素缩放。

6. MED 相似度
$$
RBF=\exp\left(-\frac{\|v_i^{stm}-v_j^{ltm}\|^2}{2\tau^2}\right),\quad
R(s_i^{stm})=\frac{1}{n_{ltm}}\sum_k RBF(\cdot)
$$
- 代码含义：`compute_rbf` + `select_representatives` + `transfer_to_ltm`。

---

## 4. 训练流程与评估流程

## 4.1 训练流程（formal）
1. 运行 `prepare_text_cache.py` 生成缓存（CIFAR/BIRDS/CATSvsDOGS）。
2. Stage A：仅更新 channel 分支（+NAM），冻结 semantic 分支。
3. Stage B：仅更新 semantic 分支（+NAM），冻结 channel 分支。
4. Stage C：A/B 交替迭代，记录验证指标并早停。
5. 每阶段保存 checkpoint、配置快照、指标 JSON/CSV。

## 4.2 评估流程
- Fig.8：MED on/off，Rayleigh，BLEU-1/BLEU-2，输出 continual map。
- Fig.9：NAM on/off，训练 SNR 规则按事实包执行，测试 `0..10 dB`。
- Fig.7：BLIP/LEMON/RAM + SD + AWGN + CATSvsDOGS + SSQ（LEMON unresolved 时明确标注）。
- Fig.10：VLM-CSC vs JSCC/WITT，输出 SSQ/分类、压缩率、参数量、语义对齐图。

---

## 5. 缺失信息清单（必须入 unresolved.md）

以下全部属于 **[为复现做的合理实现选择]**：
1. BLIP 精确 checkpoint 与 license 细节。
2. SD 精确版本、采样器、steps、guidance。
3. tokenizer/vocab/max length。
4. semantic/channel 精确 tensor shape 与 symbol 维度。
5. MI 目标的可计算替代（surrogate）与权重。
6. optimizer/lr/batch/epoch/scheduler/stopping。
7. 数据预处理与 continual split 细则。
8. Fig.10 压缩率定义、下游 classifier 结构。
9. LEMON 的可验证公开权重来源。

---

## 6. 8 步编码执行计划（每步含验收）

### Step 1：锁定正式模式边界
- 任务：所有正式脚本禁 fallback；smoke 允许 fallback 且打印固定标识。
- 验收：formal 路径若模型/缓存缺失必须报错并给明确提示。

### Step 2：打通 BLIP/SD 真接入
- 任务：`kb_blip.py`、`kb_sd.py` 完成真实加载与最小推理。
- 验收：`verify_blip.py`、`verify_sd.py` 成功产出结果。

### Step 3：固化 text cache
- 任务：`prepare_text_cache.py` 支持三数据集并落 JSONL。
- 验收：每个数据集都有缓存文件，含 `sample_id/image_path/caption/token_ids`。

### Step 4：校准 semantic + channel 闭环
- 任务：semantic/channel/NAM 信道前向全链路对齐。
- 验收：关键 tensor shape 全通过，CE 可反传。

### Step 5：完成 MED 训练接入
- 任务：STM/LTM 转移 + 采样接入训练器。
- 验收：STM 满后触发转移，训练 batch 可混采 STM/LTM。

### Step 6：执行 Stage A/B/C
- 任务：`train_all.py` formal 模式完成 A/B/C 与 checkpoint 输出。
- 验收：三阶段均有日志；C 阶段至少 1 个完整交替周期。

### Step 7：先复现 Fig.8/Fig.9
- 任务：按事实包配置完成 MED/NAM 两组实验。
- 验收：输出结果表 + 曲线数据；趋势检查通过。

### Step 8：最后处理 Fig.7/Fig.10
- 任务：BLIP/RAM 真接入；LEMON unresolved 显式输出；Fig.10 baseline 若未全接通标 scaffold。
- 验收：不冒充完成；所有 unresolved 明确写入文档。

---

## 7. 最终可直接开始编码的任务单

1. 先运行：`python scripts/verify_blip.py`、`python scripts/verify_sd.py`。
2. 生成缓存：`python scripts/prepare_text_cache.py --datasets cifar birds catsvsdogs --data-root ../data`。
3. formal 训练：`python scripts/train_all.py --mode formal --stage all --cache-file outputs/cache/captions/cifar_train.jsonl`。
4. 逐图执行：`python scripts/run_fig8.py`、`python scripts/run_fig9.py`。
5. 再做：`python scripts/run_fig7.py`、`python scripts/run_fig10.py`。
6. 写回：`unresolved.md`（未决项）与 `reproduction_report.md`（复现边界+结果）。

> 规则重申：凡非事实包明示内容，一律在实现注释与报告中标注“为复现做的合理实现选择”。
