# paper_facts.md

本文档仅记录论文中“明确写出”的事实，禁止掺入推断实现。

## A. 系统总流程（论文明确写出）

- 发射端输入图像：$x \in \mathbb{R}^{H \times W \times C}$。
- 发射端三部分：CKB、semantic encoder、channel encoder。
- 发送符号：
  $$
  y = C_\beta\big(S_\alpha(K_\theta(x), \mu), \mu\big)
  $$
- 信道模型：
  $$
  \hat{y} = h \cdot y + n
  $$
  其中 $n$ 为 AWGN。
- 接收端三部分：channel decoder、semantic decoder、cross-modal knowledge base for reconstruction。
- 重建图像：
  $$
  \hat{x} = K_{\theta'}^{-1}\Big(S_\delta^{-1}(C_\gamma^{-1}(\hat{y}, \mu), \mu)\Big)
  $$
- 文本一致性损失（CE）：
  $$
  L_{CE}(s,\hat{s}) = -\sum_{l=1}^{L} q(w_l)\log p(w_i) + (1-q(w_l))\log(1-p(w_i))
  $$
- 目标函数：
  $$
  (\alpha^*,\beta^*,\delta^*,\gamma^*) = \arg\min_{\alpha,\beta,\delta,\gamma}
  \mathbb{E}_{p(\mu)}\mathbb{E}_{p(s,\hat{s})}[L_{CE}(s,\hat{s})]
  $$

## B. 三个核心贡献（论文明确写出）

- CKB：发射端用 BLIP 提取文本语义；接收端用 SD 根据文本重建图像。
- MED：STM + LTM 混合记忆缓解灾难性遗忘。
- NAM：基于 SNR 的注意力模块，动态调整 semantic/channel coding 权重。

## C. 发射端 CKB（BLIP）结构（论文明确写出）

- BLIP: image encoder + image-grounded text decoder。
- image encoder 基于 ViT，patch + MSA + FF。
- 第 1 层 image encoder：
  $$m_{msa,1} = MSA(LN(x_p)) + x_p$$
  $$m_{ff,1} = GeLU(W_{b,f} \cdot LN(m_{msa,1}) + b_{b,f}) + m_{msa,1}$$
- 第 $L$ 层输出：
  $$m_L = LN(m_{ff,L})$$
- text decoder 由 CSA、CA、FF 组成。
- 第 1 层 text decoder：
  $$k_{csa,1} = CSA(LN(D_0)) + D_0$$
  $$k_{ca,1} = CA(LN(k_{csa,1}), m_L) + k_{csa,1}$$
  $$k_{ff,1} = ReLU(W'_{b,f} \cdot LN(k_{ca,1}) + b'_{b,f}) + k_{ca,1}$$
- 最终输出文本描述 $s$。

## D. 接收端 CKB（SD）结构（论文明确写出）

- SD: text encoder + denoising U-Net + image decoder(VAE decoder)。
- text encoder 输出固定维度语义向量作为 diffusion 条件。
- diffusion 迭代：
  $$
  Z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(Z_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} f_\theta(Z_t, t, d)\right) + \sigma_t Y
  $$

## E. MED（论文明确写出）

- 同时维护 STM 与 LTM。
- STM 满时，计算 STM 与 LTM 核距离并选择样本转移。
- RBF 核：
  $$
  RBF(s_i^{stm}, s_j^{ltm}) = \exp\left(-\frac{\|v_i^{stm}-v_j^{ltm}\|^2}{2\tau^2}\right)
  $$
- 论文给出 $\tau = 10$。
- 平均相似度：
  $$
  R(s_i^{stm}) = \frac{1}{n_{ltm}}\sum_{k=1}^{n_{ltm}} RBF(s_i^{stm}, s_k^{ltm})
  $$
- 转移规则：
  $$
  R(s_i^{stm}) > \lambda \Rightarrow M_{ltm} = M_{ltm} \cup s_i^{stm}
  $$
- STM 最大样本数 500；sample selection threshold 0.05。

## F. NAM（论文明确写出）

- SNR projection:
  $$v' = ReLU(W_{n2}\cdot ReLU(W_{n1}\cdot r + b_{n1}) + b_{n2})$$
  $$v = Sigmoid(W_{n3}\cdot v' + b_{n3})$$
- feature scaling:
  $$K = Sigmoid(e \cdot v),\quad e = W_{n4}\cdot G + b_{n4},\quad A_i = K_i\cdot G_i$$
- NAM 4 层 FF 神经元数：56, 128, 56, 56。

## G. 训练/实验设置（论文明确写出）

- semantic encoder：3 个 transformer encoder layers，与 NAM 交替。
- 每个 transformer layer：8 heads，feature dim=128。
- channel encoder：2 个 FF hidden layers，与 NAM 交替；hidden size=256, 128。
- semantic decoder/channel decoder：与 encoder 对称反向。
- BLIP 预训练参数量：129MB。
- SD 预训练参数量：1.99GB。
- STM 最大样本数：500。
- sample selection threshold：0.05。
- 环境：Windows Server 2016, Python 3.8, PyTorch 1.8.0, CUDA 11.6, Xeon Silver 4210R, Tesla T4。

## H. 评估指标（论文明确写出）

- Image-level: SSQ
  $$
  SSQ = \frac{ST(\hat{S})}{ST(S)}
  $$
  其中 $ST(\cdot)$ 为 classification accuracy。
- Text-level: BLEU
  $$
  \log BLEU = \min(1 - \frac{\hat{l}_s}{l_s}, 0) + \sum_{n=1}^{N} u_n \log p_n
  $$
  $$
  p_n = \frac{\sum_k \min(C_k(\hat{s}), C_k(s))}{\sum_k \min(C_k(\hat{s}))}
  $$

## I. 数据集与实验任务（论文明确写出）

- 数据集：CIFAR、BIRDS、CATSvsDOGS。
- Fig.7：sender-side KB (BLIP/LEMON/RAM) 对比 SSQ；receiver-side KB 固定 SD；AWGN；CATSvsDOGS。
- Fig.8：有无 MED；continual learning map；CIFAR、BIRDS、CATSvsDOGS；BLEU；Rayleigh。
- Fig.9：有无 NAM；NAM-on 在 $SNR_{train}\in[0,10]$ 均匀训练；NAM-off 在 0/2/4/8 dB 训练；测试 $SNR_{test}\in[0,10]$；BLEU。
- Fig.10：VLM-CSC vs JSCC(CNN) vs WITT(ViT)；分类性能、compression ratio、trainable parameters、semantic alignment。
