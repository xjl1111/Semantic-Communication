# VLM-CSC 预训练模型核对与工程接入审计

本文档按“论文原文优先 + 一方来源可验证 + 工程可执行”原则编写。

---

## 0) 论文事实固定结论

### 论文明确写出

A. 发送端 CKB 使用 BLIP 做图像→文本语义提取；接收端 CKB 使用 SD 做文本→图像重建；BLIP 与 SD 作为预训练 VLM，不为 CSC 系统重新训练。  
B. Fig.7 发送端对比模型为 BLIP、LEMON、RAM；接收端统一使用 SD。  
C. BLIP 侧结构：image encoder + image-grounded text decoder，image encoder 基于 ViT，text decoder 为 BERT 风格；SD 侧结构：text encoder + denoising U-Net + image decoder。  
D. 论文实验设置给出了模型族与环境，但**没有**给出精确 checkpoint 名称/版本号/下载地址。

### 为复现做的合理实现选择

- 具体 checkpoint、版本号、下载地址需要工程侧补充并显式标注。

---

## 1) 四类模型核对表

## 1.1 BLIP（主系统发送端）

1. 论文中的名称：BLIP  
2. 论文中的角色：发送端 CKB（图像→文本语义）  
3. 论文是否明确给出 checkpoint 名称：否  
4. 论文是否明确给出 version/backbone：给出“BLIP + ViT image encoder + text decoder”层面描述；未给 checkpoint 版本号  
5. 一方公开来源是否找到：是  
   - 官方 GitHub: https://github.com/salesforce/BLIP  
   - 官方 HF: https://huggingface.co/Salesforce/blip-image-captioning-base  
6. 推荐 checkpoint 名称：`Salesforce/blip-image-captioning-base`  
7. checkpoint 归类：**为复现做的合理实现选择**（非论文明示）  
8. 许可证：HF 模型卡显示 `bsd-3-clause`  
9. 是否允许研究复现：是（研究可用；需遵守 license 与模型卡限制）  
10. 不确定点：论文未给精确 BLIP checkpoint；官方 BLIP 仓库已归档，建议优先采用 HF 官方模型卡版本

## 1.2 Stable Diffusion / SD（主系统接收端）

1. 论文中的名称：SD / Stable Diffusion  
2. 论文中的角色：接收端 CKB（文本→图像重建）  
3. 论文是否明确给出 checkpoint 名称：否  
4. 论文是否明确给出 version/backbone：给出 text encoder + U-Net + image decoder 组件级描述；未给精确版本  
5. 一方公开来源是否找到：是（HF 可验证）  
   - HF: https://huggingface.co/sd-legacy/stable-diffusion-v1-5  
6. 推荐 checkpoint 名称：`sd-legacy/stable-diffusion-v1-5`  
7. checkpoint 归类：**为复现做的合理实现选择**（非论文明示）  
8. 许可证：`creativeml-openrail-m`  
9. 是否允许研究复现：是（研究可用，但有 OpenRAIL-M 使用约束）  
10. 不确定点：该仓库模型卡写明是旧 `runwayml/...` 的 mirror；论文未指定精确发行方与 hash

## 1.3 RAM（Fig.7 对比）

1. 论文中的名称：RAM  
2. 论文中的角色：Fig.7 sender-side baseline  
3. 论文是否明确给出 checkpoint 名称：否  
4. 论文是否明确给出 version/backbone：未给 checkpoint；官方仓库 README 提供 Swin-Large 14M 权重命名  
5. 一方公开来源是否找到：是  
   - 官方 GitHub: https://github.com/xinyu1205/recognize-anything  
6. 推荐 checkpoint 名称：`ram_swin_large_14m.pth`（来自官方 README“Checkpoints/Inference”）  
7. checkpoint 归类：**为复现做的合理实现选择**（非论文明示）  
8. 许可证：Apache-2.0（仓库 About）  
9. 是否允许研究复现：是（遵守 Apache-2.0）  
10. 不确定点：论文未指定是 RAM 还是 RAM++ 及具体权重版本；建议先按 RAM baseline 名称接入

## 1.4 LEMON（Fig.7 对比）

1. 论文中的名称：LEMON  
2. 论文中的角色：Fig.7 sender-side baseline  
3. 论文是否明确给出 checkpoint 名称：否  
4. 论文是否明确给出 version/backbone：否  
5. 一方公开来源是否找到：**未确认**（当前未获得可验证的一方 checkpoint 证据）  
6. 推荐 checkpoint 名称：无  
7. checkpoint 归类：无  
8. 许可证：未知  
9. 是否允许研究复现：未知（因可验证来源未确认）  
10. 不确定点：
    - **LEMON baseline unresolved: 当前未找到可验证的一方公开 checkpoint，禁止静默替代**

---

## 2) 主系统必须模型锁定

### BLIP

- 推荐：`Salesforce/blip-image-captioning-base`  
- 归类：**为复现做的合理实现选择**  
- 理由：官方 image captioning checkpoint，且模型卡声明 base 架构（ViT-base backbone），与论文 BLIP 结构描述最贴近。

### SD

- 推荐：`sd-legacy/stable-diffusion-v1-5`  
- 归类：**为复现做的合理实现选择**  
- 理由：公开稳定、Diffusers 接入成熟、组件结构与论文 text encoder + U-Net + image decoder 描述一致。

---

## 3) Fig.7 对比模型严格说明

### RAM

- 来源：官方 recognize-anything 仓库 README。  
- 推荐权重：`ram_swin_large_14m.pth`。  
- 推理方式：优先使用官方 `inference_ram.py --pretrained pretrained/ram_swin_large_14m.pth`。  
- 约束：RAM 仅用于 Fig.7 sender baseline，**不得替代主系统 BLIP**。

### LEMON

- 当前状态：未获得可验证的一方公开 checkpoint。  
- 结论：
  - **LEMON baseline unresolved: 当前未找到可验证的一方公开 checkpoint，禁止静默替代。**

---

## 4) 可执行下载与加载方案

## 4.1 依赖安装命令

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers diffusers accelerate safetensors pillow huggingface_hub
pip install sentencepiece protobuf
```

RAM（官方仓库方式）:

```bash
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

## 4.2 模型缓存目录建议

- 建议统一：`D:/model_cache/vlm_csc/`
- 子目录建议：
  - `D:/model_cache/vlm_csc/blip/`
  - `D:/model_cache/vlm_csc/sd/`
  - `D:/model_cache/vlm_csc/ram/`

## 4.3 本地目录命名建议

- BLIP: `blip-image-captioning-base`
- SD: `stable-diffusion-v1-5`
- RAM: `ram_swin_large_14m.pth`

## 4.4 Python 加载代码（可执行）

### BLIP

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

model_name = "Salesforce/blip-image-captioning-base"
cache_dir = "D:/model_cache/vlm_csc/blip"

processor = BlipProcessor.from_pretrained(model_name, cache_dir=cache_dir)
model = BlipForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
```

### SD

```python
import torch
from diffusers import StableDiffusionPipeline

model_name = "sd-legacy/stable-diffusion-v1-5"
cache_dir = "D:/model_cache/vlm_csc/sd"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=dtype,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
```

### RAM（官方脚本方式）

```python
import subprocess
import sys

cmd = [
    sys.executable,
    "inference_ram.py",
    "--image", "images/demo/demo1.jpg",
    "--pretrained", "pretrained/ram_swin_large_14m.pth",
]
subprocess.run(cmd, check=True)
```

## 4.5 失败时报错处理策略

- `from_pretrained` 失败：
  - 打印模型名、缓存目录、异常类型、异常消息；
  - 明确提示网络、权限、token、磁盘空间；
  - 正式实验脚本直接 `raise RuntimeError`，禁止 fallback。
- RAM 权重缺失：
  - 明确报 `FileNotFoundError`，并提示官方 README checkpoint 名称。

## 4.6 license 文件保存建议

- 每个模型下载后保存：
  - 模型卡链接
  - LICENSE 文本
  - 版本/commit/hash
- 建议目录：`third_party_licenses/` 下按模型分目录保存。

---

## 5) 输入输出接口规范

## 5.1 BLIP（对接 `models/kb_blip.py`）

- 输入类型：`PIL.Image.Image` 或 RGB 图像张量  
- 输入数据形式：单张图或 batch 图像  
- 预处理器：`BlipProcessor`  
- 输出类型：
  - `caption: str`
  - `input_ids: torch.LongTensor`  
- 输出形式：
  - captions: `List[str]`
  - token ids: `[B, T]`
- 对接函数：
  - `load_model()`
  - `generate_caption(image)`
  - `generate_caption_tokens(image)`

## 5.2 SD（对接 `models/kb_sd.py`）

- 输入类型：`decoded text string`  
- 输入数据形式：`List[str]` prompts  
- 预处理器：SD pipeline 内部 tokenizer + text encoder  
- 输出类型：`PIL.Image` 或图像 tensor  
- 输出形式：`List[PIL.Image]` / `[B,3,H,W]`  
- 对接函数：
  - `load_pipeline()`
  - `reconstruct_image(prompt)`
  - `expose_components()`（可选）返回 `text_encoder / unet / vae`

## 5.3 RAM（仅 Fig.7 baseline）

- 输入类型：image path 或图像  
- 输出类型：tag string（或 tags list）  
- 输出形式：`str` / `List[str]`  
- 对接位置：`models/kb_alt_vlm.py`（仅 baseline 分支）

---

## 6) 工程集成修改方案（文件级/函数级）

以下为**明确修改方案**（非论文原文）：

1. `configs/default.yaml` 新增字段：
   - `blip.model_name`
   - `sd.model_name`
   - `ram.checkpoint_path`
   - `model_cache_dir`
   - `use_fp16`
   - `disable_fallback_in_formal_experiments`

2. `models/kb_blip.py`：
   - `load_model()`
   - `generate_caption(image)`
   - `generate_caption_tokens(image)`

3. `models/kb_sd.py`：
   - `load_pipeline()`
   - `reconstruct_image(prompt)`
   - `expose_components()`（optional）

4. `models/kb_alt_vlm.py`：
   - RAM baseline loading（真实权重）
   - LEMON baseline state reporting（unresolved 明确上报）

5. 正式实验脚本：
   - 禁止 fallback；仅 `smoke_test` 允许 mock/fallback

6. 权重加载失败策略：
   - 正式实验直接报错退出（禁止 silent fallback）

---

## 7) 验证脚本交付

已生成脚本：

- `scripts/verify_blip.py`
- `scripts/verify_sd.py`
- `scripts/verify_ram.py`

脚本行为：

- 每个脚本独立可运行
- 打印：模型名、设备、成功标志、输出摘要
- 出错时打印可诊断信息

---

## 8) 不确定项清单

- 论文未给 BLIP 精确 checkpoint。
- 论文未给 SD 精确版本。
- LEMON 是否存在可验证一方 checkpoint：当前未确认。
- 论文未给逐模型显存下限与推理参数（steps/guidance）。

---

## 9) 最终结论

A. 主系统立即可接入模型：BLIP、SD  
B. 仅能做 baseline 的模型：RAM（Fig.7 sender baseline）  
C. 当前无法确认的一方模型：LEMON（checkpoint unresolved）  
D. 许可证与复现可用性：
- BLIP: BSD-3-Clause（研究复现可用）
- SD v1.5: CreativeML OpenRAIL-M（研究可用，受条款约束）
- RAM: Apache-2.0（研究复现可用）
- LEMON: 未确认
E. 集成优先级：
1) BLIP 主链路 -> 2) SD 主链路 -> 3) smoke test -> 4) formal strict mode -> 5) Fig.7 baseline 扩展（RAM）-> 6) LEMON unresolved 说明

---

## 10) 实施清单（开始大改前）

1. 先做 BLIP 主链路接入  
2. 再做 SD 主链路接入  
3. 再做 smoke test  
4. 再做 formal experiment strict mode  
5. 再做 Fig.7 baseline 扩展  
6. 最后补充 unresolved baseline 说明
