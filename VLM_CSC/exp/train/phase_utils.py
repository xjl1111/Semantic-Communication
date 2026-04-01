from __future__ import annotations

from typing import Dict, List


def freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_module(module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def _unfreeze_all_nam_params(model) -> None:
    """解冻模型中所有 NAM 参数 (仅 joint phase 使用)。"""
    for name, p in model.named_parameters():
        if "nam" in name.lower():
            p.requires_grad = True


def _unfreeze_channel_side_nam_params(model) -> None:
    """仅解冻 channel_encoder / channel_decoder 中的 NAM 参数。

    论文 Phase 1: 训练 channel enc/dec + 其 NAM，语义侧完全冻结。
    """
    for name, p in model.named_parameters():
        if "nam" in name.lower() and (
            name.startswith("channel_encoder.") or name.startswith("channel_decoder.")
        ):
            p.requires_grad = True


def _unfreeze_semantic_side_nam_params(model) -> None:
    """仅解冻 semantic_encoder / semantic_decoder 中的 NAM 参数。

    论文 Phase 2: 训练 semantic enc/dec + 其 NAM，信道侧完全冻结。
    """
    for name, p in model.named_parameters():
        if "nam" in name.lower() and (
            name.startswith("semantic_encoder.") or name.startswith("semantic_decoder.")
        ):
            p.requires_grad = True


def set_trainable_for_channel_phase(model) -> None:
    freeze_all(model)
    unfreeze_module(model.channel_encoder)
    unfreeze_module(model.channel_decoder)
    # 只解冻信道侧 NAM，不动语义侧 NAM
    _unfreeze_channel_side_nam_params(model)


def set_trainable_for_semantic_phase(model) -> None:
    freeze_all(model)
    unfreeze_module(model.embedding)
    unfreeze_module(model.semantic_encoder)
    unfreeze_module(model.semantic_decoder)
    unfreeze_module(model.lm_head)
    # 只解冻语义侧 NAM，不动信道侧 NAM
    _unfreeze_semantic_side_nam_params(model)


def set_trainable_for_joint_channel_step(model) -> None:
    """Joint phase 信道侧 step: 解冻信道 enc/dec + 全部 NAM。

    论文 Phase 3 (crossover): 交替训练信道侧和语义侧。
    因为是联合微调阶段，所有 NAM 都需要一起适应 SNR。
    """
    freeze_all(model)
    unfreeze_module(model.channel_encoder)
    unfreeze_module(model.channel_decoder)
    _unfreeze_all_nam_params(model)


def set_trainable_for_joint_semantic_step(model) -> None:
    """Joint phase 语义侧 step: 解冻语义 enc/dec + embedding + lm_head + 全部 NAM。

    论文 Phase 3 (crossover): 交替训练信道侧和语义侧。
    因为是联合微调阶段，所有 NAM 都需要一起适应 SNR。
    """
    freeze_all(model)
    unfreeze_module(model.embedding)
    unfreeze_module(model.semantic_encoder)
    unfreeze_module(model.semantic_decoder)
    unfreeze_module(model.lm_head)
    _unfreeze_all_nam_params(model)


def collect_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def masked_sequence_mse(recovered_seq, semantic_seq_detached, padding_mask):
    """Channel phase 重建损失 — **工程近似** (Engineering Approximation)。

    论文 §III-C 指出 channel phase 应最小化互信息(mutual information)相关目标，
    但未给出精确的 MI 损失公式。本实现使用 masked MSE 作为近似替代：
      L = Σ (recovered - teacher)^2 * valid_mask / Σ valid_mask

    ⚠ 严格复现阻断项 (BLOCKER):
      论文未提供 MI loss 的精确数学形式，因此 masked_sequence_mse 是工程选择，
      而非论文原文公式。正式复现报告中须标注此项为 "工程近似"。
    """
    valid = (~padding_mask).unsqueeze(-1).float()
    diff = (recovered_seq - semantic_seq_detached) ** 2
    return (diff * valid).sum() / valid.sum().clamp_min(1.0)


def assert_channel_forward_contract(out: Dict[str, object]) -> None:
    required_keys = ["semantic_seq_teacher", "recovered_seq", "padding_mask"]
    missing = [k for k in required_keys if k not in out]
    if missing:
        raise RuntimeError(f"Channel phase forward missing required keys: {missing}")


def compute_info_nce_sequence(channel_out_list: List[Dict[str, object]], temperature: float):
    import torch
    import torch.nn.functional as F

    if temperature <= 0:
        raise RuntimeError(f"info_nce_sequence requires temperature > 0, got {temperature}")
    if len(channel_out_list) < 2:
        raise RuntimeError(
            "info_nce_sequence requires at least 2 samples in a batch to provide negatives. "
            "Increase train_batch_size or reduce filtering."
        )

    recovered_repr_list = []
    teacher_repr_list = []
    for out in channel_out_list:
        assert_channel_forward_contract(out)
        recovered_seq = out["recovered_seq"]
        teacher_seq = out["semantic_seq_teacher"]
        padding_mask = out["padding_mask"]

        valid = (~padding_mask).unsqueeze(-1).float()
        recovered_repr = (recovered_seq * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        teacher_repr = (teacher_seq * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        recovered_repr_list.append(recovered_repr)
        teacher_repr_list.append(teacher_repr)

    recovered_repr = torch.cat(recovered_repr_list, dim=0)
    teacher_repr = torch.cat(teacher_repr_list, dim=0)

    recovered_repr = F.normalize(recovered_repr, dim=-1)
    teacher_repr = F.normalize(teacher_repr, dim=-1)

    logits = recovered_repr @ teacher_repr.t()
    logits = logits / float(temperature)
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i2t + loss_t2i)


def assert_semantic_forward_contract(out: Dict[str, object]) -> None:
    logits = out.get("logits")
    target_ids = out.get("target_ids")
    if logits is None or target_ids is None:
        raise RuntimeError("Semantic phase forward must return logits and target_ids")
    if tuple(logits.shape[:2]) != tuple(target_ids.shape[:2]):
        raise RuntimeError(f"Semantic phase shape mismatch: logits[:2]={tuple(logits.shape[:2])}, target_ids={tuple(target_ids.shape[:2])}")
    if bool(out.get("used_shift_right", False)) is not True:
        raise RuntimeError("Semantic phase strict check failed: used_shift_right must be True")
    if bool(out.get("used_causal_mask", False)) is not True:
        raise RuntimeError("Semantic phase strict check failed: used_causal_mask must be True")


def require_phase_block(phase_cfg: Dict, phase_name: str, required_keys: List[str]) -> Dict:
    if phase_name not in phase_cfg:
        raise RuntimeError(f"Missing train phase config block: {phase_name}")
    block = phase_cfg[phase_name]
    if not isinstance(block, dict):
        raise RuntimeError(f"Phase config must be dict for {phase_name}")
    missing = [k for k in required_keys if k not in block]
    if missing:
        raise RuntimeError(f"Phase config missing keys for {phase_name}: {missing}")
    return block
