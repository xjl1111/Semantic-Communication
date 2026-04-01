"""三阶段训练：信道阶段、语义阶段、联合阶段。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

from train.helpers import (
    _resolve_train_snr,
    _prepare_merged_batch,
    _update_med_and_check,
    _log_med_state,
    _run_semantic_train_step_merged,
    _run_joint_train_step_merged,
    _train_text_matches_label,
)
from train.phase_utils import (
    assert_channel_forward_contract as _assert_channel_forward_contract,
    assert_semantic_forward_contract as _assert_semantic_forward_contract,
    collect_trainable_params as _collect_trainable_params,
    compute_info_nce_sequence as _compute_info_nce_sequence,
    masked_sequence_mse as _masked_sequence_mse,
    set_trainable_for_channel_phase,
    set_trainable_for_semantic_phase,
    set_trainable_for_joint_channel_step,
    set_trainable_for_joint_semantic_step,
)


def run_channel_phase(
    *,
    model,
    sender: str,
    train_batches,
    val_batches,
    caption_cache: Dict[str, str],
    rng: random.Random,
    phase_cfg: Dict,
    snr_min_db: float,
    snr_max_db: float,
    snr_train_mode: str,
    checkpoint_path: Path,
):
    import torch

    objective = str(phase_cfg["channel_phase_objective"]).lower()
    if objective not in {"masked_sequence_mse", "info_nce_sequence"}:
        raise RuntimeError(f"Unsupported channel_phase_objective: {objective}")
    # ⚠ 工程近似标注 (Engineering Approximation):
    # 论文 §III-C 的 channel phase 目标是互信息(MI)最小化，但论文未给出
    # 精确 MI loss 公式。masked_sequence_mse 和 info_nce_sequence 均为
    # 工程替代方案，正式复现报告中须标注为 "工程近似" 而非 "论文公式"。
    info_nce_temperature = float(phase_cfg.get("info_nce_temperature", 0.07))

    set_trainable_for_channel_phase(model)
    optimizer = torch.optim.AdamW(
        _collect_trainable_params(model),
        lr=float(phase_cfg["lr"]),
        weight_decay=float(phase_cfg["weight_decay"]),
    )

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stop_patience = int(phase_cfg.get("early_stop_patience", 0))
    epochs = int(phase_cfg["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for batch in tqdm(train_batches, desc=f"train channel ({sender})", leave=False):
            optimizer.zero_grad(set_to_none=True)
            channel_out_list = []
            for rec in batch:
                image_path = rec["path"]
                image = Image.open(image_path).convert("RGB")
                source_text = caption_cache.get(str(image_path))
                snr_db = _resolve_train_snr(snr_train_mode, snr_min_db, snr_max_db, rng)
                out = model.forward_channel_phase(image=image, snr_db=snr_db, source_text=source_text)
                _assert_channel_forward_contract(out)
                channel_out_list.append(out)

            if objective == "masked_sequence_mse":
                sample_losses = [
                    _masked_sequence_mse(out["recovered_seq"], out["semantic_seq_teacher"], out["padding_mask"])
                    for out in channel_out_list
                ]
                batch_loss = torch.stack(sample_losses).mean()
            elif objective == "info_nce_sequence":
                batch_loss = _compute_info_nce_sequence(channel_out_list=channel_out_list, temperature=info_nce_temperature)
            else:
                raise RuntimeError(f"Unsupported channel_phase_objective: {objective}")

            if not torch.isfinite(batch_loss):
                raise RuntimeError(f"Non-finite train channel loss sender={sender}, epoch={epoch}")
            train_loss_sum += float(batch_loss.item())
            train_steps += 1
            batch_loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_lv_a_correct = 0
        val_lv_a_total = 0
        with torch.no_grad():
            for batch in tqdm(val_batches, desc=f"val channel ({sender})", leave=False):
                channel_out_list = []
                for rec in batch:
                    image_path = rec["path"]
                    image = Image.open(image_path).convert("RGB")
                    source_text = caption_cache.get(str(image_path))
                    out = model.forward_channel_phase(image=image, snr_db=float((snr_min_db + snr_max_db) / 2.0), source_text=source_text)
                    _assert_channel_forward_contract(out)
                    channel_out_list.append(out)
                    # Level-A：源端文本是否包含正确类别词
                    _lbl = int(rec.get("label", -1))
                    if source_text is not None and _lbl in (0, 1):
                        if _train_text_matches_label(source_text, _lbl):
                            val_lv_a_correct += 1
                        val_lv_a_total += 1

                if objective == "masked_sequence_mse":
                    sample_losses = [
                        _masked_sequence_mse(out["recovered_seq"], out["semantic_seq_teacher"], out["padding_mask"])
                        for out in channel_out_list
                    ]
                    batch_loss = torch.stack(sample_losses).mean()
                elif objective == "info_nce_sequence":
                    batch_loss = _compute_info_nce_sequence(channel_out_list=channel_out_list, temperature=info_nce_temperature)
                else:
                    raise RuntimeError(f"Unsupported channel_phase_objective: {objective}")

                val_loss_sum += float(batch_loss.item())
                val_steps += 1

        val_loss = val_loss_sum / max(val_steps, 1)
        train_loss = train_loss_sum / max(train_steps, 1)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "phase": "channel",
                    "selected_by": "val",
                    "val_metric_name": "val_channel_loss",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        best_mark = " *" if is_best else ""
        lv_a_str = f"  A(src)={val_lv_a_correct/val_lv_a_total:.1%}" if val_lv_a_total > 0 else ""
        early_stop_info = f"  patience={stale_epochs}/{early_stop_patience}" if (not is_best and early_stop_patience > 0) else ""
        print(
            f"[channel][{sender}] epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"{lv_a_str}{best_mark}{early_stop_info}"
        )

        if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            print(f"[channel][{sender}] early stop at epoch {epoch} (patience={early_stop_patience})")
            break

    return {"checkpoint": str(checkpoint_path), "best_epoch": best_epoch, "best_val_loss": best_val}


def run_semantic_phase(
    *,
    model,
    sender: str,
    train_batches,
    val_batches,
    caption_cache: Dict[str, str],
    criterion,
    rng: random.Random,
    phase_cfg: Dict,
    snr_min_db: float,
    snr_max_db: float,
    snr_train_mode: str,
    checkpoint_path: Path,
    med_replay_enabled: bool = False,
    med_replay_batch_size: int = 4,
    med_replay_stm_ratio: float = 0.5,
    med_replay_weight: float = 1.0,
    dataset_id: str = "train",
    med_seen_keys: set[tuple[str, str]] | None = None,
):
    import torch

    if med_replay_enabled and model.med is None:
        raise RuntimeError("with_med semantic phase requires model.med to be initialized")
    if (not med_replay_enabled) and model.med is not None:
        raise RuntimeError("without_med semantic phase requires model.med to be None")

    set_trainable_for_semantic_phase(model)
    optimizer = torch.optim.AdamW(
        _collect_trainable_params(model),
        lr=float(phase_cfg["lr"]),
        weight_decay=float(phase_cfg["weight_decay"]),
    )

    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stop_patience = int(phase_cfg.get("early_stop_patience", 0))
    epochs = int(phase_cfg["epochs"])
    memory_used_any = False

    for epoch in range(1, epochs + 1):
        model.train()
        step_idx = 0
        train_loss_sum = 0.0
        train_steps = 0
        for batch in tqdm(train_batches, desc=f"train semantic ({sender})", leave=False):
            optimizer.zero_grad(set_to_none=True)
            snr_db = _resolve_train_snr(snr_train_mode, snr_min_db, snr_max_db, rng)
            merged_batch, current_batch, has_memory = _prepare_merged_batch(
                batch=batch, caption_cache=caption_cache, dataset_id=dataset_id,
                model=model, med_replay_enabled=med_replay_enabled,
                med_replay_batch_size=med_replay_batch_size,
                med_replay_stm_ratio=med_replay_stm_ratio,
                phase_label="semantic",
            )
            if has_memory:
                memory_used_any = True

            _, batch_loss = _run_semantic_train_step_merged(
                model=model, criterion=criterion, merged_batch=merged_batch, snr_db=snr_db,
            )
            if not torch.isfinite(batch_loss):
                raise RuntimeError(f"Non-finite train semantic loss sender={sender}, epoch={epoch}")
            train_loss_sum += float(batch_loss.item())
            train_steps += 1
            batch_loss.backward()
            optimizer.step()
            _update_med_and_check(
                model=model, med_replay_enabled=med_replay_enabled,
                current_batch=current_batch, med_seen_keys=med_seen_keys,
            )

        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        val_lv_a_correct = 0
        val_lv_b_correct = 0
        val_lv_ab_total = 0
        with torch.no_grad():
            for batch in tqdm(val_batches, desc=f"val semantic ({sender})", leave=False):
                sample_losses = []
                for rec in batch:
                    image_path = rec["path"]
                    image = Image.open(image_path).convert("RGB")
                    source_text = caption_cache.get(str(image_path))
                    out = model.forward_semantic_phase(image=image, snr_db=float((snr_min_db + snr_max_db) / 2.0), source_text=source_text)
                    _assert_semantic_forward_contract(out)
                    loss = criterion(out["logits"].reshape(-1, out["logits"].size(-1)), out["target_ids"].reshape(-1))
                    sample_losses.append(loss)
                    # Level-A/B
                    _lbl = int(rec.get("label", -1))
                    if _lbl in (0, 1):
                        src_txt = source_text or ""
                        if _train_text_matches_label(src_txt, _lbl):
                            val_lv_a_correct += 1
                        # Level-B: 兼容 [T,V] 和 [1,T,V]
                        _lg = out["logits"]
                        if _lg.dim() == 3:
                            _lg = _lg[0]  # 取第一个 sample -> [T, vocab]
                        pred_ids = _lg.argmax(dim=-1).unsqueeze(0)  # [1, T]
                        rec_txt = model.tokenizer.decode(pred_ids)[0]
                        if _train_text_matches_label(rec_txt, _lbl):
                            val_lv_b_correct += 1
                        val_lv_ab_total += 1
                batch_loss = torch.stack(sample_losses).mean()
                val_loss_sum += float(batch_loss.item())
                val_steps += 1

        val_loss = val_loss_sum / max(val_steps, 1)
        train_loss = train_loss_sum / max(train_steps, 1)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val,
                    "phase": "semantic",
                    "selected_by": "val",
                    "val_metric_name": "val_semantic_loss",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        best_mark = " *" if is_best else ""
        lv_ab_str = (
            f"  A={val_lv_a_correct/val_lv_ab_total:.1%}  B={val_lv_b_correct/val_lv_ab_total:.1%}"
            if val_lv_ab_total > 0 else ""
        )
        early_stop_info = f"  patience={stale_epochs}/{early_stop_patience}" if (not is_best and early_stop_patience > 0) else ""
        print(
            f"[semantic][{sender}] epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"{lv_ab_str}{best_mark}{early_stop_info}"
        )
        if med_replay_enabled:
            _log_med_state(model, phase="semantic", epoch=epoch)

        if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            print(f"[semantic][{sender}] early stop at epoch {epoch} (patience={early_stop_patience})")
            break

    if med_replay_enabled and int(med_replay_batch_size) > 0 and not memory_used_any:
        raise RuntimeError(
            "with_med semantic phase never consumed memory replay samples (memory_count stayed 0 for all steps)."
        )

    return {"checkpoint": str(checkpoint_path), "best_epoch": best_epoch, "best_val_loss": best_val}


def run_joint_phase(
    *,
    model,
    sender: str,
    train_batches,
    val_batches,
    caption_cache: Dict[str, str],
    criterion,
    rng: random.Random,
    phase_cfg: Dict,
    snr_min_db: float,
    snr_max_db: float,
    snr_train_mode: str,
    best_checkpoint_path: Path,
    last_checkpoint_path: Path,
    med_replay_enabled: bool = False,
    med_replay_batch_size: int = 4,
    med_replay_stm_ratio: float = 0.5,
    med_replay_weight: float = 1.0,
    dataset_id: str = "train",
    med_seen_keys: set[tuple[str, str]] | None = None,
):
    import torch

    if med_replay_enabled and model.med is None:
        raise RuntimeError("with_med joint phase requires model.med to be initialized")
    if (not med_replay_enabled) and model.med is not None:
        raise RuntimeError("without_med joint phase requires model.med to be None")

    alpha = float(phase_cfg["alpha"])
    beta = float(phase_cfg["beta"])
    schedule = str(phase_cfg["schedule"]).lower()
    if schedule not in {"alternate_steps", "alternate_epochs"}:
        raise RuntimeError(f"Unsupported joint schedule: {schedule}")

    max_joint_epochs = int(phase_cfg["max_joint_epochs"])
    early_stop_patience = int(phase_cfg["early_stop_patience"])
    monitor = str(phase_cfg["monitor"]).lower()
    if monitor not in {"val_semantic_loss", "joint_score"}:
        raise RuntimeError(f"Unsupported joint monitor: {monitor}")

    set_trainable_for_joint_channel_step(model)
    opt_channel = torch.optim.AdamW(_collect_trainable_params(model), lr=float(phase_cfg["lr"]), weight_decay=float(phase_cfg["weight_decay"]))
    set_trainable_for_joint_semantic_step(model)
    opt_semantic = torch.optim.AdamW(_collect_trainable_params(model), lr=float(phase_cfg["lr"]), weight_decay=float(phase_cfg["weight_decay"]))

    best_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    memory_used_any = False

    for epoch in range(1, max_joint_epochs + 1):
        model.train()
        step_idx = 0
        train_channel_sum = 0.0
        train_semantic_sum = 0.0
        train_total_sum = 0.0
        train_steps = 0
        for batch in tqdm(train_batches, desc=f"train joint ({sender})", leave=False):
            if schedule == "alternate_steps":
                do_channel = (global_step % 2 == 0)
            else:
                do_channel = (epoch % 2 == 1)

            if do_channel:
                set_trainable_for_joint_channel_step(model)
                optimizer = opt_channel
            else:
                set_trainable_for_joint_semantic_step(model)
                optimizer = opt_semantic

            optimizer.zero_grad(set_to_none=True)
            snr_db = _resolve_train_snr(snr_train_mode, snr_min_db, snr_max_db, rng)
            merged_batch, current_batch, has_memory = _prepare_merged_batch(
                batch=batch, caption_cache=caption_cache, dataset_id=dataset_id,
                model=model, med_replay_enabled=med_replay_enabled,
                med_replay_batch_size=med_replay_batch_size,
                med_replay_stm_ratio=med_replay_stm_ratio,
                phase_label="joint",
            )
            if has_memory:
                memory_used_any = True

            out = _run_joint_train_step_merged(
                model=model, criterion=criterion, merged_batch=merged_batch, snr_db=snr_db,
            )

            channel_loss = _masked_sequence_mse(out["recovered_seq"], out["semantic_seq_detached"], out["padding_mask"])
            semantic_loss = criterion(out["logits"].reshape(-1, out["logits"].size(-1)), out["target_ids"].reshape(-1))

            # 论文 crossover: 信道 step 只 backward 信道 loss，语义 step 只 backward 语义 loss
            if do_channel:
                active_loss = alpha * channel_loss
            else:
                active_loss = beta * semantic_loss
            total_loss_for_log = alpha * channel_loss + beta * semantic_loss

            if not torch.isfinite(active_loss):
                raise RuntimeError(f"Non-finite train joint loss sender={sender}, epoch={epoch}")
            train_channel_sum += float(channel_loss.item())
            train_semantic_sum += float(semantic_loss.item())
            train_total_sum += float(total_loss_for_log.item())
            train_steps += 1
            active_loss.backward()
            optimizer.step()
            global_step += 1
            _update_med_and_check(
                model=model, med_replay_enabled=med_replay_enabled,
                current_batch=current_batch, med_seen_keys=med_seen_keys,
            )

        model.eval()
        val_semantic_sum = 0.0
        val_joint_sum = 0.0
        val_steps = 0
        val_lv_a_correct = 0
        val_lv_b_correct = 0
        val_lv_ab_total = 0
        with torch.no_grad():
            for batch in tqdm(val_batches, desc=f"val joint ({sender})", leave=False):
                for rec in batch:
                    image_path = rec["path"]
                    image = Image.open(image_path).convert("RGB")
                    source_text = caption_cache.get(str(image_path))
                    out = model.forward_joint_phase(
                        image=image,
                        snr_db=float((snr_min_db + snr_max_db) / 2.0),
                        source_text=source_text,
                        image_id=image_path.name,
                        dataset_id="val",
                    )
                    _assert_semantic_forward_contract(out)
                    channel_loss = _masked_sequence_mse(out["recovered_seq"], out["semantic_seq_detached"], out["padding_mask"])
                    semantic_loss = criterion(out["logits"].reshape(-1, out["logits"].size(-1)), out["target_ids"].reshape(-1))
                    joint_loss = alpha * channel_loss + beta * semantic_loss
                    val_semantic_sum += float(semantic_loss.item())
                    val_joint_sum += float(joint_loss.item())
                    val_steps += 1
                    # Level-A/B
                    _lbl = int(rec.get("label", -1))
                    if _lbl in (0, 1):
                        src_txt = source_text or ""
                        if _train_text_matches_label(src_txt, _lbl):
                            val_lv_a_correct += 1
                        # Level-B: 兼容 [T,V] 和 [1,T,V]
                        _lg = out["logits"]
                        if _lg.dim() == 3:
                            _lg = _lg[0]
                        pred_ids = _lg.argmax(dim=-1).unsqueeze(0)  # [1, T]
                        rec_txt = model.tokenizer.decode(pred_ids)[0]
                        if _train_text_matches_label(rec_txt, _lbl):
                            val_lv_b_correct += 1
                        val_lv_ab_total += 1
        val_joint = val_joint_sum / max(val_steps, 1)
        val_semantic = val_semantic_sum / max(val_steps, 1)
        train_channel = train_channel_sum / max(train_steps, 1)
        train_semantic = train_semantic_sum / max(train_steps, 1)
        train_joint = train_total_sum / max(train_steps, 1)
        metric = val_semantic if monitor == "val_semantic_loss" else val_joint

        is_best = metric < best_metric
        if is_best:
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
            best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "monitor": monitor,
                    "phase": "joint",
                    "selected_by": "val",
                    "val_metric_name": monitor,
                },
                best_checkpoint_path,
            )
        else:
            stale_epochs += 1

        best_mark = " *" if is_best else ""
        early_stop_info = f"  patience={stale_epochs}/{early_stop_patience}" if (not is_best and early_stop_patience > 0) else ""
        lv_ab_str = (
            f"  A={val_lv_a_correct/val_lv_ab_total:.1%}  B={val_lv_b_correct/val_lv_ab_total:.1%}"
            if val_lv_ab_total > 0 else ""
        )
        print(
            f"[joint][{sender}] epoch {epoch}/{max_joint_epochs}  "
            f"train_ch={train_channel:.4f}  train_sem={train_semantic:.4f}  train_tot={train_joint:.4f}  "
            f"val_sem={val_semantic:.4f}  val_joint={val_joint:.4f}"
            f"{lv_ab_str}{best_mark}{early_stop_info}"
        )
        if med_replay_enabled:
            _log_med_state(model, phase="joint", epoch=epoch)

        if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            print(f"[joint][{sender}] early stop at epoch {epoch} (patience={early_stop_patience})")
            break

    if med_replay_enabled and int(med_replay_batch_size) > 0 and not memory_used_any:
        raise RuntimeError(
            "with_med joint phase never consumed memory replay samples (memory_count stayed 0 for all steps)."
        )

    torch.save({"epoch": best_epoch, "state_dict": model.state_dict(), "phase": "joint_last"}, last_checkpoint_path)
    return {
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "monitor": monitor,
    }
