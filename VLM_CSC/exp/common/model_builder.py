"""VLM-CSC 模型实例构建。"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Dict


def build_vlm_system(
    vlm_module,
    *,
    sender: str,
    blip_dir: Path,
    ram_ckpt: Path | None,
    sd_dir: Path,
    channel_type: str,
    device: str,
    quiet_third_party: bool,
    use_real_receiver_ckb: bool = True,
    enable_med: bool,
    med_kwargs: Dict | None,
    max_text_len: int = 24,
    max_text_len_by_sender: Dict[str, int] | None = None,
    use_nam: bool = True,
    channel_dim: int | None = None,
    caption_mode: str = "baseline",
    caption_prompt: str | None = None,
):
    resolved_max_text_len = int(max_text_len)
    if max_text_len_by_sender:
        sender_key = str(sender).strip().lower()
        if sender_key in max_text_len_by_sender and max_text_len_by_sender[sender_key] is not None:
            resolved_max_text_len = int(max_text_len_by_sender[sender_key])

    kwargs = dict(
        feature_dim=128,
        max_text_len=resolved_max_text_len,
        channel_type=channel_type,
        sender_type=sender,
        use_real_ckb=True,
        use_real_receiver_ckb=use_real_receiver_ckb,
        enable_med=bool(enable_med),
        med_kwargs=med_kwargs,
        blip_dir=blip_dir,
        ram_ckpt=ram_ckpt,
        sd_dir=sd_dir,
        device=device,
        use_nam=bool(use_nam),
        channel_dim=channel_dim,
        caption_mode=caption_mode,
        caption_prompt=caption_prompt,
    )

    if not quiet_third_party:
        return vlm_module.VLMCscSystem(**kwargs).to(device)

    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            model = vlm_module.VLMCscSystem(**kwargs).to(device)
        return model
    except Exception:
        captured = stream.getvalue().strip()
        if captured:
            print("[THIRD_PARTY_LOG] model init log (last 1200 chars):")
            print(captured[-1200:])
        raise
