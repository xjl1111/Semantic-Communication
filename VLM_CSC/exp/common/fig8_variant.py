"""Fig8 MED 变体相关的配置解析与断言。"""
from __future__ import annotations

from typing import Dict, Tuple


def resolve_fig8_variant_med_config(variant: str, med_kwargs_base: Dict | None) -> Tuple[bool, Dict | None]:
    name = str(variant).strip().lower()
    if name == "with_med":
        if med_kwargs_base is None:
            raise RuntimeError("Fig8 variant='with_med' requires med_kwargs to be provided.")
        return True, dict(med_kwargs_base)
    if name == "without_med":
        return False, None
    raise RuntimeError(f"Unsupported Fig8 variant: {variant}")


def assert_fig8_variant_model_state(*, variant: str, enable_med: bool, med_kwargs: Dict | None, model) -> None:
    name = str(variant).strip().lower()
    if name == "with_med":
        if not bool(enable_med):
            raise RuntimeError("Fig8 variant='with_med' requires enable_med=True.")
        if med_kwargs is None:
            raise RuntimeError("Fig8 variant='with_med' requires med_kwargs (not None).")
        if getattr(model, "med", None) is None:
            raise RuntimeError("Fig8 variant='with_med' but model.med is None.")
        return

    if name == "without_med":
        if bool(enable_med):
            raise RuntimeError("Fig8 variant='without_med' requires enable_med=False.")
        if med_kwargs is not None:
            raise RuntimeError("Fig8 variant='without_med' requires med_kwargs=None.")
        if getattr(model, "med", None) is not None:
            raise RuntimeError("Fig8 variant='without_med' but model.med is not None.")
        return

    raise RuntimeError(f"Unsupported Fig8 variant: {variant}")
