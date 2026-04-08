"""VLM_CSC 实验共享工具包。"""
import sys
from pathlib import Path

# Ensure VLM_CSC root is on sys.path so model imports work
_VLM_CSC_ROOT = Path(__file__).resolve().parents[1]
if str(_VLM_CSC_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLM_CSC_ROOT))

from common.utils import (
    LABEL_MAP,
    configure_runtime_logging,
    collect_binary_images_from_split,
    collect_generic_images_from_split,
    chunk_records,
)
from common.dataset_manager import TaskDatasetManager
from common.model_builder import build_vlm_system
from common.fig8_variant import (
    resolve_fig8_variant_med_config,
    assert_fig8_variant_model_state,
)

# Re-export model for backward compatibility (renamed from vlm_csc)
import model

__all__ = [
    "LABEL_MAP",
    "configure_runtime_logging",
    "collect_binary_images_from_split",
    "collect_generic_images_from_split",
    "chunk_records",
    "TaskDatasetManager",
    "build_vlm_system",
    "resolve_fig8_variant_med_config",
    "assert_fig8_variant_model_state",
    "model",
]
