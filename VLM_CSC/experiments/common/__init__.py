"""VLM_CSC 实验共享工具包。"""
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

# Re-export vlm_csc for backward compatibility
import vlm_csc

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
    "vlm_csc",
]
