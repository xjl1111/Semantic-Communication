"""Step 5 smoke test assertions."""

import math

from scripts.smoke_test import run_step5_smoke


def test_step5_smoke_pipeline() -> None:
    result = run_step5_smoke(seed=7)

    sb = result["single_batch"]
    assert sb["token_ids_shape"][0] == 2
    assert sb["semantic_features_shape"][-1] == 128
    assert sb["symbols_shape"][-1] == 128
    assert sb["decoded_features_shape"][-1] == 128
    assert sb["logits_shape"][-1] == 500
    assert sb["recon_images_shape"][1] == 3

    assert result["stage_a"]["steps"] > 0
    assert result["stage_b"]["steps"] > 0
    assert math.isfinite(float(result["stage_a"]["loss"]))
    assert math.isfinite(float(result["stage_b"]["loss"]))
    assert result["stage_c"]["history_len"] >= 1
