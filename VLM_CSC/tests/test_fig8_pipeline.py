"""Smoke test for Fig.8 MED ablation pipeline outputs."""

from pathlib import Path

from eval.eval_fig8 import run_fig8_eval


def test_fig8_outputs_created(tmp_path: Path) -> None:
    out_dir = tmp_path / "fig8"
    result = run_fig8_eval(output_root=str(out_dir))
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "med_off_bleu1_map.png").exists()
    assert (out_dir / "med_off_bleu2_map.png").exists()
    assert (out_dir / "med_on_bleu1_map.png").exists()
    assert (out_dir / "med_on_bleu2_map.png").exists()
    assert (out_dir / "logs" / "run_meta.json").exists()
    assert Path(result["results_csv"]).exists()
