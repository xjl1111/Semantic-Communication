"""Smoke test for Fig.7 evaluation pipeline outputs."""

from pathlib import Path

from eval.eval_fig7 import run_fig7_eval


def test_fig7_outputs_created(tmp_path: Path) -> None:
    out_dir = tmp_path / "fig7"
    result = run_fig7_eval(output_root=str(out_dir))
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "curve.png").exists()
    assert (out_dir / "logs" / "run_meta.json").exists()
    assert Path(result["results_csv"]).exists()
