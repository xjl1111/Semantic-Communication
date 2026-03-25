"""Smoke test for Fig.9 NAM ablation pipeline outputs."""

from pathlib import Path

from eval.eval_fig9 import run_fig9_eval


def test_fig9_outputs_created(tmp_path: Path) -> None:
    out_dir = tmp_path / "fig9"
    result = run_fig9_eval(output_root=str(out_dir))
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "curve.png").exists()
    assert (out_dir / "logs" / "run_meta.json").exists()
    assert Path(result["results_csv"]).exists()
