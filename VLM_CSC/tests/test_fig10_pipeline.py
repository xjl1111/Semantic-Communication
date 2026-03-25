"""Smoke test for Fig.10 comparison pipeline outputs."""

from pathlib import Path

from eval.eval_fig10 import run_fig10_eval


def test_fig10_outputs_created(tmp_path: Path) -> None:
    out_dir = tmp_path / "fig10"
    result = run_fig10_eval(output_root=str(out_dir), image_size=64)
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "comparison.png").exists()
    assert (out_dir / "comparison.pdf").exists()
    assert (out_dir / "semantic_alignment.png").exists()
    assert (out_dir / "logs" / "run_meta.json").exists()
    assert Path(result["results_csv"]).exists()
