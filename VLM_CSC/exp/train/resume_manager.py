"""
训练断点续传管理器
==================

训练过程中 **每个 phase 完成后** 自动保存进度到 ``resume_state.json``。
重启时检测到旧进度文件会交互式询问用户：继续还是重新开始。

使用方式
--------
在 ``train_experiment.py`` 的入口函数中自动创建和管理，无需用户手动操作。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class ResumeManager:
    """管理训练断点续传状态。

    状态文件结构示例（fig7/9/10）::

        {
          "fig_name": "fig7",
          "created": "2025-06-01T12:00:00",
          "completed": {
            "blip": {
              "channel": "path/to/blip_phase_channel_best.pth",
              "semantic": "path/to/blip_phase_semantic_best.pth",
              "joint_best": "path/to/blip_phase_joint_best.pth",
              "joint_last": "path/to/blip_phase_joint_last.pth"
            }
          }
        }

    fig8 的 key 格式为 ``variant/sender/task``，如 ``with_med/blip/cifar``。
    """

    RESUME_FILE = "resume_state.json"

    def __init__(self, checkpoint_dir: Path, fig_name: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.fig_name = fig_name
        self.state_file = self.checkpoint_dir / self.RESUME_FILE
        self.state: Dict = self._load_or_create()

    # ── 内部方法 ─────────────────────────────────────────────

    def _load_or_create(self) -> Dict:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                if data.get("fig_name") == self.fig_name:
                    return data
                # 不同实验的旧文件，忽略
            except (json.JSONDecodeError, KeyError):
                pass
        return {
            "fig_name": self.fig_name,
            "completed": {},
            "created": datetime.now().isoformat(),
        }

    def _save(self) -> None:
        self.state["updated"] = datetime.now().isoformat()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 公共 API ─────────────────────────────────────────────

    def has_progress(self) -> bool:
        """是否存在任何已保存的进度。"""
        return len(self.state.get("completed", {})) > 0

    def mark_phase_complete(
        self, key: str, phase: str, checkpoint_path: str
    ) -> None:
        """记录某个 key 的某个 phase 已完成。"""
        if key not in self.state["completed"]:
            self.state["completed"][key] = {}
        self.state["completed"][key][phase] = checkpoint_path
        self._save()

    def get_completed_phases(self, key: str) -> Dict[str, str]:
        """获取某个 key 已完成的所有 phase 及对应 checkpoint 路径。"""
        return dict(self.state.get("completed", {}).get(key, {}))

    def is_phase_complete(self, key: str, phase: str) -> bool:
        """检查某个特定 phase 是否已完成。"""
        return phase in self.get_completed_phases(key)

    def is_fully_complete(self, key: str) -> bool:
        """三个训练阶段（channel / semantic / joint）全部完成。"""
        phases = self.get_completed_phases(key)
        return all(p in phases for p in ("channel", "semantic", "joint_best"))

    def clear(self) -> None:
        """清除所有进度（重新开始时调用）。"""
        if self.state_file.exists():
            self.state_file.unlink()
        self.state = {
            "fig_name": self.fig_name,
            "completed": {},
            "created": datetime.now().isoformat(),
        }

    def summary_lines(self) -> List[str]:
        """生成可读的进度摘要行。"""
        lines: List[str] = []
        completed = self.state.get("completed", {})
        for key in sorted(completed):
            phases = completed[key]
            phase_names = sorted(phases.keys())
            lines.append(f"  [OK] {key}: {', '.join(phase_names)}")
        return lines


# ── 交互式询问 ───────────────────────────────────────────────


def prompt_resume_or_restart(resume_mgr: ResumeManager) -> bool:
    """检测是否存在旧进度并询问用户。

    Returns
    -------
    bool
        ``True`` = 用户选择继续；``False`` = 从头开始（旧状态已清除）。
        如果无旧进度，直接返回 ``False``（静默，不打印任何内容）。
    """
    if not resume_mgr.has_progress():
        return False

    print()
    print("=" * 65)
    print("  [!] 发现之前的训练进度！")
    print("=" * 65)
    for line in resume_mgr.summary_lines():
        print(line)
    print("=" * 65)

    # 非交互环境（stdin 不是终端或设置了 FIG8_AUTO_RESUME）自动继续
    import sys, os
    if not sys.stdin.isatty() or os.environ.get("FIG8_AUTO_RESUME"):
        print("  [非交互模式] 自动继续训练\n")
        return True

    while True:
        try:
            answer = input("  是否从上次中断处继续？(y=继续 / n=重新开始): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  [非交互模式] 默认继续训练")
            return True
        if answer in ("y", "yes", "是", "继续"):
            print("  [继续] 恢复训练，跳过已完成的阶段\n")
            return True
        if answer in ("n", "no", "否", "重新"):
            print("  [重置] 清除旧状态，从头开始\n")
            resume_mgr.clear()
            return False
        print("  请输入 y 或 n")
