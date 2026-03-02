"""快速验证脚本 - 测试所有实验基本功能

运行小steps验证代码正确性，无需完整训练
预计总耗时: 5-10分钟 (CPU)
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run_command(desc: str, cmd: list[str]) -> bool:
    """运行命令并报告结果"""
    print(f"\n{'='*60}")
    print(f"🔍 {desc}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True, check=True)
        print(f"\n✅ {desc} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {desc} - 失败")
        print(f"错误码: {e.returncode}")
        return False


def main():
    print("=" * 60)
    print("🚀 Deep JSCC 快速验证脚本")
    print("=" * 60)
    print("测试内容:")
    print("  1. 实验0: 环境与模型结构验证")
    print("  2. 实验1: Matched-SNR快速训练 (1000 steps)")
    print("  3. 实验2: SNR Mismatch快速测试 (1000 steps)")
    print("  4. 实验3: Rayleigh Fading快速测试 (1000 steps)")
    print("=" * 60)
    
    # 实验0: 基础验证（必须通过）
    success = run_command(
        "实验0: 环境与模型结构验证",
        [sys.executable, "deep_jscc/tests/test_experiment0_sanity.py"]
    )
    
    if not success:
        print("\n❌ 实验0失败！模型结构有问题，请检查代码。")
        return False
    
    # 实验1: Matched-SNR (单个SNR快速测试)
    run_command(
        "实验1: Matched-SNR训练 (SNR=10dB, 1000 steps)",
        [sys.executable, "deep_jscc/experiments/exp1_matched_snr.py",
         "--kn", "1/12", "--steps", "1000", "--snr", "10", "--device", "cpu"]
    )
    
    # 实验2: SNR Mismatch (缩减版)
    run_command(
        "实验2: SNR Mismatch鲁棒性 (train=[4,13], test=[0-20], 1000 steps)",
        [sys.executable, "deep_jscc/experiments/exp2_snr_mismatch.py",
         "--kn", "1/12", "--steps", "1000", "--train-snrs", "4,13",
         "--test-snrs", "0,5,10,15,20", "--device", "cpu"]
    )
    
    # 实验3: Rayleigh Fading (单个SNR)
    run_command(
        "实验3: Rayleigh Fading (SNR=10dB, 1000 steps)",
        [sys.executable, "deep_jscc/experiments/exp3_rayleigh_fading.py",
         "--kn", "1/12", "--steps", "1000", "--snr", "10", "--device", "cpu"]
    )
    
    print("\n" + "=" * 60)
    print("✅ 快速验证完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 查看 deep_jscc/results/ 下的输出文件")
    print("  2. 如需完整实验，使用更大的 --steps 和 --device cuda")
    print("  3. 参考 README_ALIGNED.md 获取详细说明\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
