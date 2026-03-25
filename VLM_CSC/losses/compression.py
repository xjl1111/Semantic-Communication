"""Compression ratio metric/loss utilities.

Paper traceability:
- Compression ratio evaluation in Fig.10 is 论文明确写出.
- Exact formula details are 为复现做的合理实现选择.
"""


def compression_ratio(transmitted_size_bits: int, original_size_bits: int) -> float:
    """Compute transmitted/original data size ratio."""
    if original_size_bits <= 0:
        raise ValueError("original_size_bits must be > 0")
    return transmitted_size_bits / original_size_bits
