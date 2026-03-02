import torch
import traceback
from model.channel import Channel

ch = Channel('rayleigh')
x = torch.randn(2, 4, 8, 8)

try:
    y = ch(x)
    print(f"SUCCESS! Output shape: {y.shape}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
