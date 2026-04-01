import sys
import pathlib
import torch
import itertools
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# NOTE:
# This single-file runner contains all test functions and runs them sequentially.
# Pixel-level MSE monotonicity with respect to SNR is only expected
# when Encoder and Decoder are trained end-to-end. For untrained models,
# only symbol-level noise statistics should be verified.

# Ensure package root is on sys.path so local imports work
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
COMMON_DATASETS_DIR = REPO_ROOT / "data" / "datasets"
TORCHVISION_CIFAR10_DIR = COMMON_DATASETS_DIR / "torchvision_cifar10"

from model.encoder import Encoder
from model.channel import Channel
from model.decoder import Decoder


def test_end_to_end_shape():
    torch.manual_seed(0)
    B, H, W = 4, 32, 32
    C = 128

    x = torch.randn(B, 3, H, W)
    enc = Encoder(in_channels=3, c=C)
    ch = Channel(channel_type="awgn", snr_db=10.0)
    dec = Decoder(in_channels=C, out_channels=3)

    z = enc(x)
    y = ch(z)
    out = dec(y)

    assert out.shape == (B, 3, H, W)


def test_backward_gradients_flow():
    torch.manual_seed(0)
    B, H, W = 4, 32, 32
    C = 128

    x = torch.randn(B, 3, H, W, requires_grad=True)
    enc = Encoder(in_channels=3, c=C)
    ch = Channel(channel_type="awgn", snr_db=10.0)
    dec = Decoder(in_channels=C, out_channels=3)

    z = enc(x)
    y = ch(z)
    out = dec(y)

    loss = torch.mean((x - out) ** 2)
    loss.backward()

    num_channel_params = sum(1 for _ in ch.parameters())
    assert num_channel_params == 0

    missing_grads = [name for name, p in enc.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing_grads, f"Encoder parameters missing grads: {missing_grads}"


def test_channel_noise_power_matches_theory():
    torch.manual_seed(0)
    B, H, W = 4, 32, 32
    C = 128

    x = torch.randn(B, 3, H, W)

    enc = Encoder(in_channels=3, c=C)
    dec = Decoder(in_channels=C, out_channels=3)

    with torch.no_grad():
        z = enc(x)

    snrs = [0, 5, 10, 20]

    for snr in snrs:
        ch = Channel(channel_type="awgn", snr_db=float(snr))
        with torch.no_grad():
            y = dec(ch(z))
        mse = torch.mean((x - y) ** 2).item()
        print(f"SNR={snr} dB, pixel MSE={mse:.6e}")

    b, c, h, w = z.shape
    assert c % 2 == 0
    z_pair = z.reshape(b, c // 2, 2, h, w)
    s_real = z_pair[:, :, 0, :, :]
    s_imag = z_pair[:, :, 1, :, :]
    s = torch.complex(s_real, s_imag)

    for snr in snrs:
        ch = Channel(channel_type="awgn", snr_db=float(snr))
        with torch.no_grad():
            y_out = ch(z)
        y_pair = y_out.reshape(b, c // 2, 2, h, w)
        y_real = y_pair[:, :, 0, :, :]
        y_imag = y_pair[:, :, 1, :, :]
        y_complex = torch.complex(y_real, y_imag)

        noise = (y_complex - s)
        noise_power = torch.mean((noise.real ** 2 + noise.imag ** 2)).item()
        theoretical = 10 ** (-float(snr) / 10.0)
        rel_err = abs(noise_power - theoretical) / theoretical
        print(f"[symbols] SNR={snr} dB, measured noise power={noise_power:.6e}, theoretical={theoretical:.6e}, rel_err={rel_err:.3f}")
        assert rel_err < 0.15


def test_identity_decoder_snr_mse():
    torch.manual_seed(0)
    B, H, W = 4, 32, 32
    C = 128

    x = torch.randn(B, 3, H, W)
    x_small = torch.nn.functional.adaptive_avg_pool2d(x, (H // 4, W // 4))
    z = x_small.repeat(1, C // 3 + 1, 1, 1)[:, :C]

    def simple_decoder(z_tensor):
        z_rgb = z_tensor[:, :3, :, :]
        return torch.nn.functional.interpolate(z_rgb, scale_factor=4, mode="bilinear", align_corners=False)

    snrs = [0, 5, 10, 20]
    mses = []

    for snr in snrs:
        ch = Channel(channel_type="awgn", snr_db=float(snr))
        with torch.no_grad():
            y = simple_decoder(ch(z))
        mse = torch.mean((x - y) ** 2).item()
        print(f"SNR={snr} dB, MSE={mse:.6e}")
        mses.append(mse)

    print("MSEs:", mses)
    assert mses[0] > mses[-1], f"Identity-decoder test failed: MSEs={mses}"


def test_deepjscc_trained_snr_curve(steps: int = 3000):
    """This test verifies whether a minimally trained DeepJSCC model
    has learned to exploit channel SNR information.

    Pixel-level SNR–MSE monotonicity is only expected AFTER
    end-to-end training with randomized SNR.
    """
    torch.manual_seed(0)
    # training on CIFAR-10 (downloaded if necessary)
    B, H, W = 128, 32, 32
    # Reduce channel count to increase compression difficulty and amplify SNR impact
    C = 16

    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])
    TORCHVISION_CIFAR10_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.CIFAR10(root=str(TORCHVISION_CIFAR10_DIR), train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=B, shuffle=True, drop_last=True, num_workers=0)
    train_iter = iter(itertools.cycle(train_loader))

    # fixed test batch used after training
    test_ds = datasets.CIFAR10(root=str(TORCHVISION_CIFAR10_DIR), train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    test_x, _ = next(iter(test_loader))

    enc = Encoder(in_channels=3, c=C)
    dec = Decoder(in_channels=C, out_channels=3)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=3e-3)

    enc.train()
    dec.train()

    for step in range(steps):
        xb, _ = next(train_iter)
        xb = xb.clone()
        if xb.shape[0] != B:
            continue
        # sample extreme SNRs (including a harsher low SNR) to force distinct strategies
        # slightly bias toward harsh low SNR to magnify contrast
        snr = -12.0 if random.random() < 0.7 else 20.0
        ch = Channel(channel_type="awgn", snr_db=snr)
        y = dec(ch(enc(xb)))
        loss = ((xb - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 500 == 0:
            print(f"training step {step+1}/{steps}, last loss={loss.item():.6e}, snr={snr:.2f}dB")

    enc.eval()
    dec.eval()

    snrs = [-12, 0, 5, 10, 20]
    mses = []
    with torch.no_grad():
        for snr in snrs:
            ch = Channel(channel_type="awgn", snr_db=float(snr))
            y = dec(ch(enc(test_x)))
            mses.append(((test_x - y) ** 2).mean().item())
            print(f"SNR={snr} dB, MSE={mses[-1]:.6e}")

    # require at least 5% relative improvement from -12dB to 20dB
    improvement = (mses[0] - mses[-1]) / (mses[0] + 1e-12)
    assert improvement > 0.05, f"SNR utilization too weak: relative improvement={improvement:.4f}"


def main():
    tests = [
        ("End-to-end shape", test_end_to_end_shape),
        ("Backward gradients", test_backward_gradients_flow),
        ("Channel noise power vs theory", test_channel_noise_power_matches_theory),
        ("Identity decoder SNR-MSE", test_identity_decoder_snr_mse),
        ("Minimal training SNR curve", test_deepjscc_trained_snr_curve),
    ]

    for name, fn in tests:
        print(f"Running: {name}")
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL: {name} — {e}")
            raise
        except Exception as e:
            print(f"ERROR: {name} raised unexpected exception: {e}")
            raise
        else:
            print(f"PASS: {name}\n")


if __name__ == "__main__":
    main()
