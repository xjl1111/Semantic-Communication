import sys
import pathlib
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Ensure package root is on sys.path so local imports work
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.encoder import Encoder
from model.channel import Channel
from model.decoder import Decoder
from utils.metrics import mse, avg_power


def _build_models():
    enc = Encoder(in_channels=3, c=128)
    ch = Channel(channel_type="awgn", snr_db=10.0)
    dec = Decoder(in_channels=128, out_channels=3)
    return enc, ch, dec


def _get_cifar10_test_batch(batch_size: int = 4) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    x, _ = next(iter(loader))
    return x


def test_0_1_shape_interface():
    torch.manual_seed(0)
    B, H, W, C = 4, 32, 32, 128
    x = torch.randn(B, 3, H, W)

    enc, ch, dec = _build_models()
    z = enc(x)
    y = ch(z)
    x_hat = dec(y)

    # Encoder downsamples spatially by 4x via two stride-2 convs
    H_out, W_out = H // 4, W // 4
    assert z.shape == (B, C, H_out, W_out)
    assert y.shape == (B, C, H_out, W_out)
    assert x_hat.shape == (B, 3, H, W)


def test_0_2_encoder_power_normalization():
    torch.manual_seed(0)
    B, H, W = 4, 32, 32
    x = torch.randn(B, 3, H, W)

    enc, _, _ = _build_models()
    z = enc(x)
    power = avg_power(z).item()
    print(f"Encoder output power = {power:.6f}")

    assert abs(power - 1.0) <= 0.05, f"Encoder power out of range: {power:.6f}"


def test_0_3_awgn_snr_consistency():
    torch.manual_seed(0)
    B, H, W, C = 4, 32, 32, 128
    x = torch.randn(B, 3, H, W)

    enc, _, _ = _build_models()
    with torch.no_grad():
        z = enc(x)

    b, c, h, w = z.shape
    assert c % 2 == 0
    z_pair = z.reshape(b, c // 2, 2, h, w)
    s_real = z_pair[:, :, 0, :, :]
    s_imag = z_pair[:, :, 1, :, :]
    s = torch.complex(s_real, s_imag)

    snrs = [0, 5, 10, 20]
    for snr in snrs:
        ch = Channel(channel_type="awgn", snr_db=float(snr))
        with torch.no_grad():
            y_out = ch(z)
        y_pair = y_out.reshape(b, c // 2, 2, h, w)
        y_real = y_pair[:, :, 0, :, :]
        y_imag = y_pair[:, :, 1, :, :]
        y_complex = torch.complex(y_real, y_imag)

        noise = s - y_complex
        noise_power = torch.mean(noise.real ** 2 + noise.imag ** 2).item()
        theoretical = 10 ** (-float(snr) / 10.0)
        rel_err = abs(noise_power - theoretical) / theoretical
        print(
            f"SNR={snr} dB, measured noise power={noise_power:.6e}, "
            f"theoretical={theoretical:.6e}, rel_err={rel_err:.3f}"
        )
        assert rel_err < 0.05


def test_0_4_untrained_mse_snr_invariance():
    """Untrained Decoder behaves like a random function, so MSE is dominated
    by model mismatch rather than channel noise. Therefore MSE should not
    decrease significantly with higher SNR.
    """
    torch.manual_seed(0)
    B, H, W = 4, 32, 32
    x = _get_cifar10_test_batch(batch_size=B)

    enc, _, dec = _build_models()

    snrs = [0, 5, 10, 20]
    mses = []
    for snr in snrs:
        ch = Channel(channel_type="awgn", snr_db=float(snr))
        with torch.no_grad():
            x_hat = dec(ch(enc(x)))
        m = mse(x, x_hat).item()
        mses.append(m)

    print("SNR(dB) | MSE")
    for snr, m in zip(snrs, mses):
        print(f"{snr:>7} | {m:.6e}")


def main():
    tests = [
        ("0.1 Shape/Interface", test_0_1_shape_interface),
        ("0.2 Encoder Power", test_0_2_encoder_power_normalization),
        ("0.3 AWGN SNR", test_0_3_awgn_snr_consistency),
        ("0.4 Untrained MSE", test_0_4_untrained_mse_snr_invariance),
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
