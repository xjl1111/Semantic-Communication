import torch
import torch.nn as nn


class NAM(nn.Module):
    """Noise Attention Mechanism (NAM) — strict paper implementation.

    Paper specifies 4-layer FF with neuron counts: 56, 128, 56, 56.

    SNR projection path (layers 1–3):
        v' = ReLU(W_n2 · ReLU(W_n1 · r + b_n1) + b_n2)       [layers 1,2: 56, 128]
        v  = Sigmoid(W_n3 · v' + b_n3)                          [layer 3: 56]

    Feature transform (layer 4):
        e  = W_n4 · G + b_n4                                     [layer 4: 56]

    Gate & output:
        K  = Sigmoid(e · v)                                       [56-dim]
        gate = Linear(K, feature_dim)                             [back-project to feature_dim]
        A_i = Sigmoid(gate_i) · G_i
    """

    # Locked hidden dimensions from the paper — must not be changed.
    _PAPER_HIDDEN_DIMS = (56, 128, 56, 56)

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        d1, d2, d3, d4 = self._PAPER_HIDDEN_DIMS

        # SNR projection: 1 → 56 (ReLU) → 128 (ReLU) → 56 (Sigmoid)
        self.snr_fc1 = nn.Linear(1, d1)       # W_n1: layer-1, 56 neurons
        self.snr_fc2 = nn.Linear(d1, d2)      # W_n2: layer-2, 128 neurons
        self.snr_fc3 = nn.Linear(d2, d3)      # W_n3: layer-3, 56 neurons → v

        # Feature transform: feature_dim → 56
        self.feat_fc = nn.Linear(self.feature_dim, d4)  # W_n4: layer-4, 56 neurons → e

        # Back-projection: 56 → feature_dim
        # (engineering necessity: paper gate K is 56-dim but G is feature_dim-dim;
        #  this linear maps the bottleneck gate back to feature_dim for A_i = K_i * G_i)
        self.gate_proj = nn.Linear(d4, self.feature_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _format_snr(self, x: torch.Tensor, snr: float | torch.Tensor | None) -> torch.Tensor:
        batch_size = x.size(0)
        if snr is None:
            snr_tensor = torch.zeros((batch_size, 1), device=x.device, dtype=x.dtype)
        elif isinstance(snr, torch.Tensor):
            snr_tensor = snr.to(device=x.device, dtype=x.dtype)
            if snr_tensor.dim() == 0:
                snr_tensor = snr_tensor.view(1, 1).expand(batch_size, 1)
            elif snr_tensor.dim() == 1:
                snr_tensor = snr_tensor.view(-1, 1)
            if snr_tensor.size(0) == 1 and batch_size > 1:
                snr_tensor = snr_tensor.expand(batch_size, 1)
        else:
            snr_tensor = torch.full((batch_size, 1), float(snr), device=x.device, dtype=x.dtype)
        return snr_tensor

    def forward(self, x: torch.Tensor, snr: float | torch.Tensor | None = None) -> torch.Tensor:
        # --- SNR projection (equations 17–18) ---
        snr_tensor = self._format_snr(x, snr)
        v_prime = self.relu(self.snr_fc1(snr_tensor))           # (B, 56)
        v_prime = self.relu(self.snr_fc2(v_prime))              # (B, 128)
        v = self.sigmoid(self.snr_fc3(v_prime))                 # (B, 56)
        if x.dim() == 3:
            v = v.unsqueeze(1)                                  # (B, 1, 56)

        # --- Feature transform (equation 20) ---
        e = self.feat_fc(x)                                     # (B, [L,] 56)

        # --- Gate (equation 19) ---
        K = self.sigmoid(e * v)                                 # (B, [L,] 56)

        # --- Back-project to feature_dim & apply to input (equation 21) ---
        gate = self.sigmoid(self.gate_proj(K))                  # (B, [L,] feature_dim)
        return x * gate


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_nam: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.use_nam = bool(use_nam)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.feature_dim,
                    nhead=int(num_heads),
                    dim_feedforward=int(ff_dim),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        # Structural NAM: only create NAM layers when use_nam=True.
        # When use_nam=False, nam_layers is None — no NAM parameters exist in the model.
        if self.use_nam:
            self.nam_layers = nn.ModuleList([NAM(feature_dim=self.feature_dim) for _ in range(int(num_layers))])
        else:
            self.nam_layers = None

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        snr: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("SemanticEncoder expects input shape (B, L, D).")
        if x.size(-1) != self.feature_dim:
            raise ValueError(f"SemanticEncoder expects feature dim {self.feature_dim}, got {x.size(-1)}.")

        out = x
        for i, transformer_layer in enumerate(self.transformer_layers):
            out = transformer_layer(out, src_key_padding_mask=src_key_padding_mask)
            if self.use_nam:
                out = self.nam_layers[i](out, snr=snr)
        return out


class SemanticDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        use_nam: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.use_nam = bool(use_nam)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.feature_dim,
                    nhead=int(num_heads),
                    dim_feedforward=int(ff_dim),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        if self.use_nam:
            self.nam_layers = nn.ModuleList([NAM(feature_dim=self.feature_dim) for _ in range(int(num_layers))])
        else:
            self.nam_layers = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        snr: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tgt.dim() != 3 or memory.dim() != 3:
            raise ValueError("SemanticDecoder expects tgt and memory with shape (B, L, D).")
        if tgt.size(-1) != self.feature_dim or memory.size(-1) != self.feature_dim:
            raise ValueError(f"SemanticDecoder expects feature dim {self.feature_dim}.")

        out = tgt
        for i, transformer_layer in enumerate(self.transformer_layers):
            out = transformer_layer(
                tgt=out,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if self.use_nam:
                out = self.nam_layers[i](out, snr=snr)
        return out


class ChannelEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 128, use_nam: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_nam = bool(use_nam)

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, self.output_dim)
        self.relu = nn.ReLU()

        if self.use_nam:
            self.nam1 = NAM(feature_dim=256)
            self.nam2 = NAM(feature_dim=128)
        else:
            self.nam1 = None
            self.nam2 = None

    def forward(self, x: torch.Tensor, snr: float | torch.Tensor | None = None) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(f"ChannelEncoder expects last dim {self.input_dim}, got {x.size(-1)}.")

        out = self.relu(self.fc1(x))
        if self.use_nam:
            out = self.nam1(out, snr=snr)
        out = self.relu(self.fc2(out))
        if self.use_nam:
            out = self.nam2(out, snr=snr)
        out = self.fc_out(out)
        return out


class ChannelDecoder(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 128, use_nam: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_nam = bool(use_nam)

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc_out = nn.Linear(256, self.output_dim)
        self.relu = nn.ReLU()

        if self.use_nam:
            self.nam1 = NAM(feature_dim=128)
            self.nam2 = NAM(feature_dim=256)
        else:
            self.nam1 = None
            self.nam2 = None

    def forward(self, x: torch.Tensor, snr: float | torch.Tensor | None = None) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(f"ChannelDecoder expects last dim {self.input_dim}, got {x.size(-1)}.")

        out = self.relu(self.fc1(x))
        if self.use_nam:
            out = self.nam1(out, snr=snr)
        out = self.relu(self.fc2(out))
        if self.use_nam:
            out = self.nam2(out, snr=snr)
        out = self.fc_out(out)
        return out


class PhysicalChannel(nn.Module):
    """Physical channel model.

    In strict paper-repro mode, Rayleigh branch enforces power normalization and
    uses no approximation/debug fallback path.
    """

    def __init__(self, channel_type: str = "awgn", rayleigh_mode: str = "fast"):
        super().__init__()
        self.channel_type = str(channel_type).lower()
        self.rayleigh_mode = str(rayleigh_mode).lower()

    @staticmethod
    def power_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        power = x.pow(2).mean(dim=-1, keepdim=True)
        return x / torch.sqrt(power + eps)

    @staticmethod
    def _snr_to_std(snr_db: float) -> float:
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        sigma2 = 1.0 / snr_linear
        return float(sigma2 ** 0.5)

    def _awgn(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        std = self._snr_to_std(snr_db)
        noise = torch.randn_like(x) * std
        return x + noise

    def _rayleigh(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        std = self._snr_to_std(snr_db)
        if self.rayleigh_mode not in ("fast", "block"):
            raise ValueError("rayleigh_mode must be 'fast' or 'block'.")

        if self.rayleigh_mode == "fast":
            h_real = torch.randn_like(x)
            h_imag = torch.randn_like(x)
            h = torch.sqrt(h_real.pow(2) + h_imag.pow(2)) / (2.0 ** 0.5)
        else:
            shape = [x.size(0)] + [1] * (x.dim() - 1)
            h_real = torch.randn(shape, device=x.device, dtype=x.dtype)
            h_imag = torch.randn(shape, device=x.device, dtype=x.dtype)
            h = (torch.sqrt(h_real.pow(2) + h_imag.pow(2)) / (2.0 ** 0.5)).expand_as(x)

        noise = torch.randn_like(x) * std
        return h * x + noise

    def forward(self, x: torch.Tensor, snr_db: float, normalize_power: bool = True) -> torch.Tensor:
        if self.channel_type == "rayleigh" and not normalize_power:
            raise RuntimeError("Rayleigh channel requires normalize_power=True under strict protocol.")

        if normalize_power:
            x = self.power_normalize(x)

        if self.channel_type == "awgn":
            return self._awgn(x, snr_db)
        if self.channel_type == "rayleigh":
            return self._rayleigh(x, snr_db)

        raise ValueError("channel_type must be 'awgn' or 'rayleigh'.")
