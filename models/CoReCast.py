"""CoReCast

The frequency-consistency score, spectral continuation, adaptive spectral decay,
and shared linear/MLP forecaster are unchanged. Only LocalResidualAdapter is
replaced: overlapping residual patches -> shared embedding -> depthwise temporal
mixer -> small bottleneck -> zero-initialized forecast correction.

Copy this standalone file to models/SustainCastPatch.py (or models/SustainCast.py).
Interface: Model(configs); forward(x, x_mark=None, y_true=None) -> [B,T,C].
x has shape [B,L,C]. x_mark and y_true are ignored. Ordinary MSE is sufficient;
no external memory, custom CUDA extension, auxiliary loss, or test-time fitting.

Required config: seq_len >= 12, pred_len >= 1, enc_in >= 1.
Main options: d_model=64, dropout=0.1, sustaincast_patch_len=16.
Patch length is capped at L; stride=max(1,P//2); embedding and bottleneck widths
are min(16,d_model). Left replication padding aligns the last patch to the last
observed sample. No future sample is read. The old sustaincast_local_adapter flag
now enables/disables this patch adapter; all spectral ablation flags are retained.

"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["Model", "SustainCastModel", "SustainCastPatchModel"]


def _get(configs: Any, name: str, default: Any = None) -> Any:
    if isinstance(configs, Mapping):
        return configs.get(name, default)
    return getattr(configs, name, default)


def _flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        if value.lower() in ("false", "0", "no", "off"):
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _configuration(configs: Any) -> SimpleNamespace:
    values = {}
    for name in ("seq_len", "pred_len", "enc_in"):
        value = _get(configs, name)
        if value is None:
            raise AttributeError(f"configs.{name} is required.")
        values[name] = int(value)
        if values[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if values["seq_len"] < 12:
        raise ValueError("seq_len must be >= 12 for three Hann-windowed segments.")
    values["d_model"] = int(_get(configs, "d_model", 64))
    values["dropout"] = float(_get(configs, "dropout", 0.1))
    values["sustaincast_patch_len"] = int(_get(configs, "sustaincast_patch_len", 16))
    if values["sustaincast_patch_len"] < 2:
        raise ValueError("sustaincast_patch_len must be >= 2.")
    values["sustaincast_chunk_size"] = int(_get(configs, "sustaincast_chunk_size", 32))
    for name, default in (
        ("sustaincast_use_spectral", True),
        ("sustaincast_use_stability", True),
        ("sustaincast_use_damping", True),
        ("sustaincast_checkpoint", True),
        ("sustaincast_diagnostics", False),
        ("sustaincast_adaptive_decay", True),
        ("sustaincast_local_adapter", True),
    ):
        values[name] = _flag(_get(configs, name, default), name)
    if values["d_model"] < 1 or values["sustaincast_chunk_size"] < 1:
        raise ValueError("d_model and sustaincast_chunk_size must be positive.")
    if not math.isfinite(values["dropout"]) or not 0 <= values["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1).")
    # Make a private config: never mutate the caller's Namespace/dictionary.
    return SimpleNamespace(**values)


class RevIN(nn.Module):
    """Non-affine instance normalization with call-local statistics."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        return (x - mean) / std, mean, std

    @staticmethod
    def denormalize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return y * std + mean


class SpectralContinuation(nn.Module):
    """[N,L] -> historical periodic component [N,L], continuation [N,T].

    N is B*C. Every channel and frequency shares the same 3->16->1 gate.
    Stability is a historical proxy, NOT calibrated future predictability.
    The three windows are disjoint (with at most small gaps for odd lengths).
    """

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.seq_len, self.pred_len = cfg.seq_len, cfg.pred_len
        self.num_bins = self.seq_len // 2  # excludes DC; includes even Nyquist
        self.window_len = self.seq_len // 3
        last_start = self.seq_len - self.window_len
        self.starts = (0, last_start // 2, last_start)
        self.chunk_size = cfg.sustaincast_chunk_size
        self.enabled = cfg.sustaincast_use_spectral
        self.use_stability = cfg.sustaincast_use_stability
        self.use_damping = cfg.sustaincast_use_damping
        self.use_checkpoint = cfg.sustaincast_checkpoint
        self.collect_diagnostics = cfg.sustaincast_diagnostics
        self.adaptive_decay = cfg.sustaincast_adaptive_decay
        self.eps = 1e-8
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

        self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        # Small, nonzero final weights keep both gate layers connected at init.
        nn.init.normal_(self.gate[-1].weight, std=0.02)
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.raw_base_decay = nn.Parameter(torch.tensor(math.log(math.expm1(0.01))))
        self.raw_instability_decay = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        # Separate output from the existing gate hidden state: only 17 weights.
        # exp(log(4)*tanh(delta)) bounds the multiplier to (1/4, 4).
        self.decay_adjustment = nn.Linear(16, 1)
        nn.init.zeros_(self.decay_adjustment.weight)
        nn.init.zeros_(self.decay_adjustment.bias)

        # Generate constants in double precision; cast to compute dtype on use.
        # They are deterministic functions of L,T and are not checkpoint weights.
        k = torch.arange(1, self.num_bins + 1, dtype=torch.float64)
        omega = 2 * math.pi * k / self.seq_len
        starts = torch.tensor(self.starts, dtype=torch.float64)
        phase = -starts[:, None] * omega[None, :]
        window = torch.hann_window(self.window_len, periodic=False, dtype=torch.float64)
        window = window / window.sum()
        multiplicity = torch.full_like(k, 2.0)
        if self.seq_len % 2 == 0:
            multiplicity[-1] = 1.0
        h = torch.arange(1, self.pred_len + 1, dtype=torch.float64)
        # h=1 is time L, hence the same DFT phase as time zero.
        phase_time = torch.remainder(self.seq_len - 1 + h, self.seq_len)
        future_phase = phase_time[:, None] * omega[None, :]
        constants = {
            "window": window,
            "alignment_real": phase.cos(),
            "alignment_imag": phase.sin(),
            "relative_frequency": k / self.seq_len,
            "multiplicity": multiplicity,
            "future_cos": future_phase.cos(),
            "future_sin": future_phase.sin(),
            "future_distance": h / self.seq_len,
        }
        for name, value in constants.items():
            self.register_buffer(name, value, persistent=False)

    def _stability(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_stability:
            return x.new_ones(x.shape[0], self.num_bins)
        segments = torch.stack(
            [x[:, start:start + self.window_len] for start in self.starts], dim=1
        )
        window = self.window.to(dtype=x.dtype)
        local = torch.fft.rfft(segments * window, n=self.seq_len, dim=-1)[..., 1:]
        alignment = torch.complex(
            self.alignment_real.to(dtype=x.dtype),
            self.alignment_imag.to(dtype=x.dtype),
        )
        aligned = local * alignment
        summed = aligned.sum(dim=1)
        numerator = summed.real.square() + summed.imag.square()
        power = aligned.real.square() + aligned.imag.square()
        score = numerator / (3.0 * power.sum(dim=1) + self.eps)
        return score.clamp(0.0, 1.0)

    @staticmethod
    def _synthesis_block(
        coefficient_real: torch.Tensor,
        coefficient_imag: torch.Tensor,
        decay: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        # Only materialize [N,chunk,F], never the whole [N,T,F] tensor.
        envelope = torch.exp(-decay[:, None, :] * distance[None, :, None])
        waves = (
            coefficient_real[:, None, :] * cosine[None, :, :]
            - coefficient_imag[:, None, :] * sine[None, :, :]
        )
        return (envelope * waves).sum(dim=-1)

    def _synthesize(self, coefficients: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        dtype = coefficients.real.dtype
        weighted = coefficients * self.multiplicity.to(dtype=dtype)
        cosine = self.future_cos.to(dtype=dtype)
        sine = self.future_sin.to(dtype=dtype)
        distance = self.future_distance.to(dtype=dtype)
        outputs = []
        for start in range(0, self.pred_len, self.chunk_size):
            stop = min(start + self.chunk_size, self.pred_len)
            args = (weighted.real, weighted.imag, decay,
                    cosine[start:stop], sine[start:stop], distance[start:stop])
            # Chunking alone does NOT limit tensors retained by autograd.
            # Checkpointing recomputes each block in backward to avoid that cost.
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                block = checkpoint(self._synthesis_block, *args,
                                   use_reentrant=False, preserve_rng_state=False)
            else:
                block = self._synthesis_block(*args)
            outputs.append(block)
        return torch.cat(outputs, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_diagnostics = {}
        if not self.enabled:
            return torch.zeros_like(x), x.new_zeros(x.shape[0], self.pred_len)
        # Model.forward supplies float32/64, even under external autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            full_spectrum = torch.fft.rfft(x, dim=-1, norm="forward")
            spectrum = full_spectrum[:, 1:]
            stability = self._stability(x)
            energy = spectrum.real.square() + spectrum.imag.square()
            energy = energy * self.multiplicity.to(dtype=x.dtype)
            relative_energy = self.num_bins * energy / (energy.sum(-1, keepdim=True) + self.eps)
            frequency = self.relative_frequency.to(dtype=x.dtype).expand_as(stability)
            features = torch.stack((stability, relative_energy.log1p(), frequency), dim=-1)
            gate_hidden = self.gate[1](self.gate[0](features))
            gate = torch.sigmoid(self.gate[2](gate_hidden).squeeze(-1))
            decay_scale = torch.ones_like(stability)
            if self.use_damping:
                decay = (F.softplus(self.raw_base_decay)
                         + F.softplus(self.raw_instability_decay) * (1.0 - stability))
                if self.adaptive_decay:
                    adjustment = self.decay_adjustment(gate_hidden).squeeze(-1)
                    decay_scale = torch.exp(math.log(4.0) * torch.tanh(adjustment))
                    decay = decay * decay_scale
            else:
                decay = torch.zeros_like(stability)
            kept = spectrum * gate
            # DC is deliberately zero; the residual path keeps any remaining DC.
            kept_full = torch.cat((torch.zeros_like(full_spectrum[:, :1]), kept), dim=-1)
            past = torch.fft.irfft(kept_full, n=self.seq_len, dim=-1, norm="forward")
            future = self._synthesize(kept, decay)
            if self.collect_diagnostics:
                self.last_diagnostics = {
                    "stability": stability.detach(),
                    "gate": gate.detach(),
                    "decay": decay.detach(),
                    "decay_scale": decay_scale.detach(),
                }
            return past, future


class LocalResidualAdapter(nn.Module):
    """Overlapping historical residual patches, shared across variables.

    [N,L] -> [N,K,P] -> [N,K,d] -> [N,r] -> [N,T], where N=B*C.
    No attention matrix or [N,T,d] decoder activation is constructed.
    The output is zero-initialized. Its input layers start learning after the
    output weights leave zero; the surrounding V2 backbone remains trainable.
    """

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.enabled = cfg.sustaincast_local_adapter
        self.patch_len = min(cfg.sustaincast_patch_len, cfg.seq_len)
        self.stride = max(1, self.patch_len // 2)
        remainder = (cfg.seq_len - self.patch_len) % self.stride
        self.left_pad = (self.stride - remainder) % self.stride
        self.num_patches = 1 + (cfg.seq_len + self.left_pad - self.patch_len) // self.stride
        self.patch_dim = min(16, cfg.d_model)
        self.rank = min(16, cfg.d_model)
        self.input = nn.Linear(self.patch_len, self.patch_dim)
        self.norm = nn.LayerNorm(self.patch_dim)
        self.mixer = nn.Conv1d(self.patch_dim, self.patch_dim, kernel_size=3,
                               padding=1, groups=self.patch_dim)
        self.compress = nn.Linear(self.num_patches * self.patch_dim, self.rank)
        self.output = nn.Linear(self.rank, cfg.pred_len)
        self.dropout = nn.Dropout(cfg.dropout)
        nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.compress.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if self.left_pad:
            residual = F.pad(residual.unsqueeze(1), (self.left_pad, 0),
                             mode="replicate").squeeze(1)
        patches = residual.unfold(-1, self.patch_len, self.stride)
        tokens = self.input(patches)
        # Adjacent tokens contain overlapping observed samples. Convolution is
        # bidirectional within history; the complete forecast remains causal.
        mixed = self.mixer(self.norm(tokens).transpose(1, 2)).transpose(1, 2)
        tokens = tokens + F.gelu(mixed)
        hidden = F.gelu(self.compress(tokens.flatten(start_dim=1)))
        return self.output(self.dropout(hidden))


class SharedResidualPredictor(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.linear = nn.Linear(cfg.seq_len, cfg.pred_len)
        self.input = nn.Linear(2 * cfg.seq_len, cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.pred_len)
        self.dropout = nn.Dropout(cfg.dropout)
        # A mean forecast and a small nonlinear residual at initialization.
        nn.init.constant_(self.linear.weight, 1.0 / cfg.seq_len)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.output.weight, std=0.01 / math.sqrt(cfg.d_model))
        nn.init.zeros_(self.output.bias)
        self.local_adapter = LocalResidualAdapter(cfg)

    def forward(self, original: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input(torch.cat((original, residual), dim=-1)))
        forecast = self.linear(residual) + self.output(self.dropout(hidden))
        if self.local_adapter.enabled:
            forecast = forecast + self.local_adapter(residual)
        return forecast


class Model(nn.Module):
    """Public entry point: SustainCastPatch.Model(configs)."""

    def __init__(self, configs: Any):
        super().__init__()
        self.config = _configuration(configs)
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.enc_in = self.config.enc_in
        self.d_model = self.config.d_model
        self.revin = RevIN()
        self.spectral = SpectralContinuation(self.config)
        self.residual_predictor = SharedResidualPredictor(self.config)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        y_true: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Preserve the reference interface; never consume target information.
        del x_mark, y_true
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected [B,{self.seq_len},{self.enc_in}], got {tuple(x.shape)}.")
        if x.shape[0] < 1:
            raise ValueError("The batch dimension must be nonempty.")
        if not x.is_floating_point():
            raise TypeError("x must be a real floating-point tensor.")
        weight = self.residual_predictor.linear.weight
        if weight.dtype not in (torch.float32, torch.float64):
            raise TypeError("Keep weights float32/64. Use autocast instead of model.half()/bfloat16().")
        if x.device != weight.device:
            raise ValueError("Input and model must be on the same device.")
        batch = x.shape[0]
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized, mean, std = self.revin.normalize(x.to(dtype=weight.dtype))
            flat = normalized.transpose(1, 2).reshape(batch * self.enc_in, self.seq_len)
            past, periodic_future = self.spectral(flat)
            residual = flat - past
        # Only the residual MLP follows the caller's mixed-precision context.
        residual_future = self.residual_predictor(flat, residual)
        with torch.autocast(device_type=x.device.type, enabled=False):
            combined = periodic_future + residual_future.to(dtype=weight.dtype)
            combined = combined.reshape(batch, self.enc_in, self.pred_len).transpose(1, 2)
            prediction = self.revin.denormalize(combined, mean, std)
        return prediction.contiguous()

    @torch.no_grad()
    def load_base_state_dict(self, state_dict: Mapping) -> None:
        own = self.state_dict()
        added_prefixes = ("spectral.decay_adjustment.",
                          "residual_predictor.local_adapter.")
        base_keys = {key for key in own if not key.startswith(added_prefixes)}
        supplied = set(state_dict)
        missing, extra = sorted(base_keys - supplied), sorted(supplied - base_keys)
        if missing or extra:
            raise ValueError(f"Expected a V1 state_dict. Missing keys: {missing}; extra keys: {extra}.")
        for key in sorted(base_keys):
            value = state_dict[key]
            if not isinstance(value, torch.Tensor) or value.shape != own[key].shape:
                raise ValueError(f"V1 weight shape mismatch at {key}; use matching seq_len/pred_len/d_model.")
        # Preserve the randomly initialized adapter input projection, but reset
        # its output and the decay head so the loaded prediction is unchanged.
        combined = dict(own)
        combined.update(state_dict)
        reset_keys = (
            "spectral.decay_adjustment.weight", "spectral.decay_adjustment.bias",
            "residual_predictor.local_adapter.output.weight",
            "residual_predictor.local_adapter.output.bias",
        )
        for key in reset_keys:
            combined[key] = torch.zeros_like(own[key])
        self.load_state_dict(combined, strict=True)

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, Any]:
        """Optional diagnostics; set sustaincast_diagnostics=True at creation."""
        stats = self.spectral.last_diagnostics
        result: Dict[str, Any] = {"num_parameters": sum(p.numel() for p in self.parameters())}
        adapter = self.residual_predictor.local_adapter
        result.update({"patch_len": adapter.patch_len, "patch_stride": adapter.stride,
                       "num_patches": adapter.num_patches, "patch_dim": adapter.patch_dim})
        if stats:
            result.update({f"mean_{key}": value.mean().item() for key, value in stats.items()})
        return result


SustainCastModel = Model
SustainCastPatchModel = Model


def _self_test() -> None:
    torch.manual_seed(2027)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cfg = SimpleNamespace(seq_len=96, pred_len=24, enc_in=7, d_model=64,
                          dropout=0.0, sustaincast_diagnostics=True)
    model = Model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    time = torch.arange(cfg.seq_len + cfg.pred_len).float()[None, :, None]
    phase = 2 * math.pi * torch.rand(8, 1, cfg.enc_in)
    amplitude = 0.5 + torch.rand(8, 1, cfg.enc_in)
    series = amplitude * torch.sin(2 * math.pi * time / 24 + phase)
    series = series + 0.2 * torch.cos(2 * math.pi * time / 12 + 0.5 * phase)
    x, y = series[:, :cfg.seq_len], series[:, cfg.seq_len:]
    model.train()
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, None, y)
        loss = F.mse_loss(pred, y)
        loss.backward()
        assert pred.shape == y.shape and torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        optimizer.step()
        if step in (0, 4, 9):
            print(f"synthetic training step={step + 1}, mse={loss.item():.6f}")
    model.eval()
    with torch.no_grad():
        expected = model(x)
        assert torch.equal(expected, model(x, torch.randn(8, 96, 4), torch.randn_like(y)))
    stream = io.BytesIO()
    torch.save({"config": vars(model.config), "state_dict": model.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=True)
    restored = Model(saved["config"])
    restored.load_state_dict(saved["state_dict"])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(x))
    print("Output shape:", tuple(expected.shape))
    print("Diagnostics:", model.diagnostics())
    print("SustainCastPatch self-test passed. Synthetic training is not benchmark validation.")


if __name__ == "__main__":
    _self_test()
