from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .complex_cnvqg_model import TemporalStack
from .encoder import ConvEncoder


@dataclass
class EMAVQOutput:
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    perplexity: torch.Tensor
    active_fraction: torch.Tensor
    dead_fraction: torch.Tensor
    switch_rate: torch.Tensor


@dataclass
class StreamingHybridV2Output:
    enhanced: torch.Tensor
    base_enhanced: torch.Tensor
    waveform_residual: torch.Tensor
    gain: torch.Tensor
    phase_delta: torch.Tensor
    speech_features: Tuple[torch.Tensor, ...]
    noise_latent: torch.Tensor
    quantized_noise: torch.Tensor
    noise_prediction: torch.Tensor
    code_indices: torch.Tensor
    vq: EMAVQOutput


@dataclass
class BufferedStreamingV2State:
    """Correctness-first streaming state used until kernel caches are available."""

    waveform: torch.Tensor
    emitted_samples: int = 0


class CausalConv1d(nn.Module):
    """Conv1d with explicit left-only padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalConv2d(nn.Module):
    """Conv2d with symmetric frequency and left-only time padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        frequency_extent = dilation[0] * (kernel_size[0] - 1)
        self.frequency_padding = (
            frequency_extent // 2,
            frequency_extent - frequency_extent // 2,
        )
        self.time_padding = dilation[1] * (kernel_size[1] - 1)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_left, f_right = self.frequency_padding
        x = F.pad(x, (self.time_padding, 0, f_left, f_right))
        return self.conv(x)


class CausalDepthwiseResidual2d(nn.Module):
    def __init__(self, channels: int, time_dilation: int) -> None:
        super().__init__()
        self.depthwise = CausalConv2d(
            channels,
            channels,
            kernel_size=(3, 3),
            dilation=(1, time_dilation),
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.activation(self.norm(y))
        return x + y


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class CausalEncoder(nn.Module):
    def __init__(self, base_channels: int, latent_dim: int) -> None:
        super().__init__()
        widths = (base_channels, base_channels, base_channels * 2, latent_dim)
        inputs = (1, *widths[:-1])
        self.layers = nn.ModuleList(
            CausalConv1d(cin, cout, kernel_size=8, stride=2)
            for cin, cout in zip(inputs, widths)
        )
        self.activations = nn.ModuleList(nn.PReLU() for _ in widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for convolution, activation in zip(self.layers, self.activations):
            x = activation(convolution(x))
        return x


class CausalUpsampleDecoder(nn.Module):
    """Causal decoder using nearest-neighbour expansion and causal filtering."""

    def __init__(self, in_dim: int, base_channels: int) -> None:
        super().__init__()
        widths = (base_channels * 2, base_channels, base_channels, 1)
        inputs = (in_dim, *widths[:-1])
        self.layers = nn.ModuleList(
            CausalConv1d(cin, cout, kernel_size=7)
            for cin, cout in zip(inputs, widths)
        )
        self.activations = nn.ModuleList(
            [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.Tanh()]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for convolution, activation in zip(self.layers, self.activations):
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = activation(convolution(x))
        return x


class CausalTransposeDecoder(nn.Module):
    """Causal synthesis bank whose latent frame n starts at output 2*n.

    Each stage uses a length-four, stride-two transposed convolution and keeps
    exactly the first `2 * input_length` samples. A latent frame can therefore
    affect its current output position and later positions, never an earlier
    position. Cropping only removes the unfinalized right tail.
    """

    def __init__(self, in_dim: int, base_channels: int) -> None:
        super().__init__()
        widths = (base_channels * 2, base_channels, base_channels, 1)
        inputs = (in_dim, *widths[:-1])
        self.layers = nn.ModuleList(
            nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=0)
            for cin, cout in zip(inputs, widths)
        )
        self.activations = nn.ModuleList(
            [nn.PReLU(), nn.PReLU(), nn.PReLU(), nn.Tanh()]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for convolution, activation in zip(self.layers, self.activations):
            target_length = x.shape[-1] * 2
            x = activation(convolution(x)[..., :target_length])
        return x


class LookaheadTransposeDecoder(nn.Module):
    """Original high-quality synthesis bank with a bounded lookahead budget."""

    def __init__(self, in_dim: int, base_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(in_dim, base_channels * 2, 8, 2, 3),
            nn.PReLU(),
            nn.ConvTranspose1d(base_channels * 2, base_channels, 8, 2, 3),
            nn.PReLU(),
            nn.ConvTranspose1d(base_channels, base_channels, 8, 2, 3),
            nn.PReLU(),
            nn.ConvTranspose1d(base_channels, 1, 8, 2, 3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMANoiseVectorQuantizer(nn.Module):
    """EMA codebook with causal temporal pooling and dead-code replacement."""

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        update_interval: int = 2,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.update_interval = int(update_interval)
        self.commitment_weight = float(commitment_weight)
        self.decay = float(decay)
        self.eps = float(eps)
        self.dead_code_threshold = float(dead_code_threshold)
        self.update_codebook = True

        codebook = torch.empty(self.codebook_size, self.code_dim)
        nn.init.normal_(codebook, mean=0.0, std=self.code_dim ** -0.5)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_count", torch.zeros(self.codebook_size))
        self.register_buffer("ema_sum", codebook.clone())

    def _causal_pool(self, z: torch.Tensor) -> torch.Tensor:
        if self.update_interval == 1:
            return z
        # Include only the current and preceding latent frames. Padding repeats
        # the first observed state rather than introducing a learned future cue.
        z_channels = z.transpose(1, 2)
        z_channels = F.pad(
            z_channels,
            (self.update_interval - 1, 0),
            mode="replicate",
        )
        pooled = F.avg_pool1d(
            z_channels,
            kernel_size=self.update_interval,
            stride=self.update_interval,
        )
        return pooled.transpose(1, 2)

    @torch.no_grad()
    def _ema_update(
        self,
        flat: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        # EMA buffers deliberately stay in float32 even under autocast.
        flat = flat.to(self.codebook.dtype)
        assignments = F.one_hot(indices, self.codebook_size).to(flat.dtype)
        count = assignments.sum(dim=0)
        value_sum = assignments.transpose(0, 1) @ flat

        self.ema_count.mul_(self.decay).add_(count, alpha=1.0 - self.decay)
        self.ema_sum.mul_(self.decay).add_(value_sum, alpha=1.0 - self.decay)

        total = self.ema_count.sum()
        smoothed = (
            (self.ema_count + self.eps)
            / (total + self.codebook_size * self.eps)
            * total.clamp_min(1.0)
        )
        self.codebook.copy_(self.ema_sum / smoothed.unsqueeze(1).clamp_min(self.eps))

        dead = self.ema_count < self.dead_code_threshold
        if dead.any() and flat.shape[0] > 0:
            sample_ids = torch.randint(flat.shape[0], (int(dead.sum()),), device=flat.device)
            replacements = flat[sample_ids]
            self.codebook[dead] = replacements
            self.ema_sum[dead] = replacements
            self.ema_count[dead] = self.dead_code_threshold

    def forward(self, z: torch.Tensor) -> EMAVQOutput:
        # Input and output use [B, T, D]. Quantized states are held between
        # code updates so the returned temporal length matches the input.
        pooled = self._causal_pool(z)
        flat = pooled.reshape(-1, self.code_dim)
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.transpose(0, 1)
            + self.codebook.square().sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized_pooled = F.embedding(indices, self.codebook).view_as(pooled)

        if self.training and self.update_codebook:
            self._ema_update(flat.detach(), indices.detach())

        commitment = F.mse_loss(pooled, quantized_pooled.detach())
        loss = self.commitment_weight * commitment
        quantized_pooled = pooled + (quantized_pooled - pooled).detach()

        quantized = quantized_pooled.repeat_interleave(self.update_interval, dim=1)
        quantized = quantized[:, : z.shape[1]]
        expanded_indices = indices.view(pooled.shape[:2]).repeat_interleave(
            self.update_interval,
            dim=1,
        )[:, : z.shape[1]]

        probabilities = F.one_hot(indices, self.codebook_size).float().mean(dim=0)
        perplexity = torch.exp(-(probabilities * (probabilities + 1e-10).log()).sum())
        active = (probabilities > 0).float().mean()
        dead = (self.ema_count < self.dead_code_threshold).float().mean()
        if expanded_indices.shape[1] > 1:
            switches = (expanded_indices[:, 1:] != expanded_indices[:, :-1]).float().mean()
        else:
            switches = z.new_tensor(0.0)

        return EMAVQOutput(
            quantized=quantized,
            indices=expanded_indices,
            loss=loss,
            commitment_loss=commitment,
            codebook_loss=z.new_tensor(0.0),
            perplexity=perplexity,
            active_fraction=active,
            dead_fraction=dead,
            switch_rate=switches,
        )


class CausalSTFT(nn.Module):
    """Uncentred STFT pair with a nonzero-endpoint synthesis window."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        if win_length != n_fft:
            raise ValueError("StreamingHybridV2 currently requires win_length == n_fft")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.register_buffer(
            "window",
            torch.hamming_window(self.win_length, periodic=True),
            persistent=False,
        )

    def analysis(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        original_length = waveform.shape[-1]
        if original_length < self.win_length:
            right_pad = self.win_length - original_length
        else:
            remainder = (original_length - self.win_length) % self.hop_length
            right_pad = (self.hop_length - remainder) % self.hop_length
        waveform = F.pad(waveform, (0, right_pad))
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=False,
            return_complex=True,
        )
        return spectrum, original_length

    def synthesis(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        padded_length = self.win_length + self.hop_length * (spectrum.shape[-1] - 1)
        waveform = torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=spectrum.device, dtype=spectrum.real.dtype),
            center=False,
            length=padded_length,
        )
        return waveform[..., :length]


class StreamingHybridCNVQGV2(nn.Module):
    """Causal hybrid waveform/TF Mamba enhancer with an EMA noise codebook."""

    PRESETS: Dict[str, Dict[str, int]] = {
        "teacher": {
            "encoder_channels": 256,
            "latent_dim": 512,
            "speech_dim": 512,
            "noise_dim": 256,
            "waveform_temporal_layers": 10,
            "codebook_size": 256,
            "tf_hidden_dim": 64,
            "tf_temporal_layers": 4,
            "frequency_subbands": 16,
        },
        "student": {
            "encoder_channels": 128,
            "latent_dim": 256,
            "speech_dim": 256,
            "noise_dim": 128,
            "waveform_temporal_layers": 6,
            "codebook_size": 128,
            "tf_hidden_dim": 32,
            "tf_temporal_layers": 2,
            "frequency_subbands": 8,
        },
        "tiny": {
            "encoder_channels": 80,
            "latent_dim": 160,
            "speech_dim": 160,
            "noise_dim": 80,
            "waveform_temporal_layers": 4,
            "codebook_size": 64,
            "tf_hidden_dim": 24,
            "tf_temporal_layers": 2,
            "frequency_subbands": 8,
        },
    }

    def __init__(
        self,
        variant: str = "student",
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        encoder_channels: int | None = None,
        latent_dim: int | None = None,
        speech_dim: int | None = None,
        noise_dim: int | None = None,
        waveform_temporal_layers: int | None = None,
        codebook_size: int | None = None,
        tf_hidden_dim: int | None = None,
        tf_temporal_layers: int | None = None,
        frequency_subbands: int | None = None,
        use_mamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        vq_update_interval: int = 2,
        vq_commitment_weight: float = 0.25,
        vq_decay: float = 0.99,
        residual_scale: float = 0.2,
        gain_scale: float = 0.2,
        phase_scale: float = 0.08,
        predict_phase: bool = True,
        decoder_type: str = "upsample",
        enable_tf_refiner: bool = True,
        encoder_type: str = "causal",
    ) -> None:
        super().__init__()
        if variant not in self.PRESETS:
            raise ValueError(f"Unknown StreamingHybridV2 variant: {variant}")
        preset = self.PRESETS[variant]

        def choose(name: str, value: int | None) -> int:
            return int(preset[name] if value is None else value)

        self.variant = variant
        self.sample_rate = int(sample_rate)
        self.encoder_channels = choose("encoder_channels", encoder_channels)
        self.latent_dim = choose("latent_dim", latent_dim)
        self.speech_dim = choose("speech_dim", speech_dim)
        self.noise_dim = choose("noise_dim", noise_dim)
        self.waveform_temporal_layers = choose(
            "waveform_temporal_layers", waveform_temporal_layers
        )
        self.codebook_size = choose("codebook_size", codebook_size)
        self.tf_hidden_dim = choose("tf_hidden_dim", tf_hidden_dim)
        self.tf_temporal_layers = choose("tf_temporal_layers", tf_temporal_layers)
        self.frequency_subbands = choose("frequency_subbands", frequency_subbands)
        self.residual_scale = float(residual_scale)
        self.gain_scale = float(gain_scale)
        self.phase_scale = float(phase_scale)
        self.predict_phase = bool(predict_phase)
        self.decoder_type = str(decoder_type)
        self.enable_tf_refiner = bool(enable_tf_refiner)
        self.encoder_type = str(encoder_type)

        if self.encoder_type == "causal":
            self.encoder = CausalEncoder(self.encoder_channels, self.latent_dim)
        elif self.encoder_type == "lookahead":
            self.encoder = ConvEncoder(
                in_channels=1,
                base_channels=self.encoder_channels,
                latent_dim=self.latent_dim,
            )
        else:
            raise ValueError(f"Unknown waveform encoder type: {self.encoder_type}")
        self.speech_projection = nn.Conv1d(self.latent_dim, self.speech_dim, 1)
        self.noise_projection = nn.Conv1d(self.latent_dim, self.noise_dim, 1)

        self.waveform_temporal = TemporalStack(
            dim=self.speech_dim,
            layers=self.waveform_temporal_layers,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.noise_vq = EMANoiseVectorQuantizer(
            self.codebook_size,
            self.noise_dim,
            update_interval=vq_update_interval,
            commitment_weight=vq_commitment_weight,
            decay=vq_decay,
        )
        self.noise_to_film = nn.Conv1d(self.noise_dim, self.speech_dim * 2, 1)
        self.noise_predictor = nn.Sequential(
            nn.Conv1d(self.noise_dim, self.noise_dim, 1),
            nn.SiLU(),
            nn.Conv1d(self.noise_dim, n_fft // 2 + 1, 1),
            nn.Softplus(),
        )
        if self.decoder_type == "upsample":
            self.waveform_decoder = CausalUpsampleDecoder(
                self.speech_dim,
                self.encoder_channels,
            )
        elif self.decoder_type == "causal_transpose":
            self.waveform_decoder = CausalTransposeDecoder(
                self.speech_dim,
                self.encoder_channels,
            )
        elif self.decoder_type == "lookahead_transpose":
            self.waveform_decoder = LookaheadTransposeDecoder(
                self.speech_dim,
                self.encoder_channels,
            )
        else:
            raise ValueError(f"Unknown waveform decoder type: {self.decoder_type}")

        self.stft = CausalSTFT(n_fft, hop_length, win_length)
        tf_input_channels = 5 + self.noise_dim
        self.tf_input = nn.Sequential(
            CausalConv2d(tf_input_channels, self.tf_hidden_dim, (3, 3)),
            nn.GroupNorm(_group_count(self.tf_hidden_dim), self.tf_hidden_dim),
            nn.SiLU(),
        )
        self.tf_blocks = nn.Sequential(
            CausalDepthwiseResidual2d(self.tf_hidden_dim, 1),
            CausalDepthwiseResidual2d(self.tf_hidden_dim, 2),
            CausalDepthwiseResidual2d(self.tf_hidden_dim, 4),
        )
        self.tf_temporal = TemporalStack(
            dim=self.tf_hidden_dim,
            layers=self.tf_temporal_layers,
            use_mamba=use_mamba,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.cross_band = nn.Sequential(
            nn.Conv2d(self.tf_hidden_dim, self.tf_hidden_dim, 1),
            nn.SiLU(),
        )
        output_channels = 2 if self.predict_phase else 1
        self.tf_output = nn.Sequential(
            CausalDepthwiseResidual2d(self.tf_hidden_dim, 1),
            CausalConv2d(self.tf_hidden_dim, output_channels, (3, 3)),
        )
        final = self.tf_output[-1].conv
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if not self.enable_tf_refiner:
            for module in (
                self.tf_input,
                self.tf_blocks,
                self.tf_temporal,
                self.cross_band,
                self.tf_output,
            ):
                module.requires_grad_(False)

    @property
    def temporal(self) -> TemporalStack:
        """Compatibility with existing training and evaluation scripts."""
        return self.waveform_temporal

    def _align_noise_to_tf(
        self,
        quantized_noise: torch.Tensor,
        frames: int,
        bins: int,
    ) -> torch.Tensor:
        noise = quantized_noise.transpose(1, 2)
        noise = F.interpolate(noise, size=frames, mode="nearest")
        return noise.unsqueeze(2).expand(-1, -1, bins, -1)

    def forward(self, noisy: torch.Tensor) -> StreamingHybridV2Output:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError(f"Expected noisy [B, 1, T], got {tuple(noisy.shape)}")
        original_length = noisy.shape[-1]
        latent = self.encoder(noisy)
        speech = self.speech_projection(latent).transpose(1, 2)
        speech_features = []
        for index, block in enumerate(self.waveform_temporal.blocks):
            speech = block(speech)
            if index in {0, self.waveform_temporal_layers // 2, self.waveform_temporal_layers - 1}:
                speech_features.append(speech)

        noise_latent = self.noise_projection(latent).transpose(1, 2)
        vq = self.noise_vq(noise_latent)
        film = self.noise_to_film(vq.quantized.transpose(1, 2))
        gamma, beta = film.chunk(2, dim=1)
        conditioned = speech.transpose(1, 2) * (1.0 + torch.tanh(gamma)) + beta
        waveform_residual = self.waveform_decoder(conditioned)[..., :original_length]
        base_enhanced = noisy[..., : waveform_residual.shape[-1]] + self.residual_scale * waveform_residual

        noise_prediction = self.noise_predictor(vq.quantized.transpose(1, 2))
        if not self.enable_tf_refiner:
            empty_tf = base_enhanced.new_empty(
                base_enhanced.shape[0], self.stft.n_fft // 2 + 1, 0
            )
            return StreamingHybridV2Output(
                enhanced=base_enhanced,
                base_enhanced=base_enhanced,
                waveform_residual=waveform_residual,
                gain=empty_tf,
                phase_delta=empty_tf,
                speech_features=tuple(speech_features),
                noise_latent=noise_latent,
                quantized_noise=vq.quantized,
                noise_prediction=noise_prediction,
                code_indices=vq.indices,
                vq=vq,
            )

        base_wave = base_enhanced.squeeze(1)
        noisy_spectrum, _ = self.stft.analysis(noisy.squeeze(1))
        base_spectrum, _ = self.stft.analysis(base_wave)
        # Learned analysis/synthesis banks can differ by one sample for
        # arbitrary utterance lengths, which becomes one STFT frame at a hop
        # boundary.  TF refinement operates only on frames present in both
        # signals; synthesis still restores the requested original length.
        shared_frames = min(noisy_spectrum.shape[-1], base_spectrum.shape[-1])
        noisy_spectrum = noisy_spectrum[..., :shared_frames]
        base_spectrum = base_spectrum[..., :shared_frames]
        noisy_magnitude = noisy_spectrum.abs().clamp_min(1e-7)
        base_magnitude = base_spectrum.abs().clamp_min(1e-7)
        base_cos = base_spectrum.real / base_magnitude
        base_sin = base_spectrum.imag / base_magnitude
        base_log = torch.log1p(base_magnitude)
        noisy_log = torch.log1p(noisy_magnitude)

        scalar_features = torch.stack(
            (noisy_log, base_log, base_log - noisy_log, base_cos, base_sin),
            dim=1,
        )
        noise_tf = self._align_noise_to_tf(
            vq.quantized,
            frames=base_spectrum.shape[-1],
            bins=base_spectrum.shape[-2],
        )
        tf = self.tf_blocks(self.tf_input(torch.cat((scalar_features, noise_tf), dim=1)))

        bands = F.adaptive_avg_pool2d(
            tf,
            output_size=(self.frequency_subbands, tf.shape[-1]),
        )
        batch, channels, num_bands, frames = bands.shape
        band_sequences = bands.permute(0, 2, 3, 1).reshape(batch * num_bands, frames, channels)
        band_sequences = self.tf_temporal(band_sequences)
        bands = band_sequences.view(batch, num_bands, frames, channels).permute(0, 3, 1, 2)
        band_context = F.interpolate(
            bands,
            size=(tf.shape[-2], tf.shape[-1]),
            mode="nearest",
        )
        tf = tf + self.cross_band(band_context)
        correction = self.tf_output(tf)
        gain = torch.exp(self.gain_scale * torch.tanh(correction[:, 0]))
        if self.predict_phase:
            phase_delta = self.phase_scale * torch.tanh(correction[:, 1])
        else:
            phase_delta = torch.zeros_like(gain)

        refined = torch.polar(base_magnitude * gain, torch.angle(base_spectrum) + phase_delta)
        enhanced = self.stft.synthesis(refined, original_length).unsqueeze(1)
        return StreamingHybridV2Output(
            enhanced=enhanced,
            base_enhanced=base_enhanced,
            waveform_residual=waveform_residual,
            gain=gain,
            phase_delta=phase_delta,
            speech_features=tuple(speech_features),
            noise_latent=noise_latent,
            quantized_noise=vq.quantized,
            noise_prediction=noise_prediction,
            code_indices=vq.indices,
            vq=vq,
        )

    def init_streaming_state(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> BufferedStreamingV2State:
        return BufferedStreamingV2State(
            waveform=torch.empty(batch_size, 1, 0, device=device, dtype=dtype)
        )

    @torch.no_grad()
    def stream_step(
        self,
        audio_chunk: torch.Tensor,
        state: BufferedStreamingV2State,
    ) -> tuple[torch.Tensor, BufferedStreamingV2State]:
        """Emit finalized samples using the causal full-prefix reference path.

        This API is numerically authoritative but intentionally recomputes the
        observed prefix. Production inference must replace its growing buffer
        with convolution, Mamba, and overlap-add caches without changing output.
        """
        if audio_chunk.ndim != 3 or audio_chunk.shape[:2] != state.waveform.shape[:2]:
            raise ValueError("Streaming chunk and state batch/channel shapes differ")
        waveform = torch.cat((state.waveform, audio_chunk), dim=-1)
        stable_length = max(0, waveform.shape[-1] - self.stft.win_length)
        if stable_length <= state.emitted_samples:
            emitted = waveform.new_empty(waveform.shape[0], 1, 0)
        else:
            enhanced = self(waveform).enhanced
            emitted = enhanced[..., state.emitted_samples:stable_length]
        return emitted, BufferedStreamingV2State(waveform, stable_length)

    @torch.no_grad()
    def flush_stream(
        self,
        state: BufferedStreamingV2State,
    ) -> tuple[torch.Tensor, BufferedStreamingV2State]:
        if state.waveform.shape[-1] == 0:
            return state.waveform, state
        enhanced = self(state.waveform).enhanced
        tail = enhanced[..., state.emitted_samples:]
        return tail, BufferedStreamingV2State(
            waveform=state.waveform.new_empty(state.waveform.shape[0], 1, 0),
            emitted_samples=0,
        )

    @torch.no_grad()
    def enhance_offline(
        self,
        waveform: torch.Tensor,
        chunk_samples: int = 256,
    ) -> torch.Tensor:
        state = self.init_streaming_state(
            waveform.shape[0], waveform.device, waveform.dtype
        )
        pieces = []
        for start in range(0, waveform.shape[-1], chunk_samples):
            piece, state = self.stream_step(
                waveform[..., start:start + chunk_samples], state
            )
            pieces.append(piece)
        tail, _ = self.flush_stream(state)
        pieces.append(tail)
        return torch.cat(pieces, dim=-1)


class StreamingHybridV2Teacher(StreamingHybridCNVQGV2):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="teacher", **kwargs)


class StreamingHybridV2Student(StreamingHybridCNVQGV2):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="student", **kwargs)


class StreamingHybridV2Tiny(StreamingHybridCNVQGV2):
    def __init__(self, **kwargs) -> None:
        super().__init__(variant="tiny", **kwargs)
