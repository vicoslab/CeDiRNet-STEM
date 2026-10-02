"""Whole-channel BF/HAADF ablation before the model's input normalization.

Floating CHW/BCHW inputs use [BF, HAADF, auxiliary] slots. Raw or unit-scale
intensities work; the auxiliary channel is preserved, not repaired here.
"""
import math

import torch


def validate_probabilities(bf_drop_probability=0.0, haadf_drop_probability=0.0):
    """Return finite nonnegative floats whose categorical sum is at most one."""
    try:
        bf, haadf = float(bf_drop_probability), float(haadf_drop_probability)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("modality dropout probabilities must be finite numbers") from error
    if not (math.isfinite(bf) and math.isfinite(haadf)) or min(bf, haadf) < 0 or bf + haadf > 1:
        raise ValueError("modality dropout probabilities must be finite, nonnegative and sum <= 1")
    return bf, haadf


def _validate_image(image):
    if (not isinstance(image, torch.Tensor) or not image.is_floating_point()
            or image.ndim not in (3, 4) or image.shape[-3] != 3
            or any(size == 0 for size in image.shape)):
        raise ValueError("image must be a nonempty floating CHW or BCHW tensor with 3 channels")


def apply_input_mode(image, mode):
    """Keep the pair, BF only, or HAADF only without mutating the caller."""
    _validate_image(image)
    if mode not in ("paired", "bf-only", "haadf-only"):
        raise ValueError("input mode must be paired, bf-only or haadf-only")
    if mode == "paired":
        return image
    output = image.clone()
    output.select(-3, 1 if mode == "bf-only" else 0).zero_()
    return output


def apply_modality_dropout(image, bf_drop_probability=0.0, haadf_drop_probability=0.0, generator=None):
    """Draw per sample: drop BF, drop HAADF, or keep both; never drop both.

    Zero probabilities return the original tensor and consume no RNG. Enabled
    dropout uses one broadcast mask and one output allocation, preserving
    retained values, gradients, dtype/device and the caller's tensor.
    """
    bf, haadf = validate_probabilities(bf_drop_probability, haadf_drop_probability)
    _validate_image(image)
    if bf == 0 and haadf == 0:
        return image
    batch = image.unsqueeze(0) if image.ndim == 3 else image
    draw_device = generator.device if generator is not None else torch.device("cpu")
    draws = torch.rand(batch.shape[0], generator=generator, device=draw_device)
    mask = torch.stack((draws < bf, (draws >= bf) & (draws < bf + haadf),
                        torch.zeros_like(draws, dtype=torch.bool)), dim=1)
    output = batch.masked_fill(mask.to(image.device)[:, :, None, None], 0)
    return output.squeeze(0) if image.ndim == 3 else output


def create_modality_dropout_generator(seed=0, epoch=0):
    """Dedicated CPU stream; recreate once per epoch, not once per batch."""
    return torch.Generator(device="cpu").manual_seed((int(seed) + int(epoch)) % (2**63 - 1))


def modality_dropout_metadata(bf_drop_probability=0.0, haadf_drop_probability=0.0, seed=0):
    """JSON-safe policy record; no model parameters or buffers are added."""
    bf, haadf = validate_probabilities(bf_drop_probability, haadf_drop_probability)
    return {
        "bf_drop_probability": bf,
        "haadf_drop_probability": haadf,
        "policy": "categorical-per-sample",
        "fill_value": 0.0,
        "rescale_survivor": False,
        "rng": {"algorithm": "torch-cpu-generator", "seed": int(seed),
                "epoch_seed": "(seed + zero_based_epoch) % (2**63 - 1)"},
    }
