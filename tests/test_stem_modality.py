"""Frozen BF/HAADF raw-input contract for trainers and external adapters."""
import importlib

import pytest
import torch


def helper():
    return importlib.import_module("stem_modality")


def image(batch=None, scale=255.0, dtype=torch.float32):
    result = torch.stack((torch.full((3, 4), scale / 4, dtype=dtype),
                          torch.full((3, 4), scale / 2, dtype=dtype),
                          torch.zeros((3, 4), dtype=dtype)))
    return result if batch is None else result.repeat(batch, 1, 1, 1)


@pytest.mark.parametrize("bf,haadf", [(0, 0), (1, 0), (0, 1), (.2, .3), (.5, .5)])
def test_valid_probabilities(bf, haadf):
    assert helper().validate_probabilities(bf, haadf) == (float(bf), float(haadf))


@pytest.mark.parametrize("bf,haadf", [(-.1, 0), (0, -.1), (1.1, 0), (.6, .5),
                                    (float("nan"), 0), (0, float("inf")),
                                    (float("-inf"), 0), ("bad", 0), (None, 0)])
def test_invalid_probabilities_fail_closed(bf, haadf):
    with pytest.raises(ValueError, match="probabilit"):
        helper().validate_probabilities(bf, haadf)


@pytest.mark.parametrize("batch", [None, 5])
@pytest.mark.parametrize("scale", [1.0, 255.0])
@pytest.mark.parametrize("mode,drop", [("paired", None), ("bf-only", 1), ("haadf-only", 0)])
def test_input_modes_preserve_raw_values_and_caller(batch, scale, mode, drop):
    original = image(batch, scale, torch.float64)
    before = original.clone()
    result = helper().apply_input_mode(original, mode)
    expected = before.clone()
    if drop is not None:
        expected.select(-3, drop).zero_()
    assert torch.equal(result, expected)
    assert torch.equal(original, before)
    assert result.dtype == original.dtype and result.device == original.device


@pytest.mark.parametrize("value", [torch.ones(3, 4), torch.ones(4, 3, 4),
                                    torch.ones(1, 2, 3, 4), torch.ones(0, 3, 4, 4),
                                    torch.ones(3, 0, 4), torch.ones(3, 4, 4, dtype=torch.long), []])
@pytest.mark.parametrize("operation", ["apply_input_mode", "apply_modality_dropout"])
def test_invalid_input_shape_or_dtype_is_rejected(value, operation):
    args = (value, "paired") if operation == "apply_input_mode" else (value,)
    with pytest.raises(ValueError, match="image"):
        getattr(helper(), operation)(*args)


@pytest.mark.parametrize("mode", ["BF", "haadf_only", "", None])
def test_invalid_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="mode"):
        helper().apply_input_mode(image(), mode)


def test_default_dropout_is_identity_without_any_rng_consumption():
    original = image(4)
    generator = torch.Generator().manual_seed(17)
    local_before = generator.get_state().clone()
    global_before = torch.get_rng_state().clone()
    assert helper().apply_modality_dropout(original, generator=generator) is original
    assert torch.equal(generator.get_state(), local_before)
    assert torch.equal(torch.get_rng_state(), global_before)


@pytest.mark.parametrize("batch", [None, 7])
@pytest.mark.parametrize("bf,haadf,mode", [(1, 0, "haadf-only"), (0, 1, "bf-only")])
def test_endpoint_dropout_matches_deterministic_mode(batch, bf, haadf, mode):
    original = image(batch)
    result = helper().apply_modality_dropout(original, bf, haadf,
                                             generator=torch.Generator().manual_seed(19))
    assert torch.equal(result, helper().apply_input_mode(original, mode))
    assert torch.equal(original, image(batch))


def test_categorical_draws_are_per_sample_never_both_and_do_not_rescale():
    original = image(4096)
    generator = torch.Generator().manual_seed(31)
    expected_draws = torch.rand(4096, generator=torch.Generator().manual_seed(31))
    result = helper().apply_modality_dropout(original, .2, .3, generator=generator)
    bf_dropped = (result[:, 0] == 0).flatten(1).all(1)
    haadf_dropped = (result[:, 1] == 0).flatten(1).all(1)
    assert torch.equal(bf_dropped, expected_draws < .2)
    assert torch.equal(haadf_dropped, (expected_draws >= .2) & (expected_draws < .5))
    assert not (bf_dropped & haadf_dropped).any()
    assert bf_dropped.any() and haadf_dropped.any() and (~(bf_dropped | haadf_dropped)).any()
    assert torch.equal(result[~bf_dropped, 0], original[~bf_dropped, 0])
    assert torch.equal(result[~haadf_dropped, 1], original[~haadf_dropped, 1])
    assert torch.count_nonzero(result[:, 2]) == 0
    assert torch.equal(original, image(4096))


def test_dropout_preserves_autograd_and_accepts_noncontiguous_inputs():
    original = image(16).transpose(-1, -2).requires_grad_()
    result = helper().apply_modality_dropout(original, .5, .5,
                                             generator=torch.Generator().manual_seed(37))
    result.sum().backward()
    assert torch.equal(original.grad[:, :2], (result[:, :2] != 0).to(original.dtype))
    assert torch.equal(original.grad[:, 2], torch.ones_like(original.grad[:, 2]))


def test_seed_epoch_generator_is_repeatable_and_isolated():
    module = helper()
    global_before = torch.get_rng_state().clone()
    first = module.create_modality_dropout_generator(41, 3)
    repeated = module.create_modality_dropout_generator(41, 3)
    other = module.create_modality_dropout_generator(41, 4)
    assert first.device.type == "cpu"
    assert first.initial_seed() == 44
    assert torch.equal(first.get_state(), repeated.get_state())
    assert not torch.equal(first.get_state(), other.get_state())
    module.apply_modality_dropout(image(32), .2, .3, generator=first)
    assert torch.equal(torch.get_rng_state(), global_before)


def test_dropout_keeps_nonzero_auxiliary_plane():
    original = image(8)
    original[:, 2] = 13
    result = helper().apply_modality_dropout(original, .5, .5,
                                             generator=helper().create_modality_dropout_generator(19))
    assert torch.equal(result[:, 2], original[:, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_cuda_mask_matches_cpu_and_preserves_gradients(dtype):
    original = image(32, dtype=dtype).cuda().requires_grad_()
    result = helper().apply_modality_dropout(original, .25, .25,
                                             generator=helper().create_modality_dropout_generator(61))
    expected = helper().apply_modality_dropout(original.detach().cpu(), .25, .25,
                                               generator=helper().create_modality_dropout_generator(61))
    assert torch.equal(result.cpu(), expected)
    result.sum().backward()
    assert torch.equal(original.grad[:, :2], (result[:, :2] != 0).to(dtype))
    assert torch.equal(original.grad[:, 2], torch.ones_like(original.grad[:, 2]))
    assert result.device == original.device and result.dtype == dtype


def test_metadata_freezes_policy_and_rng_contract():
    assert helper().modality_dropout_metadata(.2, .3, 17) == {
        "bf_drop_probability": .2, "haadf_drop_probability": .3,
        "policy": "categorical-per-sample", "fill_value": 0.0,
        "rescale_survivor": False,
        "rng": {"algorithm": "torch-cpu-generator", "seed": 17,
                "epoch_seed": "(seed + zero_based_epoch) % (2**63 - 1)"},
    }
