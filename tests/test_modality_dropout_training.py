"""Exercise the public config and Trainer on CPU, without pretrained downloads."""
import importlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from config import get_config_args
from train import Trainer


def config(monkeypatch, tmp_path, overrides=()):
    monkeypatch.setenv('NANOPARTICLES_KI_DATASET_DIR', str(tmp_path))
    monkeypatch.setattr('sys.argv', ['train.py', '-c'] + list(overrides))
    importlib.invalidate_caches()
    module = importlib.import_module('config.nanoparticles.train')
    importlib.reload(module)
    return get_config_args('nanoparticles', 'train')


def test_config_is_disabled_by_default(monkeypatch, tmp_path):
    args = config(monkeypatch, tmp_path)
    assert args['modality_dropout'] == {'bf_drop_probability': 0.0, 'haadf_drop_probability': 0.0}


def test_existing_config_override_enables_dropout(monkeypatch, tmp_path):
    args = config(monkeypatch, tmp_path, [
        'modality_dropout.bf_drop_probability=0.25',
        'modality_dropout.haadf_drop_probability=0.25', 'seed=17'])
    assert args['modality_dropout'] == {'bf_drop_probability': .25, 'haadf_drop_probability': .25}
    assert args['seed'] == 17


@pytest.mark.parametrize('dropout', [
    {'bf_drop_probability': -.1}, {'haadf_drop_probability': float('nan')},
    {'bf_drop_probability': .6, 'haadf_drop_probability': .5},
])
def test_invalid_config_fails_before_writing_run_files(tmp_path, dropout):
    args = {'modality_dropout': dropout, 'save': True, 'save_dir': str(tmp_path / 'out')}
    with pytest.raises(ValueError, match='probabilit'):
        Trainer(0, 0, 1, args, use_distributed_data_parallel=False)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('dropout', [
    {'bf_drop_probability': .25}, {'haadf_drop_probability': .25},
    {'bf_drop_probability': .25, 'haadf_drop_probability': .25},
])
def test_enabled_dropout_rejects_cached_outputs_before_run_files(tmp_path, dropout):
    args = {
        'modality_dropout': dropout,
        'train_dataset': {'centerdir_gt_opts': {'use_cached_backbone_output': True}},
        'save': True, 'save_dir': str(tmp_path / 'out'), 'display': False,
    }
    with pytest.raises(ValueError, match='modality dropout.*cached backbone outputs'):
        Trainer(0, 0, 1, args, use_distributed_data_parallel=False)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('dropout', [None, {},
    {'bf_drop_probability': 0.0, 'haadf_drop_probability': 0.0},
])
def test_disabled_dropout_allows_cached_outputs(tmp_path, dropout):
    args = {'train_dataset': {'centerdir_gt_opts': {'use_cached_backbone_output': True}},
            'save': False, 'display': False}
    if dropout is not None:
        args['modality_dropout'] = dropout
    trainer = Trainer(0, 0, 1, args, use_distributed_data_parallel=False)
    assert trainer.modality_dropout_probabilities == (0.0, 0.0)


@pytest.mark.parametrize('gt_opts', [None, {}, {'use_cached_backbone_output': False}])
def test_enabled_dropout_allows_missing_or_disabled_cache_option(gt_opts):
    args = {'modality_dropout': {'bf_drop_probability': .25},
            'train_dataset': {'centerdir_gt_opts': gt_opts},
            'save': False, 'display': False}
    trainer = Trainer(0, 0, 1, args, use_distributed_data_parallel=False)
    assert trainer.modality_dropout_probabilities == (.25, 0.0)


def make_trainer(monkeypatch, tmp_path, dropout=None, rank=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    yy, xx = np.ogrid[:64, :64]
    masks = np.zeros((64, 64, 2), dtype=np.uint8)
    masks[:, :, 0] = (yy - 20)**2 + (xx - 20)**2 <= 6**2
    masks[:, :, 1] = (yy - 44)**2 + (xx - 42)**2 <= 5**2
    for i in range(4):
        stem = tmp_path / ('sample%d' % i)
        Image.fromarray(np.full((64, 64), 90 + i, dtype=np.uint8)).save(str(stem) + '_BF.png')
        Image.fromarray(np.full((64, 64), 170 + i, dtype=np.uint8)).save(str(stem) + '_HAADF.png')
        np.savez_compressed(str(stem) + '_masks.npz', masks)
    args = config(monkeypatch, tmp_path)
    args.update(cuda=False, save=False, save_dir=str(tmp_path / 'out'), n_epochs=1, seed=17)
    if dropout is None:
        args.pop('modality_dropout', None)  # Exercise old callers with no new config.
    else:
        args['modality_dropout'] = dropout
    db = args['train_dataset']
    db.update(workers=0, batch_size=2, shuffle=False)
    db['kwargs'].update(root_dir=str(tmp_path), subfolders=['.'], valid_sample_names=None,
                        MAX_NUM_CENTERS=32)
    # Include the existing ColorJitter after spatial transforms. Its additive
    # effect must not resurrect a detector removed by the later Trainer hook.
    for t in db['kwargs']['transform']:
        if t['name'] == 'Resize':
            t['opts']['size'] = (64, 64)
        if t['name'] == 'ColorJitter':
            t['opts']['p'] = 1.0
    db['centerdir_gt_opts']['MAX_NUM_CENTERS'] = 32
    args['model']['kwargs'].update(backbone='resnet18', pretrained=False)
    args['model']['kwargs']['fpn_args'].update(encoder_depth=4, upsampling=2)
    args['multitask_weighting'] = {'name': 'off'}
    trainer = Trainer(0, rank, 1, args, use_distributed_data_parallel=False)
    trainer.initialize_data_parallel()
    trainer.initialize()
    return trainer


@pytest.mark.parametrize('dropped', [0, 1])
def test_real_training_zeros_missing_channel_after_augmentation(monkeypatch, tmp_path, dropped):
    torch.manual_seed(41)
    trainer = make_trainer(monkeypatch, tmp_path, {
        'bf_drop_probability': float(dropped == 0),
        'haadf_drop_probability': float(dropped == 1)})
    import train as train_module
    raw = []
    apply = train_module.apply_modality_dropout
    def capture(image, *args, **kwargs):
        raw.append(image.detach().clone())
        return apply(image, *args, **kwargs)
    monkeypatch.setattr(train_module, 'apply_modality_dropout', capture)
    seen = []
    trainer.model.module.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    loss = trainer.train(0)
    assert np.isfinite(loss)
    assert len(seen) == len(raw) == 2
    assert all(torch.count_nonzero(x[:, dropped]) == 0 for x in seen)
    assert all(torch.equal(x[:, 1 - dropped], y[:, 1 - dropped]) for x, y in zip(seen, raw))
    assert all(torch.equal(x[:, 2], y[:, 2]) for x, y in zip(seen, raw))
    assert any(not torch.equal(v, trainer.model.state_dict()[k]) for k, v in before.items())
    assert any(trainer.train_dataset_it.dataset[i]['image'][dropped].count_nonzero() > 0
               for i in range(4))


def test_explicitly_disabled_matches_old_config_weights_loss_and_rng(monkeypatch, tmp_path):
    def run(path, dropout):
        torch.manual_seed(47)
        np.random.seed(47)
        trainer = make_trainer(monkeypatch, path, dropout)
        loss = trainer.train(0)
        return loss, trainer.model.state_dict(), torch.get_rng_state(), np.random.get_state()
    old = run(tmp_path / 'old', None)
    off = run(tmp_path / 'off', {'bf_drop_probability': 0, 'haadf_drop_probability': 0})
    assert old[0] == off[0]
    assert all(torch.equal(v, off[1][k]) for k, v in old[1].items())
    assert torch.equal(old[2], off[2])
    assert np.array_equal(old[3][1], off[3][1])


@pytest.mark.parametrize('rank,epoch', [(0, 0), (2, 3)])
def test_trainer_reuses_private_stream_across_batches(monkeypatch, tmp_path, rank, epoch):
    import train as train_module
    from stem_modality import create_modality_dropout_generator
    torch.manual_seed(59)
    trainer = make_trainer(monkeypatch, tmp_path, {
        'bf_drop_probability': .25, 'haadf_drop_probability': .25}, rank=rank)
    streams = []
    apply = train_module.apply_modality_dropout
    expected = create_modality_dropout_generator(17 + rank, epoch)
    def capture(image, *args, **kwargs):
        generator = kwargs['generator']
        streams.append(generator)
        assert torch.equal(generator.get_state(), expected.get_state())
        global_state = torch.get_rng_state().clone()
        result = apply(image, *args, **kwargs)
        torch.rand(len(image), generator=expected)
        assert torch.equal(generator.get_state(), expected.get_state())
        assert torch.equal(torch.get_rng_state(), global_state)
        return result
    monkeypatch.setattr(train_module, 'apply_modality_dropout', capture)
    assert np.isfinite(trainer.train(epoch))
    assert len(streams) == 2 and streams[0] is streams[1]


def test_run_records_dropout_without_changing_model_state_keys(monkeypatch, tmp_path):
    torch.manual_seed(53)
    trainer = make_trainer(monkeypatch, tmp_path, {
        'bf_drop_probability': .25, 'haadf_drop_probability': .25})
    trainer.args['save'] = True
    original_keys = set(trainer.model.state_dict())
    trainer.run()
    saved = torch.load(str(tmp_path / 'out' / 'checkpoint.pth'), map_location='cpu')
    assert set(saved['model_state_dict']) == original_keys
    assert saved['modality_dropout']['bf_drop_probability'] == .25
    assert saved['modality_dropout']['haadf_drop_probability'] == .25
    assert saved['modality_dropout']['fill_value'] == 0
    assert saved['modality_dropout']['rng']['seed'] == 17
    # The normal Trainer JSON path also preserves the user configuration.
    trainer.args['save_dir'] = str(tmp_path / 'params')
    Trainer(0, 0, 1, trainer.args, use_distributed_data_parallel=False)
    assert json.loads((tmp_path / 'params' / 'params.json').read_text())['modality_dropout'] == trainer.args['modality_dropout']
