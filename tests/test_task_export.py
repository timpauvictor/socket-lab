"""CPU export behavior; no model download, CUDA, PEFT or running server."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
safetensors = pytest.importorskip('safetensors.torch')

from training.export_task_adapter import export_adapter, main

MODEL = 'Qwen/Qwen3-0.6B-Base'
PREFIX = 'base_model.model.model.layers.0.self_attn.q_proj'


def adapter(path: Path, rank: int, seed: int) -> dict:
    path.mkdir()
    config = dict(base_model_name_or_path=MODEL, peft_type='LORA', task_type='CAUSAL_LM',
                  bias='none', r=rank, lora_alpha=rank*2, lora_dropout=0.05,
                  target_modules=['q_proj','k_proj','v_proj','o_proj'])
    generator = torch.Generator().manual_seed(seed)
    state = {PREFIX+'.lora_A.weight': torch.randn(rank, 3, generator=generator),
             PREFIX+'.lora_B.weight': torch.randn(4, rank, generator=generator)}
    (path/'adapter_config.json').write_text(json.dumps(config))
    safetensors.save_file(state, str(path/'adapter_model.safetensors'))
    return state


def test_cpu_export_preserves_linear_cross_terms_plus_task_and_sources(tmp_path):
    paths = [tmp_path/name for name in ('ro','en','task')]
    states = [adapter(path, rank, seed) for seed, (path, rank) in enumerate(zip(paths,(16,16,32)))]
    originals = {p: p.read_bytes() for source in paths for p in source.iterdir()}
    output = tmp_path/'combined'
    assert main(['--task',str(paths[2]),'--box',str(paths[0]),'--box',str(paths[1]),'--output',str(output)]) == 0
    combined = safetensors.load_file(str(output/'adapter_model.safetensors'))
    a, b = (combined[PREFIX+suffix] for suffix in ('.lora_A.weight','.lora_B.weight'))
    expected = sum(2 * left[PREFIX+'.lora_B.weight'] @ right[PREFIX+'.lora_A.weight']
                   for left in states[:2] for right in states[:2])
    expected += 2 * states[2][PREFIX+'.lora_B.weight'] @ states[2][PREFIX+'.lora_A.weight']
    torch.testing.assert_close(b @ a, expected, atol=2e-5, rtol=2e-5)
    assert a.shape == (48,3) and b.shape == (4,48) and a.device.type == 'cpu'
    config = json.loads((output/'adapter_config.json').read_text())
    assert config['r'] == config['lora_alpha'] == 48
    receipt = json.loads((output/'export-info.json').read_text())
    assert len(receipt['sources']) == 3 and receipt['device'] == 'cpu'
    assert all(path.read_bytes() == data for path,data in originals.items())
    with pytest.raises(ValueError, match='fresh'):
        export_adapter(paths[2],paths[:2],output,MODEL)
    assert all(path.read_bytes() == data for path,data in originals.items())


@pytest.mark.parametrize('fault',['keys','shape','nonfinite','rank','duplicate','inside_source'])
def test_export_rejects_invalid_inputs_before_output(tmp_path, fault):
    boxes = [tmp_path/'ro',tmp_path/'en']
    task = tmp_path/'task'
    states = [adapter(path,rank,seed) for seed,(path,rank) in enumerate(zip([*boxes,task],(16,16,32)))]
    output = tmp_path/'combined'
    if fault == 'duplicate': boxes[1] = boxes[0]
    elif fault == 'inside_source': output = task/'derived'
    else:
        if fault == 'keys': states[2].pop(PREFIX+'.lora_B.weight')
        if fault == 'shape': states[2][PREFIX+'.lora_A.weight'] = torch.zeros(32,5)
        if fault == 'nonfinite': states[2][PREFIX+'.lora_A.weight'][0,0] = float('nan')
        if fault == 'rank': states[2][PREFIX+'.lora_A.weight'] = torch.zeros(31,3)
        safetensors.save_file(states[2],str(task/'adapter_model.safetensors'))
    with pytest.raises(ValueError): export_adapter(task,boxes,output,MODEL)
    assert not output.exists()
