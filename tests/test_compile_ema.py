import pytest
import torch
import tempfile
from pathlib import Path
from yolo.utils.module_utils import unwrap_model
from yolo.tasks.detection.solver import DetectionTrainModel
from yolo.training.callbacks import EMA
from tests.conftest import get_cfg

@pytest.mark.parametrize("model_name", ["v9-t"])
def test_torch_compile_save_load(model_name):
    """Test that a compiled model's checkpoint can be saved cleanly and loaded into both compiled and uncompiled models."""
    # 1. Setup config with compilation enabled and task=train
    cfg = get_cfg(overrides=[f"model={model_name}", "model.compile.enabled=True", "task=train"])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # 2. Create and Compile Model
        model = DetectionTrainModel(cfg)
        
        # Verify it's actually compiled (OptimizedModule)
        if hasattr(torch, "compile"):
            assert hasattr(model.model, "_orig_mod"), "Model should be compiled but doesn't have _orig_mod"
        
        # 3. Save Checkpoint (Simulate Lightning behavior)
        checkpoint = {"state_dict": model.state_dict()}
        model.on_save_checkpoint(checkpoint)
        
        # Verify keys are cleaned of '_orig_mod'
        for k in checkpoint["state_dict"].keys():
            assert "_orig_mod" not in k, f"Key {k} still contains '_orig_mod' after cleaning"
            
        checkpoint_path = tmpdir / "test_compile.ckpt"
        torch.save(checkpoint, checkpoint_path)
        
        # 4. Load into another COMPILED model
        cfg_compile = get_cfg(overrides=[f"model={model_name}", "model.compile.enabled=True", "task=train"])
        model_compiled = DetectionTrainModel(cfg_compile)
        
        ckpt_to_load = torch.load(checkpoint_path)
        model_compiled.on_load_checkpoint(ckpt_to_load)
        
        # Should load strictly without error
        model_compiled.load_state_dict(ckpt_to_load["state_dict"], strict=True)
        print("Successfully loaded compiled checkpoint into compiled model (strict=True)")
        
        # 5. Load into a NON-COMPILED model
        cfg_no_compile = get_cfg(overrides=[f"model={model_name}", "model.compile.enabled=False", "task=train"])
        model_no_compile = DetectionTrainModel(cfg_no_compile)
        
        ckpt_no_compile = torch.load(checkpoint_path)
        model_no_compile.on_load_checkpoint(ckpt_no_compile)
        
        # Should also load strictly
        model_no_compile.load_state_dict(ckpt_no_compile["state_dict"], strict=True)
        print("Successfully loaded compiled checkpoint into non-compiled model (strict=True)")

@pytest.mark.parametrize("model_name", ["v9-t"])
def test_ema_compile_integration(model_name):
    """Test that EMA shadow weights are saved cleanly and can be loaded into compiled models."""
    cfg = get_cfg(overrides=[f"model={model_name}", "model.compile.enabled=True", "task=train"])
    model = DetectionTrainModel(cfg)
    
    ema = EMA(decay=0.99)
    # Manually initialize shadow to simulate training state
    unwrapped_model = unwrap_model(model.model)
    ema.shadow = {k: v.clone().cpu() for k, v in unwrapped_model.state_dict().items()}
    
    # 1. Save EMA shadow (Simulate Callback hook)
    checkpoint = {}
    ema.on_save_checkpoint(None, model, checkpoint)
    
    # Verify EMA shadow keys are clean
    assert "ema_shadow" in checkpoint
    for k in checkpoint["ema_shadow"].keys():
        assert "_orig_mod" not in k, f"EMA key {k} contains '_orig_mod'"
        
    # 2. Load EMA shadow into a DIFFERENT compiled model
    new_cfg = get_cfg(overrides=[f"model={model_name}", "model.compile.enabled=True", "task=train"])
    new_model = DetectionTrainModel(new_cfg)
    new_ema = EMA(decay=0.99)
    
    new_ema.on_load_checkpoint(None, new_model, checkpoint)
    
    # Verify loaded shadow can be applied to the model structure
    new_ema.apply_shadow(new_model)
    print("EMA shadow successfully restored and applied to compiled model")
