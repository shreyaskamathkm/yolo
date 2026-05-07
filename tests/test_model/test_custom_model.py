import torch
from torch import nn
from yolo.registry import MODELS
from yolo.model.builder import create_model
from omegaconf import OmegaConf

@MODELS.register_module()
class CustomTestModel(nn.Module):
    def __init__(self, model_cfg, class_num=80):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.class_num = class_num

    def forward(self, x):
        return self.conv(x)

def test_custom_model_registration():
    model_cfg = OmegaConf.create({
        "name": "custom",
        "type": "CustomTestModel",
        "anchor": {"strides": [8, 16, 32], "reg_max": 16},
        "model": {},
        "compile": {"enabled": False}
    })
    
    model = create_model(model_cfg, weight_path=None, class_num=10)
    assert isinstance(model, CustomTestModel)
    assert model.class_num == 10
    
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    assert output.shape == (1, 16, 62, 62)
