import torch
import torch.nn as nn

from agentprobe.core.hooks import ModelHookManager


class DummyLayer(nn.Module):
    def forward(self, x):
        return x

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer() for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def test_hooks_attachment():
    model = DummyModel()
    manager = ModelHookManager(model, target_layers=[0, 1, 2])

    x = torch.randn(1, 10, 16)
    model(x)

    assert len(manager.buffer.residual_stream) == 3
    for i in range(3):
        assert torch.allclose(manager.buffer.residual_stream[i], x)

    manager.remove()
    model(x)
    assert len(manager.buffer.residual_stream) == 0
