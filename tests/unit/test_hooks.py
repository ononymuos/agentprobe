import pytest
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
from agentprobe.core.hooks import ModelHookManager

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def test_hooks():
    model = DummyModel()
    manager = ModelHookManager(model)
    
    x = torch.randn(1, 10)
    out = model(x)
    
    assert len(manager.buffer.residual_stream) > 0
    manager.remove()
    assert len(manager.buffer.residual_stream) == 0
