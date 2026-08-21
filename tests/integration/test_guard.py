import pytest
import torch
import torch.nn as nn

from agentprobe.core.hooks import ModelHookManager
from agentprobe.interceptor.guard import AgentGuard, DelusionInterceptException


class MockLayer(nn.Module):
    def forward(self, x):
        return x

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer() for _ in range(3)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def test_integration_delusion_intercept():
    model = MockModel()
    manager = ModelHookManager(model, target_layers=[0, 1, 2])
    guard = AgentGuard(manager, grounding_threshold=0.42, sandbox_root="/workspace")

    # Simulate a forward pass to populate the hooks
    # We create orthogonal vectors to simulate a hallucination (score should be ~0.0)
    dummy_input = torch.zeros(1, 10, 16)
    dummy_input[:, 0:5, :] = 1.0
    dummy_input[:, 5:10, :] = -1.0
    model(dummy_input)

    with pytest.raises(DelusionInterceptException) as excinfo:
        guard.evaluate_step(
            obs_span=(0, 5),
            action_span=(5, 10),
            proposed_action_name="write",
            proposed_action_params={"path": "/workspace/test.txt"},
            is_read_only=False
        )

    assert "Neural Grounding Score" in str(excinfo.value)
