import pytest
torch = pytest.importorskip("torch")
from agentprobe.core.causal_engine import CausalAttributionEngine

def test_causal_engine_identical():
    engine = CausalAttributionEngine()
    residual_stream = {
        0: torch.ones(1, 10, 5),
        1: torch.ones(1, 10, 5)
    }
    score = engine.compute_grounding_score(
        residual_stream,
        attention_map=None,
        obs_span=(0, 5),
        action_span=(5, 10)
    )
    assert score > 0.9

def test_causal_engine_orthogonal():
    engine = CausalAttributionEngine()
    
    res_stream = torch.zeros(1, 10, 2)
    res_stream[0, 0:5, 0] = 1.0  # obs
    res_stream[0, 5:10, 1] = 1.0  # act
    
    score = engine.compute_grounding_score(
        {0: res_stream},
        attention_map=None,
        obs_span=(0, 5),
        action_span=(5, 10)
    )
    assert score <= 0.6
