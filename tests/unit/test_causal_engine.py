import torch
from agentprobe.core.causal_engine import CausalAttributionEngine

def test_causal_engine_identical_spans():
    engine = CausalAttributionEngine()
    # Simulate identical spans (should align to 1.0)
    # Shape: [batch, seq_len, d_model]
    stream = {0: torch.ones(1, 10, 16)}
    score = engine.compute_grounding_score(stream, None, (0, 5), (0, 5))
    assert score == 1.0

def test_causal_engine_orthogonal():
    engine = CausalAttributionEngine()
    # Simulate orthogonal spans (score -> 0.5 because we normalize [-1,1] to [0,1], 0 maps to 0.5)
    tensor = torch.zeros(1, 10, 16)
    tensor[:, 0:5, :] = 1.0
    tensor[:, 5:10, :] = -1.0
    stream = {0: tensor}
    score = engine.compute_grounding_score(stream, None, (0, 5), (5, 10))
    assert abs(score - 0.0) < 1e-5
