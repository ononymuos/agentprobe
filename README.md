# AgentProbe

Real-Time Mechanistic Interpretability & Neuro-Symbolic Delusion Interceptor for Compound AI Systems.

## Overview
AgentProbe intercepts ungrounded hallucinations (Latent Delusions) in compound AI reasoning loops. It cross-references neural grounding (via residual stream cosine similarity) against symbolic logic formal bounds (using Microsoft Z3).

## Installation

### Standard (Development)
```bash
pip install -e .
```

### With Machine Learning Backends (Requires GPU/RAM)
```bash
pip install -e .[ml]
```

## Usage

Wrap your AI loop:

```python
from agentprobe.interceptor.guard import AgentGuard
from agentprobe.core.hooks import ModelHookManager

# Initialize Hook Manager
hook_mgr = ModelHookManager(your_pytorch_model)

# Initialize Guard
guard = AgentGuard(model_hook_manager=hook_mgr, grounding_threshold=0.42)

# ... inside reasoning loop ...
guard.evaluate_step(
    obs_span=(0, 50), 
    action_span=(51, 60), 
    proposed_action_name="write", 
    proposed_action_params={"path": "/workspace/out.txt"}
)
```
