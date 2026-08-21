# AgentProbe

Real-Time Mechanistic Interpretability & Neuro-Symbolic Delusion Interceptor for Compound AI Systems.

## Overview
AgentProbe provides real-time causal attribution metrics across the transformer residual stream and combines them with symbolic Satisfiability Modulo Theories (SMT) verification.

## The Real-World Problem 🌍
When Compound AI systems (like autonomous coding agents) operate in complex environments, they frequently suffer from hallucinations or logical "delusions"—where the model makes confident but fundamentally flawed assumptions. Traditional guardrails only check the *final output text*, catching errors too late, often after a bad API call or system action has already been triggered.

AgentProbe solves this by looking *inside* the model's computation. By hooking directly into the PyTorch forward pass (extracting layer-wise residual stream activations and attention maps) and applying Satisfiability Modulo Theories (SMT), it mathematically verifies the agent's causal reasoning in real-time. If it detects a delusion forming, it intercepts the execution before the action is taken, throwing a `DelusionInterceptException` with a specific remediation prompt to auto-correct the agent's trajectory.

## How to Use It 💡

### 1. Installation
Since AgentProbe uses Poetry, you can install it directly in your environment:
```bash
cd agentprobe
pip install .
```

### 2. Basic Usage
Integrate the interceptor directly into your agent's generation loop to catch delusions before they become actions:

```python
from agentprobe.core.hooks import ModelHookManager
from agentprobe.interceptor.guard import DelusionInterceptException

# 1. Attach low-overhead forward pre-hooks to your PyTorch LLM
hook_manager = ModelHookManager(model)
hook_manager.register_hooks()

try:
    # 2. Run your agent's forward pass / thought generation.
    # AgentProbe continuously monitors the residual stream for causal disconnects.
    outputs = model.generate(**inputs)

except DelusionInterceptException as e:
    # 3. Intercepted a hallucination/delusion before the action was taken!
    print(f"Delusion detected: {e}")
    
    # 4. Feed the built-in remediation prompt back to the agent to self-correct
    agent.apply_correction(e.remediation_prompt)
```
