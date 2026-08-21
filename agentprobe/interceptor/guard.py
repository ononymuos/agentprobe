"""
agentprobe.interceptor.guard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The central execution interceptor that combines causal neural grounding
with SMT formal verification in agent execution loops.
"""

from typing import Any, Callable, Dict, Optional, Tuple
from agentprobe.core.causal_engine import CausalAttributionEngine
from agentprobe.core.hooks import ModelHookManager
from agentprobe.verifier.smt_compiler import SMTInvariantVerifier


class DelusionInterceptException(Exception):
    def __init__(self, message: str, remediation_prompt: str):
        super().__init__(message)
        self.remediation_prompt = remediation_prompt


class AgentGuard:
    def __init__(
        self,
        model_hook_manager: ModelHookManager,
        grounding_threshold: float = 0.42,
        sandbox_root: str = "/workspace"
    ):
        self.hook_mgr = model_hook_manager
        self.causal_engine = CausalAttributionEngine()
        self.verifier = SMTInvariantVerifier()
        self.threshold = grounding_threshold
        self.sandbox_root = sandbox_root

    def evaluate_step(
        self,
        obs_span: Tuple[int, int],
        action_span: Tuple[int, int],
        proposed_action_name: str,
        proposed_action_params: Dict[str, Any],
        is_read_only: bool = False
    ) -> None:
        """
        Synchronously evaluated before executing any tool or OS action.
        Raises DelusionInterceptException if safety or grounding fails.
        """
        # 1. Neural Mechanistic Grounding Verification
        grounding_score = self.causal_engine.compute_grounding_score(
            residual_stream=self.hook_mgr.buffer.residual_stream,
            attention_map=self.hook_mgr.buffer.attention_maps.get(self.hook_mgr.target_layers[-1]),
            obs_span=obs_span,
            action_span=action_span
        )

        if grounding_score < self.threshold:
            remediation = (
                f"[AGENTPROBE INTERCEPT: NEURAL DELUSION DETECTED]\n"
                f"Your proposed action '{proposed_action_name}' exhibited a Grounding Attribution Score "
                f"of {grounding_score:.3f} (Threshold: {self.threshold}). "
                f"Your action is not grounded in the recent environment observation. "
                f"Re-evaluate the output of the previous step and resolve the discrepancy."
            )
            raise DelusionInterceptException(
                message=f"Neural Grounding Score {grounding_score:.3f} below threshold {self.threshold}",
                remediation_prompt=remediation
            )

        # 2. Symbolic Invariant SMT Verification
        if "path" in proposed_action_params:
            valid, err_msg = self.verifier.verify_filesystem_safety(
                command_type=proposed_action_name,
                target_path=str(proposed_action_params["path"]),
                sandbox_root=self.sandbox_root,
                is_read_only=is_read_only
            )
            if not valid:
                remediation = (
                    f"[AGENTPROBE INTERCEPT: FORMAL SMT INVARIANT VIOLATION]\n"
                    f"Action '{proposed_action_name}' was blocked by symbolic verification.\n"
                    f"Reason: {err_msg}\n"
                    f"Please adjust parameters to strictly obey sandbox boundaries."
                )
                raise DelusionInterceptException(
                    message=err_msg or "Invariant Violation",
                    remediation_prompt=remediation
                )
