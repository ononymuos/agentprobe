"""
agentprobe.core.causal_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-speed tensor engine computing causal grounding metrics,
subspace projection angles, and attention entropy over agent spans.
"""

import math

import torch


class CausalAttributionEngine:
    def __init__(self, epsilon: float = 1e-8):
        self.eps = epsilon

    @torch.inference_mode()
    def compute_grounding_score(
        self,
        residual_stream: dict[int, torch.Tensor],
        attention_map: torch.Tensor | None,
        obs_span: tuple[int, int],
        action_span: tuple[int, int]
    ) -> float:
        """
        Computes the Causal Grounding Score between observation tokens and action tokens.
        """
        obs_start, obs_end = obs_span
        act_start, act_end = action_span

        if obs_start >= obs_end or act_start >= act_end:
            return 0.0

        layer_scores = []
        for layer_idx, hidden_state in residual_stream.items():
            # hidden_state: [batch_size, seq_len, d_model]
            obs_vec = hidden_state[:, obs_start:obs_end, :].mean(dim=1)  # [B, D]
            act_vec = hidden_state[:, act_start:act_end, :].mean(dim=1)  # [B, D]

            # Cosine similarity in latent manifold
            cos_sim = torch.cosine_similarity(obs_vec, act_vec, dim=-1).item()
            # Normalize from [-1, 1] to [0, 1]
            norm_sim = max(0.0, (cos_sim + 1.0) / 2.0)
            layer_scores.append(norm_sim)

        base_alignment = sum(layer_scores) / max(1, len(layer_scores))

        # Entropy calculation if attention tensor is available
        entropy_penalty = 0.0
        if attention_map is not None:
            # attention_map: [batch, heads, seq_len, seq_len]
            # Cross-attention slice: action tokens querying observation tokens
            act_to_obs_attn = attention_map[:, :, act_start:act_end, obs_start:obs_end]
            mean_attn = act_to_obs_attn.mean(dim=(0, 1, 2))  # [obs_len]
            probs = mean_attn / (mean_attn.sum() + self.eps)

            entropy = -torch.sum(probs * torch.log(probs + self.eps)).item()
            max_entropy = math.log(max(1, obs_end - obs_start))
            normalized_entropy = entropy / max(self.eps, max_entropy) if max_entropy > 0 else 0.0
            entropy_penalty = normalized_entropy

        # Final Composite Grounding Metric
        final_score = base_alignment * (1.0 - 0.5 * entropy_penalty)
        return float(final_score)
