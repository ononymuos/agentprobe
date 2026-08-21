"""
agentprobe.core.hooks
~~~~~~~~~~~~~~~~~~~~~
Low-overhead PyTorch forward pre-hooks for extracting layer-wise
residual stream activations and attention maps without breaking torch.compile.
"""

import typing
import torch
import torch.nn as nn

class ActivationBuffer:
    def __init__(self) -> None:
        self.residual_stream: dict[int, torch.Tensor] = {}
        self.attention_maps: dict[int, torch.Tensor] = {}

    def clear(self) -> None:
        self.residual_stream.clear()
        self.attention_maps.clear()

class ModelHookManager:
    def __init__(self, model: nn.Module, target_layers: list[int] | None = None):
        self.model = model
        self.buffer = ActivationBuffer()
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        # Auto-detect transformer layer structure (Llama, Mistral, Qwen, Gemma)
        self.layers = self._resolve_layers()
        self.target_layers = target_layers or [
            0,
            len(self.layers) // 2,
            len(self.layers) - 1
        ]
        self._register_hooks()

    def _resolve_layers(self) -> nn.ModuleList:
        for attr in ["layers", "model.layers", "transformer.h", "model.decoder.layers"]:
            try:
                curr = self.model
                for part in attr.split("."):
                    curr = getattr(curr, part)
                return typing.cast(nn.ModuleList, curr)
            except AttributeError:
                continue
        raise ValueError("Unsupported model architecture. Could not resolve transformer layers.")

    def _register_hooks(self) -> None:
        for idx in self.target_layers:
            layer = self.layers[idx]

            def make_hook(layer_idx: int) -> typing.Callable[..., None]:
                def hook_fn(module: nn.Module, input: typing.Any, output: typing.Any) -> None:
                    # Handle tuple outputs (hidden_states, attns)
                    if isinstance(output, tuple):
                        hidden = output[0]
                        if len(output) > 1 and isinstance(output[1], torch.Tensor):
                            self.buffer.attention_maps[layer_idx] = output[1].detach()
                    else:
                        hidden = output
                    self.buffer.residual_stream[layer_idx] = hidden.detach()
                return hook_fn

            self.handles.append(layer.register_forward_hook(make_hook(idx)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.buffer.clear()
