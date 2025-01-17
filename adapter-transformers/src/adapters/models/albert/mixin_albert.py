from typing import Callable, Iterable, Tuple

import torch.nn as nn

from ...composition import adjust_tensors_for_parallel_
from ...layer import AdapterLayer
from ...lora import Linear as LoRALinear
from ...model_mixin import EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin
from ...prefix_tuning import PrefixTuningShim


class AlbertAttentionAdaptersMixin:
    """Adds adapters to the AlbertAttention module of ALBERT."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.query = LoRALinear.wrap(self.query, "selfattn", model_config, adapters_config, attn_key="q")
        self.key = LoRALinear.wrap(self.key, "selfattn", model_config, adapters_config, attn_key="k")
        self.value = LoRALinear.wrap(self.value, "selfattn", model_config, adapters_config, attn_key="v")

        self.attention_adapters = AdapterLayer("mh_adapter")

        self.prefix_tuning = PrefixTuningShim(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )


class AlbertEncoderLayerAdaptersMixin:
    """Adds adapters to the AlbertLayer module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.ffn = LoRALinear.wrap(self.ffn, "intermediate", model_config, adapters_config)
        self.ffn_output = LoRALinear.wrap(self.ffn_output, "output", model_config, adapters_config)

        # Set location keys for prefix tuning
        self.location_key = "output_adapter"

        self.output_adapters = AdapterLayer("output_adapter")

        self.attention.location_key = "self"


class AlbertModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the AlbertModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        # Set hook for parallel composition
        for _, layer in self.iter_layers():
            self._set_layer_hook_for_parallel(layer)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        i = 0
        for albertLayerGroup in self.encoder.albert_layer_groups:
            for albertLayer in albertLayerGroup.albert_layers:
                yield i, albertLayer
                i += 1

    def hook_after_embeddings(self, hook_fn: Callable):
        return self.embeddings.register_forward_hook(hook_fn)
