"""The bridge model provider builds a critic (value head, untied) and the HF load skips that head."""

import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from miles.backends.megatron_utils import model_provider as provider_module
from miles.backends.megatron_utils.checkpoint import _hide_value_head
from miles.backends.megatron_utils.model_provider import LinearForLastLayer

HIDDEN = 64


class _FakeProvider:
    hidden_size = HIDDEN
    sequence_parallel = False
    share_embeddings_and_output_weights = True

    def __init__(self):
        self.finalized = False

    def finalize(self):
        self.finalized = True

    def provide(self, pre_process=True, post_process=True, vp_stage=None):
        model = torch.nn.Module()
        model.config = self
        model.output_layer = torch.nn.Linear(HIDDEN, 1000)
        model.forward = lambda *a, **k: None
        return model


def _install_fake_bridge(monkeypatch, provider):
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = provider
    module = types.ModuleType("megatron.bridge")
    module.AutoBridge = SimpleNamespace(from_hf_pretrained=lambda *a, **k: bridge)
    monkeypatch.setitem(sys.modules, "megatron.bridge", module)
    monkeypatch.setattr(provider_module, "_apply_bridge_runtime_config", lambda provider, args: None)


def _bridge_args():
    return Namespace(megatron_to_hf_mode="bridge", hf_checkpoint="/nonexistent", enable_witness=False)


def test_bridge_critic_builds_untied_scalar_head(monkeypatch):
    provider = _FakeProvider()
    _install_fake_bridge(monkeypatch, provider)

    model = provider_module.get_model_provider_func(_bridge_args(), role="critic")(
        pre_process=False, post_process=True
    )

    assert provider.finalized
    assert provider.share_embeddings_and_output_weights is False
    assert isinstance(model.output_layer, LinearForLastLayer)
    assert tuple(model.output_layer.weight.shape) == (1, HIDDEN)


def test_bridge_actor_keeps_lm_head(monkeypatch):
    provider = _FakeProvider()
    _install_fake_bridge(monkeypatch, provider)

    model = provider_module.get_model_provider_func(_bridge_args(), role="actor")(pre_process=False, post_process=True)

    assert provider.share_embeddings_and_output_weights is True
    assert tuple(model.output_layer.weight.shape) == (1000, HIDDEN)


def test_hide_value_head_hides_only_value_heads_and_restores():
    critic = torch.nn.Module()
    critic.output_layer = LinearForLastLayer(HIDDEN, 1, config=SimpleNamespace(sequence_parallel=False))
    head = critic.output_layer
    actor = torch.nn.Module()
    actor.output_layer = torch.nn.Linear(HIDDEN, 1000)
    no_head = torch.nn.Module()

    with _hide_value_head([critic, actor, no_head]):
        assert not any(n.startswith("output_layer") for n, _ in critic.named_parameters())
        assert any(n.startswith("output_layer") for n, _ in actor.named_parameters())

    assert critic.output_layer is head


def test_hide_value_head_restores_on_error():
    critic = torch.nn.Module()
    critic.output_layer = LinearForLastLayer(HIDDEN, 1, config=SimpleNamespace(sequence_parallel=False))
    head = critic.output_layer

    try:
        with _hide_value_head([critic]):
            raise RuntimeError("load failed")
    except RuntimeError:
        pass

    assert critic.output_layer is head
