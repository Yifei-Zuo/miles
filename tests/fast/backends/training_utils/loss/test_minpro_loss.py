"""MinPRO kernel tests (arXiv:2601.22718, xorl minpro_loss hard-clip variant).

Verifies against the reference semantics of xorl's minpro_loss.py:
prefix minimum is EXCLUSIVE, per-sample, mask-transparent, identity-1 on empty prefix;
gradient flows only through the current token's ratio (rho_bar detached) and clipped
tokens are gated out entirely.
"""

import math

import pytest
import torch

from miles.backends.training_utils.loss_hub.math_utils import (
    compute_minpro_loss,
    compute_minpro_prefix_min,
    compute_policy_loss,
)


def _prefix_reference(ratios, mask):
    """Straight-line reimplementation of the xorl reference: rho_bar_t = min over action
    tokens strictly before t within the sample; 1 when that set is empty."""
    out = []
    for t in range(len(ratios)):
        prior = [ratios[i] for i in range(t) if mask[i]]
        out.append(min(prior) if prior else 1.0)
    return out


def test_prefix_min_matches_reference_with_masks():
    torch.manual_seed(0)
    for _ in range(20):
        n = int(torch.randint(1, 30, (1,)))
        lp = torch.randn(n) * 0.3
        olp = torch.randn(n) * 0.3
        mask = (torch.rand(n) > 0.3).int()
        got = compute_minpro_prefix_min([lp], [olp], [mask])
        ratios = ((lp - olp).clamp(-20, 20)).exp().tolist()
        want = _prefix_reference(ratios, mask.tolist())
        assert torch.allclose(got, torch.tensor(want), atol=1e-6), (got, want)


def test_prefix_min_restarts_per_sample():
    lp1, olp1 = torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])  # ratios e^-1
    lp2, olp2 = torch.tensor([0.0]), torch.tensor([0.0])  # ratio 1
    m = torch.ones(2).int()
    got = compute_minpro_prefix_min([lp1, lp2], [olp1, olp2], [m, torch.ones(1).int()])
    # sample 1: [1 (empty prefix), e^-1]; sample 2 restarts: [1]
    assert torch.allclose(got, torch.tensor([1.0, math.exp(-1.0), 1.0]), atol=1e-6)


def test_prefix_min_is_exclusive_and_mask_transparent():
    # token 0 masked (env), token 1 tiny ratio, token 2 normal
    lp = torch.tensor([-5.0, -2.0, 0.0])
    olp = torch.tensor([0.0, 0.0, 0.0])
    mask = torch.tensor([0, 1, 1]).int()
    got = compute_minpro_prefix_min([lp], [olp], [mask])
    # t0: empty -> 1; t1: prefix has only masked t0 -> 1 (exclusive + transparent);
    # t2: min over {ratio_1 = e^-2}
    assert torch.allclose(got, torch.tensor([1.0, 1.0, math.exp(-2.0)]), atol=1e-6)


def test_minpro_loss_gradient_placement():
    # gradient must be rho_bar * rho * A on unclipped tokens, 0 on clipped tokens,
    # and rho_bar must carry NO gradient (detached)
    old = torch.tensor([0.0, 0.0, 0.0])
    logp = torch.tensor([-1.5, 0.0, 0.05], requires_grad=True)  # ratios: e^-1.5, 1.0, e^0.05
    mask = torch.ones(3).int()
    prefix = compute_minpro_prefix_min([logp], [old], [mask])
    assert torch.allclose(prefix, torch.tensor([1.0, math.exp(-1.5), math.exp(-1.5)]), atol=1e-6)
    adv = torch.tensor([1.0, 1.0, -1.0])
    ppo_kl = old - logp
    pg, clipfrac = compute_minpro_loss(ppo_kl, prefix, adv, eps_clip=0.2, eps_clip_high=0.28)
    # verify via autograd instead of hand-reasoning each clip branch:
    pg.sum().backward()
    combined = (prefix * (logp.detach() - old).exp()).tolist()
    for t in range(3):
        clipped = clipfrac[t].item() > 0
        expected = 0.0 if clipped else -adv[t].item() * combined[t]
        assert abs(logp.grad[t].item() - expected) < 1e-5, (t, clipped, logp.grad[t].item(), expected)


def test_minpro_loss_clip_band_semantics():
    # combined ratio above 1+eps_high with adv>0 must clip (gate out)
    prefix = torch.tensor([1.0])
    ppo_kl = torch.tensor([-0.5])  # ratio = e^0.5 ~ 1.65 > 1.28
    adv = torch.tensor([1.0])
    pg, clipfrac = compute_minpro_loss(ppo_kl, prefix, adv, 0.2, 0.28)
    assert clipfrac.item() == 1.0
    assert torch.allclose(pg, torch.tensor([-1.28]), atol=1e-6)
    # same ratio with adv<0: unclipped branch wins (pessimism rule), no gating
    pg2, clipfrac2 = compute_minpro_loss(ppo_kl, prefix, torch.tensor([-1.0]), 0.2, 0.28)
    assert clipfrac2.item() == 0.0
    assert torch.allclose(pg2, torch.tensor([math.exp(0.5)]), atol=1e-5)


def test_minpro_reduces_to_grpo_when_prefix_is_identity():
    torch.manual_seed(1)
    ppo_kl = torch.randn(50) * 0.01
    adv = torch.randn(50)
    ones = torch.ones(50)
    pg_m, cf_m = compute_minpro_loss(ppo_kl, ones, adv, 0.2, 0.28)
    pg_g, cf_g = compute_policy_loss(ppo_kl, adv, 0.2, 0.28)
    assert torch.allclose(pg_m, pg_g, atol=1e-6)
    assert torch.allclose(cf_m, cf_g, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
