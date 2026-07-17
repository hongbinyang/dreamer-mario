import torch

from dreamer.rssm import RSSM


def make_rssm(action_dim=3, embed_dim=8, deter=16, stoch=4, classes=4, hidden=16, unimix=0.01):
    return RSSM(action_dim=action_dim, embed_dim=embed_dim, deter=deter, stoch=stoch,
                classes=classes, hidden=hidden, unimix=unimix)


def test_unimix_floors_minimum_probability():
    rssm = make_rssm(classes=5, unimix=0.1)
    logits = torch.zeros(2, rssm.stoch, rssm.classes)
    logits[..., 0] = 50.0  # near-certain on class 0 before unimix mixing
    probs = rssm._dist(logits).base_dist.probs
    floor = 0.1 / 5
    assert probs.min().item() >= floor - 1e-6
    assert probs.min().item() < floor + 1e-3  # floor should be tight, not slack


def test_kl_loss_free_bits_clamps_zero_kl():
    rssm = make_rssm()
    logits = torch.randn(2, 3, rssm.stoch, rssm.classes)
    post = {"logits": logits}
    prior = {"logits": logits.clone()}  # identical -> true KL == 0
    loss, metrics = rssm.kl_loss(post, prior, free_bits=1.0, dyn_scale=0.5, rep_scale=0.1)
    assert metrics["kl_dyn"] < 1e-5
    assert metrics["kl_rep"] < 1e-5
    # The *clamped* loss sits at the free-bits floor, not at 0.
    assert torch.isclose(loss, torch.tensor(0.5 + 0.1), atol=1e-4)


def _grad_sum(module):
    g = module[-1].weight.grad
    return None if g is None else g.abs().sum().item()


def test_kl_loss_balancing_isolates_gradients_within_a_single_step():
    """dyn loss (sg on posterior) should only update prior_net; rep loss (sg
    on prior) should only update post_net, for the same-timestep KL term
    itself. This is the entire point of KL balancing -- if the stop-
    gradients were ever on the wrong side, training would still run and look
    plausible, just be wrong.

    Uses T=1 deliberately: at T>1 the recurrence carries the posterior's
    straight-through sample into the *next* step's prior, so post_net does
    legitimately receive some dyn-loss gradient through later timesteps --
    see test_dyn_loss_leaks_into_post_net_across_timesteps below. T=1 is the
    only way to isolate the direct, same-step stop-gradient contract."""
    rssm = make_rssm(action_dim=3, embed_dim=8, deter=16, stoch=4, classes=4, hidden=16)
    B, T = 2, 1
    embed = torch.randn(B, T, 8)
    action = torch.zeros(B, T, 3)
    action[..., 0] = 1.0
    is_first = torch.ones(B, T)  # forces the zero initial state, no recurrence to entangle

    post, prior = rssm.observe(embed, action, is_first)
    loss_dyn, _ = rssm.kl_loss(post, prior, free_bits=0.0, dyn_scale=1.0, rep_scale=0.0)
    rssm.zero_grad()
    loss_dyn.backward()
    assert _grad_sum(rssm.prior_net) is not None and _grad_sum(rssm.prior_net) > 0
    assert _grad_sum(rssm.post_net) in (None, 0.0)

    post, prior = rssm.observe(embed, action, is_first)
    loss_rep, _ = rssm.kl_loss(post, prior, free_bits=0.0, dyn_scale=0.0, rep_scale=1.0)
    rssm.zero_grad()
    loss_rep.backward()
    assert _grad_sum(rssm.post_net) is not None and _grad_sum(rssm.post_net) > 0
    assert _grad_sum(rssm.prior_net) in (None, 0.0)


def test_dyn_loss_leaks_into_post_net_across_timesteps():
    """Documents an intentional asymmetry: prior_net is a pure leaf (only
    ever consumed as prior['logits']), so it never receives rep-loss
    gradient at any T. post_net's straight-through sample instead feeds the
    *next* step's img_step, so multi-step dyn loss does reach it -- this is
    the RSSM's recurrence working as designed, not a stop-gradient bug."""
    rssm = make_rssm(action_dim=3, embed_dim=8, deter=16, stoch=4, classes=4, hidden=16)
    B, T = 2, 4
    embed = torch.randn(B, T, 8)
    action = torch.zeros(B, T, 3)
    action[..., 0] = 1.0
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0

    post, prior = rssm.observe(embed, action, is_first)
    loss_dyn, _ = rssm.kl_loss(post, prior, free_bits=0.0, dyn_scale=1.0, rep_scale=0.0)
    rssm.zero_grad()
    loss_dyn.backward()
    assert _grad_sum(rssm.prior_net) > 0
    assert _grad_sum(rssm.post_net) > 0  # the documented leak, not isolation

    post, prior = rssm.observe(embed, action, is_first)
    loss_rep, _ = rssm.kl_loss(post, prior, free_bits=0.0, dyn_scale=0.0, rep_scale=1.0)
    rssm.zero_grad()
    loss_rep.backward()
    assert _grad_sum(rssm.post_net) > 0
    assert _grad_sum(rssm.prior_net) in (None, 0.0)  # prior_net stays isolated even at T>1


def test_imagine_shapes():
    rssm = make_rssm(action_dim=3, embed_dim=8, deter=16, stoch=4, classes=4, hidden=16)
    start = rssm.initial(5, torch.device("cpu"))

    class ConstantPolicy:
        def __call__(self, feat):
            probs = torch.zeros(feat.shape[0], 3)
            probs[:, 0] = 1.0
            return torch.distributions.OneHotCategorical(probs=probs)

    seq, actions = rssm.imagine(start, ConstantPolicy(), horizon=6)
    assert seq["deter"].shape == (7, 5, 16)
    assert seq["stoch"].shape == (7, 5, 4 * 4)
    assert actions.shape == (6, 5, 3)
