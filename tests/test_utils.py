import torch

from dreamer.utils import Moments, TwoHotDistSymlog, lambda_return, symexp, symlog


def test_symlog_symexp_roundtrip():
    x = torch.tensor([-100.0, -5.0, -1.0, 0.0, 1.0, 5.0, 100.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_symlog_compresses_large_magnitudes():
    # Should grow much slower than linear for large |x| -- that's the point.
    assert symlog(torch.tensor(1000.0)) < 10 * symlog(torch.tensor(10.0))


def test_twohot_log_prob_peaks_at_the_matching_bin():
    # Uniform logits + a target landing exactly on bin 5 (value 0.0) should
    # give the same log_prob as any single-bin target under a uniform
    # categorical: log(1/num_bins).
    num_bins = 11
    dist = TwoHotDistSymlog(torch.zeros(1, num_bins), low=-5.0, high=5.0)
    logp = dist.log_prob(torch.tensor([0.0]))
    assert torch.allclose(logp, torch.log(torch.tensor(1.0 / num_bins)), atol=1e-5)


def test_twohot_mean_recovers_target_after_fitting():
    # Fit logits by gradient descent against log_prob (the actual training
    # signal) and check the decoded mean converges to the target -- this
    # exercises encode (log_prob) and decode (mean) together.
    torch.manual_seed(0)
    target = torch.tensor([3.7, -12.0, 0.0])
    logits = torch.zeros(3, 255, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.1)
    for _ in range(300):
        loss = -TwoHotDistSymlog(logits).log_prob(target).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.allclose(TwoHotDistSymlog(logits).mean, target, atol=0.5)


def test_lambda_return_matches_hand_computation():
    reward = torch.tensor([[1.0], [2.0]])
    value = torch.tensor([[3.0], [4.0]])  # v_1, v_2 (value at the *next* state)
    discount = torch.tensor([[0.9], [0.9]])
    returns = lambda_return(reward, value, discount, lam=0.5)
    r1 = 2.0 + 0.9 * 4.0  # bootstrapped by v_2
    r0 = 1.0 + 0.9 * (0.5 * 3.0 + 0.5 * r1)
    assert torch.allclose(returns[1, 0], torch.tensor(r1), atol=1e-5)
    assert torch.allclose(returns[0, 0], torch.tensor(r0), atol=1e-5)


def test_lambda_return_lam_zero_is_one_step_td():
    reward = torch.tensor([[1.0], [2.0], [3.0]])
    value = torch.tensor([[5.0], [6.0], [7.0]])
    discount = torch.tensor([[0.99], [0.99], [0.99]])
    returns = lambda_return(reward, value, discount, lam=0.0)
    assert torch.allclose(returns, reward + discount * value, atol=1e-5)


def test_moments_scale_floors_at_one():
    m = Moments(decay=0.99)
    assert m(torch.zeros(100)) == 1.0  # P95 - P5 == 0 -> floored to 1.0


def test_moments_tracks_percentile_range():
    m = Moments(decay=0.99)
    scale = m(torch.linspace(0, 100, 1000))
    assert 85.0 < scale < 95.0  # P95 - P5 of a uniform [0, 100] range is ~90
