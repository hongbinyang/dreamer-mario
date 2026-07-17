import numpy as np
import pytest

from dreamer.replay import ReplayBuffer


def make_buffer(capacity=13, seq_len=5, batch_size=4, seed=0):
    return ReplayBuffer(capacity=capacity, obs_shape=(2, 2, 1), num_actions=3,
                         seq_len=seq_len, batch_size=batch_size, seed=seed)


def fill(buf, n, start=0):
    for i in range(n):
        g = start + i
        obs = np.zeros((2, 2, 1), dtype=np.uint8)
        buf.add(obs, action=g % 3, reward=float(g), is_first=(g == 0), is_terminal=False)


def test_len_and_full_flag_across_wraparound():
    buf = make_buffer(capacity=10)
    fill(buf, 6)
    assert len(buf) == 6 and not buf.full
    fill(buf, 10, start=6)  # 16 total adds into capacity 10 -> wraps
    assert len(buf) == 10 and buf.full


def test_valid_start_count_matches_capacity_minus_seq_len():
    buf = make_buffer(capacity=10, seq_len=4)
    fill(buf, 25)  # wraps multiple times
    assert buf.full
    valid_count = sum(buf._valid_start(i) for i in range(buf.capacity))
    assert valid_count == buf.capacity - buf.seq_len


def test_sampled_sequences_are_temporally_contiguous_after_wraparound():
    # The real payoff test: smoke_test.py never fills the buffer past
    # capacity, so this wraparound/seam-avoidance path has never actually
    # been exercised before. reward == the global write index makes
    # contiguity directly checkable.
    buf = make_buffer(capacity=13, seq_len=5, batch_size=16)
    fill(buf, 97)  # several wraps around capacity=13
    for _ in range(50):
        batch = buf.sample()
        diffs = np.diff(batch["reward"], axis=1)
        assert np.all(diffs == 1.0), "sampled sequence is not temporally contiguous"


def test_sample_shapes_and_one_hot_actions():
    buf = make_buffer(capacity=50, seq_len=6, batch_size=5)
    fill(buf, 50)
    batch = buf.sample()
    assert batch["obs"].shape == (5, 6, 2, 2, 1)
    assert batch["action"].shape == (5, 6, 3)
    assert np.allclose(batch["action"].sum(-1), 1.0)  # valid one-hot rows


def test_first_step_of_each_sampled_sequence_marked_is_first():
    buf = make_buffer(capacity=50, seq_len=6, batch_size=5)
    fill(buf, 50)
    batch = buf.sample()
    assert np.all(batch["is_first"][:, 0] == 1.0)


def test_sample_raises_if_not_enough_data():
    buf = make_buffer(capacity=10, seq_len=8)
    fill(buf, 5)
    with pytest.raises(AssertionError):
        buf.sample()
