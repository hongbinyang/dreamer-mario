import torch

from dreamer.networks import ActorHead, ConvDecoder, ConvEncoder, TwoHotHead


def test_conv_encoder_decoder_shapes():
    enc = ConvEncoder(in_channels=3, depth=8)
    dec = ConvDecoder(enc.out_dim, out_channels=3, depth=8)
    x = torch.randn(5, 3, 64, 64)
    embed = enc(x)
    assert embed.shape == (5, enc.out_dim)
    assert dec(embed).shape == (5, 3, 64, 64)


def test_twohot_head_zero_init_predicts_zero():
    head = TwoHotHead(in_dim=10, hidden=16, layers=1, num_bins=51, zero_init=True)
    dist = head(torch.randn(4, 10))
    assert torch.allclose(dist.mean, torch.zeros(4), atol=1e-4)


def test_actor_head_unimix_floor():
    head = ActorHead(in_dim=10, hidden=16, layers=1, num_actions=4, unimix=0.2)
    with torch.no_grad():
        head.net[-1].weight.zero_()
        head.net[-1].bias.copy_(torch.tensor([50.0, -50.0, -50.0, -50.0]))
    dist = head(torch.zeros(3, 10))
    assert dist.probs.min().item() >= (0.2 / 4) - 1e-4
