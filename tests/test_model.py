import pytest
import torch
from src import config, model


def test_forward_returns_one_value_per_target():
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7)
    out = net(torch.randn(8, 20), torch.randn(8, 7))
    assert out.shape == (8, len(config.TARGETS))


def test_outputs_are_bounded_to_unit_interval():
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7)
    out = net(torch.randn(64, 20) * 50, torch.randn(64, 7) * 50)
    assert (out >= 0).all() and (out <= 1).all()


def test_club_input_actually_changes_the_prediction():
    torch.manual_seed(0)
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7).eval()
    player = torch.randn(4, 20)
    a = net(player, torch.zeros(4, 7))
    b = net(player, torch.ones(4, 7))
    assert not torch.allclose(a, b), "conditioning on the club has no effect"


def test_gradients_reach_both_towers():
    """Both towers must receive gradient of non-zero MAGNITUDE, not merely a
    non-None grad tensor. The towers end in ReLU, so a tower whose units are all
    saturated off still populates `.grad` — with zeros. That is the shape the
    dead-player-tower bug took in the shipped ensemble (finding #1), and a
    `is not None` assertion sails straight past it."""
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7)
    out = net(torch.randn(4, 20), torch.randn(4, 7))
    out.sum().backward()
    assert net.player_tower[0].weight.grad is not None
    assert net.club_tower[0].weight.grad is not None
    assert net.player_tower[0].weight.grad.abs().sum() > 0
    assert net.club_tower[0].weight.grad.abs().sum() > 0


def test_player_input_actually_changes_the_prediction():
    """The mirror of the club-conditioning test. With the club held fixed,
    different players must score differently — otherwise the model is a lookup
    table on the destination club."""
    torch.manual_seed(0)
    net = model.TwoTowerNet(n_player_features=20, n_club_features=7).eval()
    club = torch.ones(4, 7)
    a = net(torch.randn(4, 20), club)
    b = net(torch.randn(4, 20) * 2 - 1, club)
    assert not torch.allclose(a, b), "the player features have no effect"


def test_combined_loss_is_zero_for_perfect_prediction():
    y = torch.tensor([[0.5, 0.5, 1.0]])
    assert model.combined_loss(y, y).item() == pytest.approx(0.0, abs=1e-6)


def test_combined_loss_weights_all_three_heads_equally():
    y = torch.tensor([[0.0, 0.0, 0.0]])
    only_first = model.combined_loss(torch.tensor([[0.5, 0.0, 0.0]]), y)
    only_second = model.combined_loss(torch.tensor([[0.0, 0.5, 0.0]]), y)
    assert only_first.item() == pytest.approx(only_second.item())
