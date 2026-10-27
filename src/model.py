import torch
from torch import nn

from src import config

N_TARGETS = len(config.TARGETS)


def _tower(n_in: int, hidden: int, embed: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, embed),
        nn.ReLU(),
    )


class TwoTowerNet(nn.Module):
    """Player and club embedded separately, combined by concatenation plus an
    elementwise product. The product term is where player-by-club interactions
    live — it is the reason the model can answer 'fits THIS side' rather than
    only 'good player'.

    Three heads share one trunk, which is multi-task regularisation: the trunk
    must serve all three outcomes, which is harder to overfit than any one.
    """

    def __init__(self, n_player_features: int, n_club_features: int,
                 hidden: int = 64, embed: int = 32, dropout: float = 0.3):
        super().__init__()
        self.player_tower = _tower(n_player_features, hidden, embed, dropout)
        self.club_tower = _tower(n_club_features, hidden, embed, dropout)
        self.head = nn.Sequential(
            nn.Linear(embed * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_TARGETS),
        )

    def forward(self, player_x: torch.Tensor, club_x: torch.Tensor) -> torch.Tensor:
        p = self.player_tower(player_x)
        c = self.club_tower(club_x)
        combined = torch.cat([p, c, p * c], dim=1)
        return torch.sigmoid(self.head(combined))


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted sum across heads.

    Training weights are equal by design and are unrelated to the query-time
    weights the CLI exposes — those combine already-trained predictions.
    """
    return nn.functional.huber_loss(pred, target, delta=0.2)
