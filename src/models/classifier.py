"""PD head, optional domain head, gradient reversal between them.

A baseline run constructs this with `n_domain=None`, so the domain head is
not present at all -- a dead head would still change capacity and BatchNorm
traffic. A DANN run passes the number of corpora in the *training* split,
which is 3 under mixed and 2 under strict LODO.
"""

import torch.nn as nn

from src.models.grl import grl


class Classifier(nn.Module):
    def __init__(self, trunk, n_domain=None):
        super().__init__()
        self.trunk = trunk
        self.label = nn.Linear(trunk.out_dim, 2)
        self.domain = (nn.Linear(trunk.out_dim, n_domain)
                       if n_domain and n_domain > 1 else None)

    def forward(self, x, mask=None, lam=0.0):
        h = self.trunk(x, mask=mask)
        out = {"features": h, "label": self.label(h)}
        if self.domain is not None:
            out["domain"] = self.domain(grl(h, lam))
        return out
