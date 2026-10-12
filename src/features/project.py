"""Remove the linearly corpus-predictive directions from a feature matrix.

The shallow analogue of DANN's domain head: instead of learning an encoder that
a corpus classifier cannot read, fit a *linear* corpus classifier and delete the
directions it uses.

    from src.features import project
    state = project.fit(x_train, meta_train, iters=2)
    x_train = project.apply(x_train, state)
    x_test = project.apply(x_test, state)

Unlike `store.fit_stats(..., "corpus")` and `coral.fit`, this is **one global
map applied identically to every row**. It never looks at a row's corpus label
at apply time, so its corpus balanced accuracy is comparable to the `global`
reference and it is computable under strict LODO on a corpus it has not seen.
That is what makes it the only shallow control in this ablation that is
like-for-like with DANN. See `experiments/corpus_norm_ablation/PLAN_second_order.md`.

Method is INLP (Ravfogel et al. 2020) with one deviation. Composing per-round
projectors gives a matrix that is not itself a projector, so instead every
round's coefficient rows are accumulated and a single projector is rebuilt from
the whole stack, via an orthonormal basis `B` of its row space:

    P = I - B.T @ B

which is symmetric and idempotent *by construction* -- `I - pinv(W) @ W` is the
same subspace but drifts off symmetry by more than 1e-8 once the coefficient
rows are near-dependent, which they are for a 3-class classifier. It removes
exactly the span of every direction found so far, and `verify` asserts both
properties plus the resulting rank.

PD/HC labels never enter the fit. Only `meta.dataset` is read.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

MAX_ITER = 2000


def fit(values, meta, iters=1, seed=0):
    """Projector removing the corpus-predictive directions of `values`.

    Fit on training rows only. `iters` rounds, each re-fitting the corpus
    classifier on what the previous rounds left behind, so later rounds find
    directions that were not visible while the earlier ones were still present.
    """
    n_feat = values.shape[1]
    if iters == 0:
        return {"P": np.eye(n_feat), "removed": 0, "iters": 0}

    corpora = np.asarray(meta.dataset.values)
    stack = np.zeros((0, n_feat))
    P = np.eye(n_feat)
    for _ in range(iters):
        model = LogisticRegression(max_iter=MAX_ITER, random_state=seed)
        model.fit(values @ P, corpora)
        # Coefficients live in the projected space, but P is idempotent and
        # symmetric, so P @ w is the same direction expressed in the original
        # basis -- and it is already orthogonal to everything removed so far.
        stack = np.vstack([stack, model.coef_ @ P])
        P, removed = _null_space_projector(stack, n_feat)
    return {"P": P, "removed": removed, "iters": int(iters)}


def _null_space_projector(stack, n_feat):
    """Projector onto the orthogonal complement of `stack`'s row space."""
    _, singular, right = np.linalg.svd(stack, full_matrices=False)
    keep = singular > singular.max() * 1e-10
    basis = right[keep]
    return np.eye(n_feat) - basis.T @ basis, int(keep.sum())


def apply(values, state):
    """Project `values` with a previously fitted projector."""
    return values @ state["P"]


def verify(state, n_feat):
    """Assert the map is a genuine projector of the expected rank."""
    P = state["P"]
    if P.shape != (n_feat, n_feat):
        raise ValueError(f"projector is {P.shape}, expected {(n_feat, n_feat)}")
    if not np.allclose(P, P.T, atol=1e-8):
        raise ValueError("projector is not symmetric")
    if not np.allclose(P @ P, P, atol=1e-8):
        raise ValueError("projector is not idempotent")
    rank = int(np.linalg.matrix_rank(P))
    if rank != n_feat - state["removed"]:
        raise ValueError(
            f"projector rank {rank}, expected {n_feat - state['removed']} "
            f"after removing {state['removed']} directions")
    return rank
