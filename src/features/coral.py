"""CORAL: match each corpus's full covariance to the pooled training covariance.

    from src.features import coral
    state = coral.fit(x_train, meta_train)
    x_train = coral.apply(x_train, meta_train, state)
    x_test = coral.apply(x_test, meta_test, state)

Sun & Saenko (2016), with a ridge on the covariance before the inverse square
root. Each corpus is centred, whitened by its own training covariance, recoloured
by the pooled training covariance, and shifted to the pooled training mean.

Why this is the *only* second-order increment left to test: `store.fit_stats(...,
"corpus")` already equalises per-corpus means and per-corpus marginal standard
deviations, and correlation is invariant to a per-column positive-scale affine
map. So the corpus correlation matrices survive that transform untouched, and
they are exactly what CORAL removes and per-corpus z-scoring does not.

Why its numbers are not comparable to `global`: like the `corpus` scope, this
needs a row's corpus label at apply time to pick its map. It is a domain-
*conditional* transform, not domain removal, so it cannot serve strict LODO
except through the same pooled `__all__` fallback that took ItalianPD to 0.504.
Report it as a transductive extra, never as a DANN competitor. Contrast
`project.py`, which is one global map. See
`experiments/corpus_norm_ablation/PLAN_second_order.md`.

Vectors only. Do not CORAL Mel frames.
"""

import numpy as np

RIDGE = 1e-3


def _sqrt_and_inverse_sqrt(cov, ridge):
    """Symmetric square root and inverse square root of a covariance."""
    cov = cov + ridge * np.eye(cov.shape[0])
    eigenvalues, vectors = np.linalg.eigh(cov)
    # eigh on a ridged covariance should not return negatives, but clip rather
    # than take the square root of a small negative from rounding.
    eigenvalues = np.maximum(eigenvalues, ridge)
    root = vectors @ np.diag(np.sqrt(eigenvalues)) @ vectors.T
    inverse = vectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ vectors.T
    return root, inverse


def fit(values, meta, ridge=RIDGE):
    """Per-corpus alignment maps, to be fitted on training rows only.

    The reference is the pooled *training* covariance of this configuration,
    not any one corpus.
    """
    pooled_mean = values.mean(axis=0)
    pooled_root, _ = _sqrt_and_inverse_sqrt(np.cov(values, rowvar=False), ridge)
    maps = {}
    for name in sorted(set(meta.dataset.values)):
        rows = np.flatnonzero(np.asarray(meta.dataset.values) == name)
        if len(rows) < values.shape[1] + 1:
            # Fewer rows than dimensions makes the covariance singular in a way
            # the ridge would paper over silently. Leave the corpus alone.
            maps[name] = None
            continue
        _, inverse = _sqrt_and_inverse_sqrt(
            np.cov(values[rows], rowvar=False), ridge)
        maps[name] = (values[rows].mean(axis=0), inverse @ pooled_root)
    return {"maps": maps, "pooled_mean": pooled_mean,
            "pooled_root": pooled_root, "ridge": ridge}


def apply(values, meta, state):
    """Align `values` with previously fitted maps.

    A corpus absent from the fitted set -- the strict LODO target -- is left as
    it arrived rather than aligned with someone else's map. That is the same
    `__all__` convention `store.apply` uses, and the same admission: the method
    has no defined behaviour on a corpus it has not seen.
    """
    out = np.array(values, copy=True)
    names = np.asarray(meta.dataset.values)
    for name in np.unique(names):
        entry = state["maps"].get(name)
        if entry is None:
            continue
        rows = np.flatnonzero(names == name)
        mean, transform = entry
        out[rows] = (values[rows] - mean) @ transform + state["pooled_mean"]
    return out


def verify(values, meta, state, tolerance=0.05):
    """Assert per-corpus covariance matches the reference on the fitted set."""
    aligned = apply(values, meta, state)
    reference = state["pooled_root"] @ state["pooled_root"]
    names = np.asarray(meta.dataset.values)
    worst = 0.0
    for name, entry in state["maps"].items():
        if entry is None:
            continue
        rows = np.flatnonzero(names == name)
        after = np.cov(aligned[rows], rowvar=False)
        relative = (np.linalg.norm(after - reference)
                    / max(np.linalg.norm(reference), 1e-12))
        worst = max(worst, relative)
    if worst > tolerance:
        raise ValueError(
            f"CORAL left a {worst:.3f} relative covariance gap on the fitted "
            f"set (tolerance {tolerance})")
    return worst
