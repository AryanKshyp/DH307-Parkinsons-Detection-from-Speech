"""Geometry of learned embeddings: KL, TCM, t-SNE.

KL is computed between diagonal Gaussians in a PCA space fitted on the
pooled embeddings. Raw 128-d Gaussians are singular with 16–58 speakers per
cell; 16 components is enough to see whether DANN collapsed corpus structure
without inventing a density estimator.

TCM is the trace of the (unreduced) covariance of intra-class embeddings,
matching Ibarra et al. Table 6: smaller means the class is tighter.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def embedding_cols(frame):
    return [c for c in frame.columns if c.startswith("e") and c[1:].isdigit()]


def _gaussian_kl(mean_p, std_p, mean_q, std_q):
    """KL(N_p || N_q) for diagonal Gaussians."""
    var_p = np.square(std_p)
    var_q = np.square(std_q)
    return 0.5 * np.sum(
        np.log(var_q / var_p) + var_p / var_q
        + np.square(mean_p - mean_q) / var_q - 1.0)


def pairwise_kl(frame, dim=16, seed=0):
    """Symmetric KL between corpora, within each class."""
    cols = embedding_cols(frame)
    x = frame[cols].to_numpy(dtype=np.float64)
    n = min(dim, x.shape[0] - 1, x.shape[1])
    z = PCA(n_components=n, random_state=seed).fit_transform(x)
    z = pd.DataFrame(z, index=frame.index)
    rows = []
    corpora = sorted(frame.dataset.unique())
    for label in ("HC", "PD"):
        at_label = frame.label == label
        for i, left in enumerate(corpora):
            for right in corpora[i + 1:]:
                a = z[at_label & (frame.dataset == left)].to_numpy()
                b = z[at_label & (frame.dataset == right)].to_numpy()
                if len(a) < 3 or len(b) < 3:
                    rows.append({"label": label, "pair": f"{left}/{right}",
                                 "kl": float("nan"), "n_a": len(a), "n_b": len(b)})
                    continue
                ma, sa = a.mean(0), np.clip(a.std(0), 1e-6, None)
                mb, sb = b.mean(0), np.clip(b.std(0), 1e-6, None)
                kl = 0.5 * (_gaussian_kl(ma, sa, mb, sb)
                            + _gaussian_kl(mb, sb, ma, sa))
                rows.append({"label": label, "pair": f"{left}/{right}",
                             "kl": float(kl), "n_a": len(a), "n_b": len(b)})
    return pd.DataFrame(rows)


def tcm(frame):
    """Trace of the covariance of embeddings, per class and per class x corpus."""
    cols = embedding_cols(frame)
    rows = []
    for label, group in frame.groupby("label"):
        x = group[cols].to_numpy(dtype=np.float64)
        rows.append({"label": label, "dataset": "all",
                     "tcm": float(np.trace(np.cov(x, rowvar=False))),
                     "n": len(group)})
        for dataset, sub in group.groupby("dataset"):
            if len(sub) < 2:
                continue
            x = sub[cols].to_numpy(dtype=np.float64)
            rows.append({"label": label, "dataset": dataset,
                         "tcm": float(np.trace(np.cov(x, rowvar=False))),
                         "n": len(sub)})
    return pd.DataFrame(rows)
