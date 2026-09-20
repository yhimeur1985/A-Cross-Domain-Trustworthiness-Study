"""Finite-difference check of the hand-derived alignment gradients.

    python -m tests.test_gradients      (from the lodo_audit folder)
"""
import numpy as np

from iotaudit.models import moment_alignment


def check(cov_weight):
    rng = np.random.default_rng(0)
    h = rng.standard_normal((30, 5))
    groups = [np.arange(0, 10), np.arange(10, 18), np.arange(18, 30)]
    v, g = moment_alignment(h, groups, cov_weight)
    num = np.zeros_like(h)
    eps = 1e-6
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            hp, hm = h.copy(), h.copy()
            hp[i, j] += eps
            hm[i, j] -= eps
            num[i, j] = (moment_alignment(hp, groups, cov_weight)[0]
                         - moment_alignment(hm, groups, cov_weight)[0]) / (2 * eps)
    err = np.abs(num - g).max() / (np.abs(num).max() + 1e-12)
    print(f"cov_weight={cov_weight:>6}: relative max error {err:.2e}")
    assert err < 1e-5


if __name__ == "__main__":
    for cw in (0.0, 1.0, 500.0):
        check(cw)
    print("gradient checks passed")
