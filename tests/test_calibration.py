import numpy as np
from sklearn.isotonic import IsotonicRegression

from depthwizard.calibration import IsotonicMap, LinearMap, normalise, oracle_affine


def test_linear_recovers_line_and_clamps():
    x = np.linspace(-1, 2, 1000)
    m = LinearMap().fit(x, 3 * x + 1)
    assert np.isclose(m.a, 3) and np.isclose(m.b, 1)
    assert m(np.array([-5.0]))[0] == 0.0          # heights are clamped at 0


def test_isotonic_interp_matches_sklearn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000); y = np.maximum(0, 2 * x + rng.normal(size=5000))
    m = IsotonicMap().fit(x, y)
    ref = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
    q = np.linspace(-6, 6, 999)
    assert np.allclose(m(q), np.maximum(0, ref.predict(q)))


def test_normalise_is_invariant_to_affine_changes():
    rng = np.random.default_rng(1)
    d = rng.gamma(2, size=(64, 64))
    for method in ("pct", "medmad"):
        assert np.allclose(normalise(d, method), normalise(5 * d + 7, method))


def test_oracle_beats_any_affine_of_same_image():
    rng = np.random.default_rng(2)
    d = rng.normal(size=(64, 64)); gt = np.maximum(0, 4 * d + 3 + rng.normal(size=d.shape))
    v = np.ones_like(d, bool)
    p, a, b = oracle_affine(d, gt, v)
    unclamped = a * d + b
    for aa, bb in [(3, 3), (4, 2.5), (5, 3)]:
        assert ((unclamped - gt) ** 2).mean() <= ((aa * d + bb - gt) ** 2).mean() + 1e-12
