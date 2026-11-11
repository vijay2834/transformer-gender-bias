import numpy as np

from genderbias.cda import augment, swap
from genderbias.debias import gender_direction, hard_debias
from genderbias.weat import weat


def _planted(rng, bias):
    g = np.zeros(20); g[0] = 1.0
    mk = lambda n, s: rng.normal(0, 1, (n, 20)) + s * g * 3
    return mk(8, bias), mk(8, -bias), mk(8, 1), mk(8, -1)


def test_weat_detects_planted_bias():
    rng = np.random.default_rng(0)
    r = weat(*_planted(rng, 1.0), n_perm=2000)
    assert r.effect_size > 1.2 and r.p_value < 0.01


def test_weat_null_when_no_bias():
    rng = np.random.default_rng(1)
    r = weat(*_planted(rng, 0.0), n_perm=2000)
    assert abs(r.effect_size) < 0.9 and r.p_value > 0.01


def test_weat_effect_is_antisymmetric():
    rng = np.random.default_rng(2)
    X, Y, A, B = _planted(rng, 1.0)
    assert np.isclose(weat(X, Y, A, B, n_perm=10).effect_size, -weat(Y, X, A, B, n_perm=10).effect_size)


def test_hard_debias_removes_gender_component_from_neutral_words():
    rng = np.random.default_rng(3)
    g = np.zeros(10); g[0] = 1
    emb = {"he": g + rng.normal(0, .05, 10), "she": -g + rng.normal(0, .05, 10),
           "man": g + rng.normal(0, .05, 10), "woman": -g + rng.normal(0, .05, 10),
           "nurse": -0.8 * g + rng.normal(0, .3, 10), "engineer": 0.8 * g + rng.normal(0, .3, 10)}
    pairs = [("she", "he"), ("woman", "man")]
    d = gender_direction(emb, pairs)
    out = hard_debias(emb, {"nurse", "engineer"}, pairs, pairs)
    assert abs(out["nurse"] @ d) < 1e-6 and abs(out["engineer"] @ d) < 1e-6
    assert np.isclose(out["he"] @ out["nurse"], out["she"] @ out["nurse"], atol=1e-6)  # equalized


def test_cda_swaps_preserve_case_and_resolve_her():
    assert swap("He gave her book to his Sister.") == "She gave his book to her Brother."
    assert swap("I saw her.") == "I saw him."
    assert swap("The King and the queen") == "The Queen and the king"
    assert len(augment(["he runs"])) == 2
