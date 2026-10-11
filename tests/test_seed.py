import random

import numpy as np
import pytest

from burnscars.seed import set_seed


def test_same_seed_same_stream():
    set_seed(7)
    a = (random.random(), np.random.rand(3).tolist())
    set_seed(7)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b


def test_different_seed_different_stream():
    set_seed(1)
    a = np.random.rand(3)
    set_seed(2)
    assert not np.allclose(a, np.random.rand(3))


@pytest.mark.parametrize("bad", [-1, 1.5, "3", None])
def test_rejects_bad_seed(bad):
    with pytest.raises(ValueError):
        set_seed(bad)
