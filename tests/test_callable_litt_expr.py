import pytest

from optcom.utils.callable_litt_expr import CallableLittExpr

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------


@pytest.mark.callable_litt_expr
def test_one_var_only_func():
    """Should fail if the result is not the one expected."""
    a = lambda x : x**2
    b = lambda y : 1 / y
    d = lambda x : x
    CLE = CallableLittExpr([a, b, a, d, b], ['+', '/', '-', '*'])
    assert CLE(3.) ==  (8.+(1./27.))


@pytest.mark.callable_litt_expr
def test_one_var_func_and_float():
    """Should fail if the result is not the one expected."""
    a = lambda x : x**2
    b = lambda y : 1 / y
    d = lambda x : x
    CLE = CallableLittExpr([a, 3., a, 4., b], ['+', '/', '-', '*'])
    assert CLE(3.) == 8.0


@pytest.mark.callable_litt_expr
def test_multi_var_only_func():
    """Should fail if the result is not the one expected."""
    a = lambda x, y : x**2 * y**2
    b = lambda x, y : 1 / (y -x)
    d = lambda x, y : x - y
    CLE = CallableLittExpr([a, b, a, d, b], ['+', '//', '%', '**'])
    assert CLE(3., 4.) == 144.0


@pytest.mark.callable_litt_expr
def test_multi_var_func_and_float():
    """Should fail if the result is not the one expected."""
    a = lambda x, y : x**2 + y**2
    b = lambda x, y : 1 / (y -x)
    d = lambda x, y : x - y
    CLE = CallableLittExpr([a, 3.1, a, 6., b], ['+', '/', '-', '*'])
    assert CLE(3., 4.) == 19.124


@pytest.mark.callable_litt_expr
def test_multi_var_func_and_float_and_parantheses():
    """Should fail if the result is not the one expected."""
    # multi var - func and float
    a = lambda x, y : x**2 + y**2
    b = lambda x, y : 1 / (y -x)
    d = lambda x, y : x - y
    CLE = CallableLittExpr([a, 3.1, a, 6., b], ['(', '+', '/', '-', '*', ')'])
    assert CLE(3., 4.) == 19.124


@pytest.mark.callable_litt_expr
def test_unity_func():
    """Should fail if the result is not the one expected."""
    CLE = CallableLittExpr([lambda identity: identity], [])
    assert CLE(70) == 70
