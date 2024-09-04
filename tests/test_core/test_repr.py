from tests.test_core.conftest import BaseDummy


def test_repr_public_property_private_attribute():
    dummy = BaseDummy(a=1, b=2)
    assert repr(dummy) == "Dummy(a = 1, b = 2, )"
