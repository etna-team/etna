from tests.test_core.conftest import Dummy


def test_repr_public_property_private_attribute():
    dummy = Dummy()
    assert repr(dummy) == "Dummy(a = 1, b = 2, )"
