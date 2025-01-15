from etna.core.mixins import BaseMixin


class BaseDummy(BaseMixin):
    def __init__(self, a: int = 1, b: int = 2):
        self.a = a
        self._b = b

    @property
    def b(self):
        return self._b
