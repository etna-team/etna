from typing import List
from typing import Union


def shift(steps_number: int, values: List[int]) -> List[int]:
    """Shift values for steps_number steps."""
    return [v + steps_number for v in values]


def mult(first: int, second: int) -> int:
    """Multiply arguments."""
    return first * second


def concat(first: Union[int, str], second: Union[int, str]) -> str:
    """Concat arguments."""
    return str(first) + str(second)


def arange(start: int, stop: int, gap: int = 1) -> List[int]:
    """Make range."""
    return list(range(start, stop, gap))
