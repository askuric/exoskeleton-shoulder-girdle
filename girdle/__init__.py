"""Shoulder girdle observability and estimation with an arm exoskeleton (companion code)."""
from .estimation import CASES, estimate, estimate_case, sensors
from .model import Shoulder
from .observability import move_along, observability
from .robot import Exoskeleton
