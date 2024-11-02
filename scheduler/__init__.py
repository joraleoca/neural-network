"""
This module implements various learning rate schedulers for neural networks.

Classes:
    Scheduler: Base class for all schedulers.
    FactorScheduler: Adjusts the learning rate by a specified factor.
    CosineScheduler: Adjusts the learning rate using a cosine annealing schedule.
"""

from .scheduler import Scheduler
from .factor_scheduler import FactorScheduler
from .cosine_scheduler import CosineScheduler

__all__ = ["Scheduler", "FactorScheduler", "CosineScheduler"]
