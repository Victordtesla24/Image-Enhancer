"""Core processor package."""

from .base import BaseProcessor
from .batch import BatchProcessor

class Processor(BatchProcessor):
    """Main processor class combining all functionality."""
    pass

__all__ = ['Processor', 'BaseProcessor', 'BatchProcessor'] 