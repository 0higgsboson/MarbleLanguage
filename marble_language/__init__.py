"""
Marble Language v2 - Enhanced Artificial Language System

A machine learning project implementing a transformer-based language model
for a constrained marble language with collision detection and wall systems.
"""

__version__ = "2.0.0"
__author__ = "Marble Language Team"

from .core.config import MARBLE_CONFIG, MarbleConfig, validate_sentence_rules
from .core.generator import EnhancedMarbleSentenceGenerator
from .utils.validation import MarbleLanguageValidator

# Training modules require torch - import conditionally
try:
    from .training.model import MarbleTransformer
    from .training.trainer import MarbleLanguageTrainer
    _TRAINING_AVAILABLE = True
except ImportError:
    MarbleTransformer = None
    MarbleLanguageTrainer = None
    _TRAINING_AVAILABLE = False

__all__ = [
    'MARBLE_CONFIG',
    'MarbleConfig', 
    'validate_sentence_rules',
    'EnhancedMarbleSentenceGenerator',
    'MarbleLanguageValidator'
]

if _TRAINING_AVAILABLE:
    __all__.extend(['MarbleTransformer', 'MarbleLanguageTrainer'])