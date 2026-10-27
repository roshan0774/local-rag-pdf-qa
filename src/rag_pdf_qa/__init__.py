"""Local, private retrieval-augmented question answering over PDFs."""

from .config import Settings
from .pipeline import Answer, RAGPipeline

__all__ = ["Settings", "RAGPipeline", "Answer"]
__version__ = "0.1.0"
