"""
Generation interfaces following SOLID principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class GenerationModel(Enum):
    """Types of generation models."""
    LLM = "llm"
    TEMPLATE = "template"
    HYBRID = "hybrid"


@dataclass
class GenerationContext:
    """Context for text generation."""
    query: str
    retrieved_content: list[dict[str, Any]]
    user_context: dict[str, Any] | None = None
    generation_parameters: dict[str, Any] | None = None

    def __post_init__(self):
        if self.user_context is None:
            self.user_context = {}
        if self.generation_parameters is None:
            self.generation_parameters = {}


@dataclass
class GenerationResult:
    """Result of text generation."""
    generated_text: str
    model_used: GenerationModel
    confidence_score: float
    metadata: dict[str, Any]
    source_citations: list[str] = None

    def __post_init__(self):
        if self.source_citations is None:
            self.source_citations = []


class TextGenerator(ABC):
    """Abstract base class for text generators."""

    @abstractmethod
    def generate(self, context: GenerationContext) -> GenerationResult:
        """Generate text based on context."""
        pass

    @abstractmethod
    def get_model_type(self) -> GenerationModel:
        """Get the type of generation model."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the generator is available."""
        pass


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def create_prompt(self, context: GenerationContext) -> str:
        """Create a prompt from the given context."""
        pass

    @abstractmethod
    def get_template_name(self) -> str:
        """Get the name of this template."""
        pass

    @abstractmethod
    def supports_context_type(self, context_type: str) -> bool:
        """Check if this template supports the given context type."""
        pass


class GenerationService(ABC):
    """Abstract base class for generation services."""

    @abstractmethod
    def add_generator(self, generator: TextGenerator) -> None:
        """Add a text generator to the service."""
        pass

    @abstractmethod
    def remove_generator(self, model_type: GenerationModel) -> None:
        """Remove a generator from the service."""
        pass

    @abstractmethod
    def generate_response(self, context: GenerationContext) -> GenerationResult:
        """Generate a response using the best available generator."""
        pass

    @abstractmethod
    def get_available_models(self) -> list[GenerationModel]:
        """Get list of available generation models."""
        pass
