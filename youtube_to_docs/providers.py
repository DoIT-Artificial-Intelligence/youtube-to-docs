from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseProvider(ABC):
    """Base class for all providers."""

    def __init__(self, model_name: str):
        self.model_name = model_name


class LLMProvider(ABC):
    """Interface for Large Language Model services."""

    @abstractmethod
    def generate_content(
        self, prompt: str, **kwargs
    ) -> Tuple[str, int, int]:
        """Returns (response_text, input_tokens, output_tokens)."""
        pass


class STTProvider(ABC):
    """Interface for Speech-to-Text services."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        url: str,
        language: str = "en",
        duration_seconds: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, str, int, int]:
        """Returns (transcript_text, srt_content, input_tokens, output_tokens)."""
        pass


class TTSProvider(ABC):
    """Interface for Text-to-Speech services."""

    @abstractmethod
    def generate_speech(
        self, text: str, voice: str, language_code: Optional[str] = None, **kwargs
    ) -> bytes:
        """Returns raw audio bytes."""
        pass


class TranslationProvider(ABC):
    """Interface for Translation services."""

    @abstractmethod
    def translate(
        self, text: str, target_lang: str, **kwargs
    ) -> str:
        """Returns translated text."""
        pass


class MultimodalProvider(ABC):
    """Interface for Multimodal (Vision) services."""

    @abstractmethod
    def generate_alt_text(
        self, image_bytes: bytes, language: str = "en", **kwargs
    ) -> Tuple[str, int, int]:
        """Returns (alt_text, input_tokens, output_tokens)."""
        pass


_registry: Dict[str, Any] = {}


def register_provider(name: str, provider_class: Any):
    _registry[name] = provider_class


def get_provider(model_name: str) -> BaseProvider:
    """Factory to get the appropriate provider instance for a model name."""
    # This will be populated by concrete implementations
    if model_name.startswith("gemini") or model_name.startswith("gemma"):
        from youtube_to_docs.llms import GeminiProvider
        return GeminiProvider(model_name)
    elif model_name.startswith("vertex"):
        from youtube_to_docs.llms import VertexProvider
        return VertexProvider(model_name)
    elif model_name.startswith("bedrock") or model_name.startswith("nova") or model_name.startswith("claude"):
        from youtube_to_docs.llms import BedrockProvider
        return BedrockProvider(model_name)
    elif model_name.startswith("gcp-"):
        from youtube_to_docs.llms import GCPProvider
        return GCPProvider(model_name)
    elif model_name.startswith("aws-"):
        from youtube_to_docs.llms import AWSProvider
        return AWSProvider(model_name)
    elif model_name.startswith("foundry"):
        from youtube_to_docs.llms import AzureFoundryProvider
        return AzureFoundryProvider(model_name)
    
    raise ValueError(f"Unknown provider for model: {model_name}")
