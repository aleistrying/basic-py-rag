"""
Unified Embedding Service
Supports both local models (SentenceTransformers) and API-based models (OpenAI)
"""

import os
import logging
from typing import List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings for input texts"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using SentenceTransformers
    100% offline after initial model download
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu",
        cache_folder: str = "./models/embeddings"
    ):
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.model = None
        self._dimension = 768  # Default for E5-base

        # Ensure cache folder exists
        os.makedirs(cache_folder, exist_ok=True)

        # Configure for offline operation
        os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force offline mode
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self._load_model()

    def _load_model(self):
        """Load model with offline-first configuration"""
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # Detect device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                f"🔧 Loading local embedding model: {self.model_name} on {self.device}")

            # Load model with explicit cache directory for offline usage
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder,
                trust_remote_code=False,
                use_auth_token=False
            )

            # Optimize for device
            if self.device == "cuda":
                self.model = self.model.cuda()
                logger.info("🚀 Model loaded on GPU")
            else:
                self.model = self.model.cpu()
                logger.info("💻 Model loaded on CPU")

            # Get actual dimension
            test_embedding = self.model.encode(
                ["test"], show_progress_bar=False)
            self._dimension = test_embedding.shape[1]

            logger.info(
                f"✅ Local embedding model loaded successfully (dim={self._dimension})")

        except Exception as e:
            logger.error(f"❌ Failed to load local embedding model: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings using local model"""
        if isinstance(texts, str):
            texts = [texts]

        # Add E5 prefix for better performance
        prefixed_texts = [f"passage: {text}" for text in texts]

        embeddings = self.model.encode(
            prefixed_texts,
            show_progress_bar=kwargs.get("show_progress_bar", False),
            batch_size=kwargs.get("batch_size", 32),
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )

        return embeddings

    def get_dimension(self) -> int:
        return self._dimension

    def is_available(self) -> bool:
        return self.model is not None


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider
    Requires API key and internet connection
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.client = None

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }

        self._load_client()

    def _load_client(self):
        """Initialize OpenAI client"""
        if not self.api_key:
            logger.warning(
                "⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return

        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            logger.info(
                f"✅ OpenAI embedding client initialized (model={self.model})")

        except ImportError:
            logger.error(
                "❌ OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check API key.")

        if isinstance(texts, str):
            texts = [texts]

        try:
            # OpenAI supports batch requests
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"❌ OpenAI embedding error: {e}")
            raise

    def get_dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)

    def is_available(self) -> bool:
        return self.client is not None


class EmbeddingService:
    """
    Unified embedding service that manages multiple providers
    """

    def __init__(
        self,
        provider: str = "local",
        local_model: str = "intfloat/multilingual-e5-base",
        openai_model: str = "text-embedding-3-small",
        device: str = "cpu",
        cache_folder: str = "./models/embeddings"
    ):
        self.provider_name = provider
        self.provider: Optional[EmbeddingProvider] = None

        # Initialize provider
        if provider == "local":
            self.provider = LocalEmbeddingProvider(
                model_name=local_model,
                device=device,
                cache_folder=cache_folder
            )
        elif provider == "openai":
            self.provider = OpenAIEmbeddingProvider(
                model=openai_model
            )
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'local' or 'openai'")

        logger.info(
            f"🎯 Embedding service initialized with provider: {provider}")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings"""
        if not self.provider or not self.provider.is_available():
            raise RuntimeError(
                f"Embedding provider '{self.provider_name}' not available")

        return self.provider.encode(texts, **kwargs)

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.provider.get_dimension()

    def is_available(self) -> bool:
        """Check if service is available"""
        return self.provider is not None and self.provider.is_available()

    def get_provider_info(self) -> dict:
        """Get provider information"""
        return {
            "provider": self.provider_name,
            "dimension": self.get_dimension(),
            "available": self.is_available()
        }


# Convenience function for backward compatibility
def get_embedding_service(
    provider: str = None,
    **kwargs
) -> EmbeddingService:
    """
    Get embedding service instance

    Args:
        provider: "local" (default) or "openai"
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingService instance

    Examples:
        # Local model (fully offline)
        service = get_embedding_service(provider="local")

        # OpenAI API
        service = get_embedding_service(provider="openai")
    """
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "local")
    return EmbeddingService(provider=provider, **kwargs)
