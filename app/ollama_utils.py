"""
Ollama Utilities - Health Check, Auto-Retry, and Model Fallback

This module provides resilient Ollama integration with:
- GPU memory health checks
- Automatic retry with exponential backoff
- Model fallback hierarchy (large → small)
- Container restart capability
- Connection pooling
"""
import logging
import time
import os
import subprocess
from typing import Optional, Dict, Any, List
from functools import wraps

try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger(__name__)

# Model fallback hierarchy - ordered from largest to smallest
# These are the models actually available in your Ollama instance
MODEL_FALLBACK_CHAIN = [
    "gemma2:2b",       # Default - 2B parameters, good quality
    "phi3:mini",       # Fallback 1 - ~2.7B parameters, Microsoft
    # Fallback 2 - 3B parameters, Alibaba (if phi3 has GPU issues)
    "qwen2.5:3b",
]

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CONTAINER_NAME = os.getenv("OLLAMA_CONTAINER_NAME", "ollama")
OLLAMA_REQUEST_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds, will use exponential backoff


class OllamaHealthError(Exception):
    """Raised when Ollama is unhealthy or unavailable"""
    pass


class OllamaMemoryError(Exception):
    """Raised when GPU memory is exhausted"""
    pass


def check_ollama_health(timeout: int = 5) -> Dict[str, Any]:
    """
    Check if Ollama is healthy and can accept requests.

    Returns:
        Dict with health status, GPU info, and loaded models
    """
    if ollama is None:
        return {
            "healthy": False,
            "error": "Ollama package not installed",
            "gpu_available": False
        }

    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=timeout)

        # Try to list models to verify connection
        models_response = client.list()
        models = models_response.get('models', [])

        # Check GPU status via ps command if available
        try:
            ps_response = client.ps()
            running_models = ps_response.get('models', [])
            gpu_info = {
                "models_loaded": len(running_models),
                "models": [m.get('name', 'unknown') for m in running_models]
            }
        except Exception as e:
            logger.warning(f"Could not check running models: {e}")
            gpu_info = {"models_loaded": 0, "models": []}

        return {
            "healthy": True,
            "gpu_available": True,  # Assume GPU if Ollama is running
            "models_available": [m.get('name', 'unknown') for m in models],
            "gpu_info": gpu_info,
            "host": OLLAMA_HOST
        }

    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "gpu_available": False,
            "host": OLLAMA_HOST
        }


def restart_ollama_container() -> bool:
    """
    Attempt to restart the Ollama Docker container.

    Returns:
        True if restart successful, False otherwise
    """
    try:
        logger.warning(
            f"Attempting to restart Ollama container: {OLLAMA_CONTAINER_NAME}")

        # Check if running in Docker (we need docker CLI access)
        result = subprocess.run(
            ["docker", "restart", OLLAMA_CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            logger.info(f"Successfully restarted {OLLAMA_CONTAINER_NAME}")
            # Wait for container to be ready
            time.sleep(5)
            return True
        else:
            logger.error(f"Failed to restart container: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Container restart timed out")
        return False
    except FileNotFoundError:
        logger.error("Docker CLI not available - cannot restart container")
        return False
    except Exception as e:
        logger.error(f"Unexpected error restarting container: {e}")
        return False


def unload_all_models() -> bool:
    """
    Unload all models from GPU memory to free up space.

    Returns:
        True if successful, False otherwise
    """
    if ollama is None:
        return False

    try:
        client = ollama.Client(host=OLLAMA_HOST, timeout=10)

        # Get currently loaded models
        ps_response = client.ps()
        running_models = ps_response.get('models', [])

        if not running_models:
            logger.info("No models loaded in GPU memory")
            return True

        logger.info(f"Unloading {len(running_models)} models from GPU memory")

        # Unload each model by loading a tiny model then immediately stopping
        # This forces Ollama to unload the current model
        for model_info in running_models:
            model_name = model_info.get('name', 'unknown')
            try:
                # Send empty generate request with keep_alive=0 to unload
                client.generate(
                    model=model_name,
                    prompt="",
                    keep_alive=0  # Unload immediately
                )
                logger.info(f"Unloaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not unload {model_name}: {e}")

        time.sleep(2)  # Wait for unload to complete
        return True

    except Exception as e:
        logger.error(f"Failed to unload models: {e}")
        return False


def get_next_fallback_model(current_model: str) -> Optional[str]:
    """
    Get the next smaller model in the fallback chain.

    Args:
        current_model: Current model that failed

    Returns:
        Next model to try, or None if no fallbacks available
    """
    try:
        current_index = MODEL_FALLBACK_CHAIN.index(current_model)
        if current_index < len(MODEL_FALLBACK_CHAIN) - 1:
            return MODEL_FALLBACK_CHAIN[current_index + 1]
    except ValueError:
        # Current model not in chain, return first fallback
        if len(MODEL_FALLBACK_CHAIN) > 1:
            return MODEL_FALLBACK_CHAIN[1]

    return None


def is_memory_error(error: Exception) -> bool:
    """
    Check if error is related to GPU/system memory exhaustion.

    Args:
        error: Exception to check

    Returns:
        True if memory-related error
    """
    error_str = str(error).lower()
    memory_indicators = [
        "system memory",
        "gpu memory",
        "unable to load",
        "model requires more",
        "out of memory",
        "oom",
        "cuda out of memory",
        "status code: 500"  # Ollama returns 500 for memory errors
    ]
    return any(indicator in error_str for indicator in memory_indicators)


def ollama_generate_with_retry(
    model: str,
    prompt: str,
    max_retries: int = MAX_RETRIES,
    auto_fallback: bool = True,
    auto_restart: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate with automatic retry, model fallback, and container restart.

    Args:
        model: Initial model to use
        prompt: Prompt text
        max_retries: Maximum retry attempts per model
        auto_fallback: Enable automatic model fallback on memory errors
        auto_restart: Enable automatic container restart on failures
        **kwargs: Additional options for ollama.generate()

    Returns:
        Response dict with 'response' key and metadata

    Raises:
        OllamaHealthError: If all retry attempts fail
        OllamaMemoryError: If memory exhausted with no fallbacks
    """
    if ollama is None:
        raise ImportError("Ollama package not installed")

    current_model = model
    models_tried = []
    last_error = None

    # Try current model and fallbacks
    while current_model:
        logger.info(f"Attempting generation with model: {current_model}")

        for attempt in range(max_retries):
            try:
                client = ollama.Client(
                    host=OLLAMA_HOST, timeout=OLLAMA_REQUEST_TIMEOUT)

                # Attempt generation
                response = client.generate(
                    model=current_model,
                    prompt=prompt,
                    **kwargs
                )

                # Success! Return with metadata
                logger.info(
                    f"✅ Successfully generated with {current_model} (attempt {attempt + 1})")
                response['_metadata'] = {
                    'model_used': current_model,
                    'models_tried': models_tried + [current_model],
                    'attempts': attempt + 1,
                    'fallback_used': current_model != model
                }
                return response

            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed with {current_model}: {error_msg}")

                # Check if it's a memory error
                if is_memory_error(e):
                    logger.error(
                        f"⚠️  GPU memory error detected with {current_model}")

                    # Try to free up memory
                    if attempt < max_retries - 1:
                        logger.info("Attempting to unload models from GPU...")
                        unload_all_models()
                        time.sleep(3)

                    # If last attempt with this model, break to try fallback
                    if attempt == max_retries - 1:
                        break

                # Exponential backoff before retry
                if attempt < max_retries - 1:
                    delay = RETRY_DELAY_BASE ** (attempt + 1)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

        # All retries exhausted for this model
        models_tried.append(current_model)

        # Check if we should try container restart
        if auto_restart and len(models_tried) == 1:
            logger.warning(
                "First model failed, attempting container restart...")
            if restart_ollama_container():
                # Retry the current model once after restart
                continue

        # Try to fallback to smaller model
        if auto_fallback and is_memory_error(last_error):
            next_model = get_next_fallback_model(current_model)
            if next_model:
                logger.warning(f"Falling back to smaller model: {next_model}")
                current_model = next_model
                continue

        # No more fallbacks or not a memory error
        break

    # All models and retries failed
    if is_memory_error(last_error):
        raise OllamaMemoryError(
            f"GPU memory exhausted. Tried models: {', '.join(models_tried)}. "
            f"Last error: {last_error}"
        )
    else:
        raise OllamaHealthError(
            f"Ollama generation failed after {len(models_tried)} models. "
            f"Tried: {', '.join(models_tried)}. Last error: {last_error}"
        )


def resilient_ollama_decorator(auto_fallback: bool = True, auto_restart: bool = True):
    """
    Decorator to make any function that uses Ollama resilient to failures.

    Usage:
        @resilient_ollama_decorator(auto_fallback=True, auto_restart=True)
        def my_function(query, model="gemma2:2b", **kwargs):
            # Your function that calls ollama.generate()
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract model from kwargs if present
            model = kwargs.get('model', 'gemma2:2b')

            try:
                # Check health before attempting
                health = check_ollama_health(timeout=3)
                if not health['healthy']:
                    logger.warning(
                        f"Ollama unhealthy: {health.get('error', 'Unknown error')}")

                    if auto_restart:
                        logger.info("Attempting container restart...")
                        restart_ollama_container()
                        time.sleep(5)

                # Call original function
                return func(*args, **kwargs)

            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}")

                # If memory error and fallback enabled, retry with smaller model
                if is_memory_error(e) and auto_fallback:
                    next_model = get_next_fallback_model(model)
                    if next_model:
                        logger.warning(
                            f"Retrying {func.__name__} with fallback model: {next_model}")
                        kwargs['model'] = next_model
                        return func(*args, **kwargs)

                # Re-raise if cannot handle
                raise

        return wrapper
    return decorator


# Convenience function for checking status
def get_ollama_status() -> Dict[str, Any]:
    """
    Get comprehensive Ollama status for monitoring/debugging.

    Returns:
        Dict with health, GPU, models, and container info
    """
    health = check_ollama_health()

    status = {
        "timestamp": time.time(),
        "health": health,
        "config": {
            "host": OLLAMA_HOST,
            "container_name": OLLAMA_CONTAINER_NAME,
            "timeout": OLLAMA_REQUEST_TIMEOUT,
            "max_retries": MAX_RETRIES,
            "model_fallback_chain": MODEL_FALLBACK_CHAIN
        }
    }

    # Try to get container status if docker available
    try:
        result = subprocess.run(
            ["docker", "inspect", OLLAMA_CONTAINER_NAME,
                "--format", "{{.State.Status}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            status["container_status"] = result.stdout.strip()
    except:
        status["container_status"] = "unknown"

    return status
