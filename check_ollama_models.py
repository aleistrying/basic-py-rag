#!/usr/bin/env python3
"""
Check and Update Ollama Model Availability

This script checks which models are actually available and updates the fallback chain.
"""

import requests
import subprocess
import json


def get_available_models():
    """Get list of available models from Ollama"""
    try:
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"Error running ollama list: {result.stderr}")
            return []

        lines = result.stdout.strip().split('\n')
        models = []

        # Skip header line and parse model names
        for line in lines[1:]:
            if line.strip():
                model_name = line.split()[0]  # First column is the model name
                models.append(model_name)

        return models

    except Exception as e:
        print(f"Error getting models: {e}")
        return []


def check_model_sizes():
    """Get model details including sizes"""
    try:
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        print("Available Ollama Models:")
        print("=" * 50)
        print(result.stdout)

        return result.stdout

    except Exception as e:
        print(f"Error: {e}")
        return ""


def test_health_endpoint():
    """Test the health endpoint to see what it reports"""
    try:
        response = requests.get(
            "http://localhost:8080/ollama/health", timeout=5)
        data = response.json()

        print("\nHealth Endpoint Response:")
        print("=" * 50)
        print(json.dumps(data, indent=2))

        return data

    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return {}


def main():
    print("🔍 Ollama Model Availability Check")
    print("=" * 50)

    # Check actual available models
    models = get_available_models()
    print(f"\nFound {len(models)} models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")

    # Show detailed model list with sizes
    check_model_sizes()

    # Test health endpoint
    test_health_endpoint()

    # Recommend optimal fallback chain
    print("\n🎯 Recommended Fallback Chain:")
    print("=" * 50)

    if models:
        # Sort by likely size (smaller models first for fallback)
        size_order = {
            'gemma2:2b': 1,
            'phi3:mini': 2,
            'qwen2.5:3b': 3,
            'llama3.1:8b': 4
        }

        available_sorted = sorted(
            [m for m in models if m in size_order],
            key=lambda x: size_order.get(x, 999)
        )

        print("Current chain should be:")
        for i, model in enumerate(available_sorted, 1):
            label = "(Default)" if i == 1 else f"(Fallback {i-1})"
            print(f"  {i}. {model} {label}")
    else:
        print("⚠️  No models found!")


if __name__ == "__main__":
    main()
