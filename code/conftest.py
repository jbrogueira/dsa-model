"""Pytest configuration.

On Apple Silicon JAX defaults to the experimental Metal backend, which has no
float64. lifecycle_jax.py selects the CPU platform at import, but a test that
imports jax before lifecycle_jax would leave the whole session on Metal. Set
the platform here, before any test module is imported; setdefault preserves an
explicit user override.
"""
import os
import platform

if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
