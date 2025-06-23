# utils/mt.py
"""Utility helpers around CTranslate2 + NLLB.

Provides:
    - load_translator: load CTranslate2 translator and HF tokenizer.
    - translate_batch: wrapper for batch translation using tokenizer + translator, with logging.

Usage:
    from mt_pipeline.utils.mt import load_translator, translate_batch

Functions assume you have converted an NLLB-200 model to CTranslate2 format.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import time

import ctranslate2 as ct2
from transformers import AutoTokenizer

# HF repo for tokenizer vocab
_DEFAULT_TOKENIZER_REPO = "facebook/nllb-200-3.3B"


def _get_devices() -> List[str]:
    """Return the list of devices reported by the installed CTranslate2.

    Handles the API changes between versions (available_devices, list_devices, etc.).
    """
    if hasattr(ct2, "available_devices"):
        return ct2.available_devices()
    elif hasattr(ct2, "list_devices"):
        return ct2.list_devices()
    elif hasattr(ct2, "get_supported_devices"):
        return ct2.get_supported_devices()
    elif hasattr(ct2, "Device") and hasattr(ct2.Device, "list_devices"):
        return ct2.Device.list_devices()
    else:
        return []


def _detect_device() -> str:
    devices = _get_devices()
    return "cuda" if "cuda" in devices else "cpu"
