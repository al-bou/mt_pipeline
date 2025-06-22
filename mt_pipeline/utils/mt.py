# utils/mt.py
"""Utility helpers around CTranslate2 + NLLB.

For Option B we commençons par *load_translator* :
    >>> from utils.mt import load_translator
    >>> translator, tokenizer = load_translator("/home/you/models/nllb-3.3B-int8")

The translator is a ctranslate2.Translator ready for CPU (or GPU if
available).  The tokenizer is the Hugging Face AutoTokenizer from the original
NLLB checkpoint, so we can use it to encode/decode segments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import ctranslate2 as ct2
from transformers import AutoTokenizer

# Default NLLB repo used only for the tokenizer (weights are not downloaded)
_DEFAULT_TOKENIZER_REPO = "facebook/nllb-200-3.3B"


def _detect_device() -> str:
    """Return "cuda" if a compatible GPU is visible to CTranslate2, else "cpu"."""
    if ct2.get_supported_devices() and "cuda" in ct2.get_supported_devices():
        return "cuda"
    return "cpu"


def load_translator(
    model_dir: str | Path,
    *,
    beam: int = 5,
    device: str | None = None,
    inter_threads: int | None = None,
    intra_threads: int | None = None,
) -> Tuple[ct2.Translator, AutoTokenizer]:
    """Return a *(translator, tokenizer)* tuple ready for batch translation.

    Parameters
    ----------
    model_dir : str | Path
        Path to the CTranslate2‑converted NLLB model (e.g. `~/models/nllb-3.3B-int8`).
    beam : int, default=5
        Beam size used for `translate_batch` (kept in the translator object).
    device : {"cuda", "cpu", "auto"}, optional
        "auto" (None) chooses GPU if available, else CPU.
    inter_threads / intra_threads : int, optional
        Parallelism knobs for CPU (ignored on GPU).  If omitted, CTranslate2
        picks sensible defaults (inter = n_physical_cores // 2).
    """
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if device is None or device == "auto":
        device = _detect_device()

    translator = ct2.Translator(
        str(model_dir),
        device=device,
        device_index=0,
        inter_threads=inter_threads or 0,  # 0 lets CT2 choose
        intra_threads=intra_threads or 0,
        compute_type="int8",  # the model is already quantised
    )
    translator.beam_size = beam

    # We do *not* download the HF weights; AutoTokenizer only needs the vocab
    # files (20 MB).  They are cached in ~/.cache/huggingface.
    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_TOKENIZER_REPO)

    return translator, tokenizer


if __name__ == "__main__":  # rudimentary self‑test
    import argparse, textwrap, sys

    parser = argparse.ArgumentParser(description="Quick smoke‑test for load_translator")
    parser.add_argument("model_dir", help="Path to the CTranslate2 model")
    args = parser.parse_args()

    try:
        tr, tok = load_translator(args.model_dir)
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    print("✓ Translator loaded (device:", tr.device, ") – vocab size:", len(tok), ")")
    print(textwrap.dedent("""
        You can now import this function from your scripts:
            from utils.mt import load_translator
            translator, tokenizer = load_translator("/path/to/model")
    """))
