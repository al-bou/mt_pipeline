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


def _detect_device() -> str:
    """Return 'cuda' if CUDA is available for CTranslate2, else 'cpu'."""
    try:
        devices = ct2.available_devices()
    except AttributeError:
        try:
            devices = ct2.Device.get_supported_devices()
        except Exception:
            return "cpu"
    return "cuda" if "cuda" in devices else "cpu"


def load_translator(
    model_dir: str | Path,
    device: Optional[str] = None,
    inter_threads: Optional[int] = None,
    intra_threads: Optional[int] = None,
) -> Tuple[ct2.Translator, AutoTokenizer]:
    """Load a CTranslate2 translator and a HF tokenizer.

    Parameters
    ----------
    model_dir: Path to CTranslate2 model directory
    device: 'cpu', 'cuda', or None for auto
    inter_threads, intra_threads: CPU threading config

    Returns
    -------
    (translator, tokenizer)
    """
    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # List supported devices for debugging
    try:
        supported = ct2.available_devices()
    except AttributeError:
        supported = ct2.Device.get_supported_devices() if hasattr(ct2, 'Device') else []
    print(f"[mt] Supported devices: {supported}")

    device_choice = device if device else ("cuda" if "cuda" in supported else "cpu")
    print(f"[mt] Loading translator on device: {device_choice}")

    translator = ct2.Translator(
        str(model_path),
        device=device_choice,
        device_index=0,
        inter_threads=inter_threads or 0,
        intra_threads=intra_threads or 0,
        compute_type="int8",
    )

    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_TOKENIZER_REPO)
    print(f"[mt] Tokenizer loaded, vocab size: {len(tokenizer)}")

    return translator, tokenizer


def translate_batch(
    sentences: List[str],
    translator: ct2.Translator,
    tokenizer: AutoTokenizer,
    beam: int = 5,
) -> List[str]:
    """Translate a list of sentences/paragraphs using given beam size.

    - Tokenize sentences into token strings
    - Call translator.translate_batch per segment for granular logging
    - Decode output tokens to string
    - Logs progress and timing
    """
    num = len(sentences)
    print(f"[mt] Starting granular translation of {num} segments with beam={beam}...")
    t_total_start = time.time()
    translations: List[str] = []

    # Tokenize once
    tokenized: List[List[str]] = [tokenizer.tokenize(p) for p in sentences]

    for idx, tokens in enumerate(tokenized, start=1):
        print(f"[mt] Translating segment {idx}/{num} ({len(tokens)} tokens)")
        t0 = time.time()
        # NLLB requires target_prefix token to indicate language
        prefix = [["<2spa_Latn>"]]
        # translate single segment
        result = translator.translate_batch(
            [tokens],
            target_prefix=prefix,
            beam_size=beam,
        )
        duration = time.time() - t0
        # Decode top hypothesis
        out_tokens = result[0].hypotheses[0]
        text = tokenizer.convert_tokens_to_string(out_tokens)
        translations.append(text)
        print(f"[mt] Segment {idx} done in {duration:.2f}s")

    print(f"[mt] All segments translated in {time.time() - t_total_start:.2f}s")
    return translations

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Test loading translator and translating a sample.")
    parser.add_argument("model_dir", help="Path to CTranslate2 model")
    parser.add_argument("--beam", type=int, default=5, help="Beam size for translation batch.")
    args = parser.parse_args()

    try:
        tr, tok = load_translator(args.model_dir)
    except Exception as e:
        sys.exit(f"Error loading translator: {e}")

    sample = ["Bonjour le monde.", "Comment ça va ?"]
    out = translate_batch(sample, tr, tok, beam=args.beam)
    print(f"Translations: {out}")
