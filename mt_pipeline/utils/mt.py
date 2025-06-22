# utils/mt.py
"""Utility helpers around CTranslate2 + NLLB.

Provides:
    - load_translator: load CTranslate2 translator and HF tokenizer.
    - translate_batch: wrapper for batch translation using tokenizer + translator.

Usage:
    from mt_pipeline.utils.mt import load_translator, translate_batch

Functions assume you have converted an NLLB-200 model to CTranslate2 format.
"""

from pathlib import Path
from typing import Tuple, List, Optional

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

    device_choice = device if device else _detect_device()

    translator = ct2.Translator(
        str(model_path),
        device=device_choice,
        device_index=0,
        inter_threads=inter_threads or 0,
        intra_threads=intra_threads or 0,
        compute_type="int8",
    )

    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_TOKENIZER_REPO)

    return translator, tokenizer


def translate_batch(
    sentences: List[str],
    translator: ct2.Translator,
    tokenizer: AutoTokenizer,
    beam: int = 5,
) -> List[str]:
    """Translate a list of sentences/paragraphs using given beam size.

    - Tokenize sentences into token strings
    - Call translator.translate_batch with beam_size
    - Decode output tokens to string
    """
    # Tokenize into string tokens
    tokenized: List[List[str]] = [tokenizer.tokenize(p) for p in sentences]

    # Translate with beam
    results = translator.translate_batch(
        tokenized,
        beam_size=beam,
    )

    # Decode top hypotheses: list of token strings -> string
    translations: List[str] = []
    for res in results:
        tokens = res.hypotheses[0]
        # Convert token list back to text
        text = tokenizer.convert_tokens_to_string(tokens)
        translations.append(text)

    return translations


if __name__ == "__main__":
    import argparse, sys, time

    parser = argparse.ArgumentParser(description="Test loading translator and translating a sample.")
    parser.add_argument("model_dir", help="Path to CTranslate2 model")
    parser.add_argument("--beam", type=int, default=5, help="Beam size for translation batch.")
    args = parser.parse_args()

    try:
        tr, tok = load_translator(args.model_dir)
    except Exception as e:
        sys.exit(f"Error loading translator: {e}")

    print(f"✓ Translator loaded (device: {tr.device}) – tokenizer vocab size: {len(tok)}")
    sample = ["Bonjour le monde.", "Comment ça va ?"]
    start = time.time()
    out = translate_batch(sample, tr, tok, beam=args.beam)
    print(f"Translations: {out}")
    print(f"Time: {time.time() - start:.2f}s")
