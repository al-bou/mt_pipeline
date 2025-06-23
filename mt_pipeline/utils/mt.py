# utils/mt.py
"""Utility helpers around CTranslate2 + NLLB‑200.

Run as a module  (Python ≥3.8):

    python -m mt_pipeline.utils.mt <MODEL_DIR> [--beam 5] [--device cuda] \
        [--text "Bonjour" "Une autre phrase"]

• If *--text* is omitted, the module reads one sentence per line from STDIN.
• The script prints the translated sentences to STDOUT in the same order.

This file also exposes the helper functions ``load_translator`` and
``translate_batch`` for programmatic use.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import ctranslate2 as ct2
from transformers import AutoTokenizer

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
_DEFAULT_TOKENIZER_REPO = "facebook/nllb-200-3.3B"
_SUPPORTED_SOURCES: Tuple[str, ...] = ("fra_Latn",)
_SUPPORTED_TARGETS: Tuple[str, ...] = ("spa_Latn",)

# ----------------------------------------------------------------------------
# Device helpers
# ----------------------------------------------------------------------------

def _get_devices() -> List[str]:
    """Return a list like ``['cpu', 'cuda']`` depending on installed CT2."""
    if hasattr(ct2, "available_devices"):
        return ct2.available_devices()
    if hasattr(ct2, "list_devices"):
        return ct2.list_devices()
    if hasattr(ct2, "get_supported_devices"):
        return ct2.get_supported_devices()
    if hasattr(ct2, "Device") and hasattr(ct2.Device, "list_devices"):
        return ct2.Device.list_devices()
    if hasattr(ct2, "has_cuda") and ct2.has_cuda():
        return ["cpu", "cuda"]
    return ["cpu"]


def _detect_device() -> str:
    return "cuda" if "cuda" in _get_devices() else "cpu"

# ----------------------------------------------------------------------------
# Translator / Tokenizer loaders
# ----------------------------------------------------------------------------

def load_translator(model_dir: Path | str, device: str | None = None, *, int8: bool = True) -> Tuple[ct2.Translator, "AutoTokenizer"]:
    """Load a CTranslate2 Translator + HF tokenizer."""
    device = device or _detect_device()
    translator = ct2.Translator(str(model_dir), device=device, compute_type="int8" if int8 else "auto")
    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_TOKENIZER_REPO)
    return translator, tokenizer

# ----------------------------------------------------------------------------
# Translation wrapper
# ----------------------------------------------------------------------------

def translate_batch(
    sentences: List[str],
    translator: ct2.Translator,
    tokenizer,
    *,
    beam: int = 5,
    src_lang: str = _SUPPORTED_SOURCES[0],
    tgt_lang: str = _SUPPORTED_TARGETS[0],
) -> List[str]:
    """Translate *sentences* using NLLB and CTranslate2."""

    # Tokenise (prepend language tokens)
    batch_tokens = [tokenizer.convert_ids_to_tokens([tokenizer.lang_code_to_id[src_lang]]) + tokenizer.tokenize(s) for s in sentences]
    translations = translator.translate_batch(
        batch_tokens,
        beam_size=beam,
        target_prefix=[[f"<{tgt_lang}>"] for _ in sentences],
    )
    outputs: List[str] = []
    for t in translations:
        # detokenise, skip language tag
        ids = tokenizer.convert_tokens_to_ids(t.hypotheses[0])
        text = tokenizer.decode(ids, skip_special_tokens=True)
        outputs.append(text)
    return outputs

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _cli():
    p = argparse.ArgumentParser("Translate sentences with NLLB via CTranslate2")
    p.add_argument("model_dir", type=Path, help="Path to the CTranslate2-converted model directory")
    p.add_argument("--beam", type=int, default=5, help="Beam size (default: 5)")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Force device (default: auto-detect)")

    # one or many sentences provided on the command line
    p.add_argument("--text", nargs="*", help="Sentences to translate; if omitted, read from stdin")

    args = p.parse_args()

    translator, tokenizer = load_translator(args.model_dir, args.device)
    sentences = args.text if args.text else [line.strip() for line in sys.stdin if line.strip()]

    if not sentences:
        print("[mt] No input sentences provided", file=sys.stderr)
        sys.exit(1)

    start = time.perf_counter()
    outputs = translate_batch(sentences, translator, tokenizer, beam=args.beam)
    elapsed = time.perf_counter() - start

    for t in outputs:
        print(t)

    print(f"[mt] Done ({len(sentences)} segments, {elapsed:.2f}s)", file=sys.stderr)


if __name__ == "__main__":
    _cli()
