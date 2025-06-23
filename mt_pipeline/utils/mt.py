# utils/mt.py
"""Utility helpers around CTranslate2 + NLLB‑200.

Example (Python ≥3.8)
---------------------
Translate one or several sentences:

    python -m mt_pipeline.utils.mt <MODEL_DIR> \
        --device cuda --beam 5 \
        --text "Bonjour le monde" "Encore une phrase"

If *--text* is omitted, the script reads one sentence per line from STDIN.
The module also exposes ``load_translator`` and ``translate_batch`` for
programmatic use.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import ctranslate2 as ct2
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_TOKENIZER_REPO = "facebook/nllb-200-3.3B"
_SUPPORTED_SOURCES: Tuple[str, ...] = ("fra_Latn",)
_SUPPORTED_TARGETS: Tuple[str, ...] = ("spa_Latn",)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_devices() -> List[str]:
    """Return e.g. ``['cpu', 'cuda']`` depending on CT2 version."""
    if hasattr(ct2, "available_devices"):
        return ct2.available_devices()
    if hasattr(ct2, "list_devices"):
        return ct2.list_devices()
    if hasattr(ct2, "get_supported_devices"):
        return ct2.get_supported_devices()
    if hasattr(ct2, "Device") and hasattr(ct2.Device, "list_devices"):
        return ct2.Device.list_devices()
    # Fallback: parse ldd output for CUDA libs
    so = Path(ct2.__file__).with_suffix(".so")
    try:
        out = subprocess.check_output(["ldd", so], text=True)
        if re.search(r"libcu(blas|dart)", out):
            return ["cpu", "cuda"]
    except Exception:
        pass
    return ["cpu"]


def _detect_device() -> str:
    """Auto‑select CUDA when available."""
    return "cuda" if "cuda" in _get_devices() else "cpu"

# ---------------------------------------------------------------------------
# Translator / Tokenizer loaders
# ---------------------------------------------------------------------------

def load_translator(
    model_dir: Path | str,
    device: str | None = None,
    *,
    int8: bool = True,
):
    """Return a ``(Translator, Tokenizer)`` tuple ready to use."""
    device = device or _detect_device()
    translator = ct2.Translator(
        str(model_dir), device=device, compute_type="int8" if int8 else "auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_TOKENIZER_REPO)
    return translator, tokenizer

# ---------------------------------------------------------------------------
# Helpers for language tags
# ---------------------------------------------------------------------------

def _find_lang_tag(tok, code: str) -> str:
    """Return the special token corresponding to *code* in *tok*."""
    # 1) Modern tokenizer exposes mapping id -> token
    if hasattr(tok, "lang_code_to_id") and code in tok.lang_code_to_id:
        return tok.convert_ids_to_tokens(tok.lang_code_to_id[code])
    # 2) Search additional_special_tokens
    for t in getattr(tok, "additional_special_tokens", []):
        if code in t:
            return t
    # 3) Try common textual patterns
    for cand in (f"<<{code}>>", f"<{code}>", f"__{code}__"):
        if tok.convert_tokens_to_ids(cand) != tok.unk_token_id:
            return cand
    raise ValueError(f"Cannot find language tag for {code} in tokenizer")

# ---------------------------------------------------------------------------
# Translation wrapper
# ---------------------------------------------------------------------------

def translate_batch(
    sentences: List[str],
    translator: ct2.Translator,
    tokenizer,
    *,
    beam: int = 5,
    src_lang: str = _SUPPORTED_SOURCES[0],
    tgt_lang: str = _SUPPORTED_TARGETS[0],
) -> List[str]:
    """Translate *sentences* with proper BOS/src/tgt prompts.

    The encoder expects ``<s> <src_lang> tokens`` and the decoder is
    primed with ``<s> <tgt_lang>``.  Supplying BOS twice (or omitting it)
    can cause degenerate repetition; this helper enforces a correct
    prompt and protects with *no‑repeat‑ngram*.
    """

    # Resolve BOS token reliably
    if tokenizer.bos_token_id is not None:
        bos = tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)
    else:
        bos = "<s>"  # fallback textual token present in NLLB vocab

    src_tag = _find_lang_tag(tokenizer, src_lang)
    tgt_tag = _find_lang_tag(tokenizer, tgt_lang)

    batch_tokens = [[bos, src_tag] + tokenizer.tokenize(s) for s in sentences]

    results = translator.translate_batch(
        batch_tokens,
        beam_size=beam,
        target_prefix=[[tgt_tag] for _ in sentences],  # BOS implicit in decoder
        end_token="</s>",
        no_repeat_ngram_size=3,
        max_decoding_length=256,
    )

    outputs: List[str] = []
    for r in results:
        tokens = r.hypotheses[0]
        # Remove leading BOS/lang tags if present
        while tokens and tokens[0] in {bos, tgt_tag, src_tag}:
            tokens.pop(0)
        outputs.append(tokenizer.convert_tokens_to_string(tokens).strip())
    return outputs

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser("Translate with NLLB via CTranslate2")
    parser.add_argument("model_dir", type=Path, help="Path to CTranslate2 model")
    parser.add_argument("--beam", type=int, default=5, help="Beam size (default 5)")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force device")
    parser.add_argument("--text", nargs="*", help="Sentences to translate")
    args = parser.parse_args()

    translator, tokenizer = load_translator(args.model_dir, args.device)
    sentences = args.text if args.text else [l.strip() for l in sys.stdin if l.strip()]
    if not sentences:
        print("[mt] No input sentences", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    outs = translate_batch(sentences, translator, tokenizer, beam=args.beam)
    dt = time.perf_counter() - t0

    for line in outs:
        print(line)
    print(f"[mt] Done ({len(sentences)} segs, {dt:.2f}s)", file=sys.stderr)


if __name__ == "__main__":
    _cli()
