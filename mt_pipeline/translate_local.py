# mt_pipeline/translate_local.py
"""
Traduit tous les fichiers .md de data/01_chunks_fr
et écrit le résultat dans data/02_raw_es.
"""

import argparse
import pathlib
import yaml
from utils.mt import load_translator, translate_file
from utils.io import iter_chunk_files, write_outfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config/settings.yaml",
        help="Chemin vers le YAML de configuration."
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    src_dir  = pathlib.Path("data/01_chunks_fr")
    tgt_dir  = pathlib.Path("data/02_raw_es")
    tgt_dir.mkdir(parents=True, exist_ok=True)

    translator = load_translator(cfg["model_dir"], beam=cfg["beam_size"])

    for fp in iter_chunk_files(src_dir):
        out_path = tgt_dir / fp.name
        if out_path.exists():
            print(f"◦ Skip {fp.name} (déjà traduit)")
            continue
        print(f"→ {fp.name}")
        es_text = translate_file(fp, translator, src_lang="fra_Latn", tgt_lang="spa_Latn")
        write_outfile(out_path, es_text)

if __name__ == "__main__":
    main()
