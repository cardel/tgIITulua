#!/usr/bin/env python3
"""
generate_results.py — Consolidate LLM verdicts into a results Excel file
following the same tabular pattern as tgIITulua/final_results.xlsx.

Reads
-----
    papers/tg-roger-joshua/sections/_auto/corpus.xlsx
        Unified corpus: id, title, keywords, abstract, source, expert_category
    papers/tg-roger-joshua/verdicts/<model_slug>/<project_id>.json
        One JSON verdict per project produced by classify.py

Writes (one file per model found in verdicts/)
-----
    papers/tg-roger-joshua/sections/_auto/llm_results_<model_slug>.xlsx
        Mirrors the tgIITulua final_results.xlsx layout, replacing
        "NMF Topic" with LLM classification columns.

    papers/tg-roger-joshua/sections/_auto/llm_distribution_<model_slug>.xlsx
        Category distribution table (# projects, %) per model — equivalent
        to the Atlas.ti / NMF summary tables used in prior versions of the paper.

USAGE
    bash containers/run.sh paper-data \\
        python papers/tg-roger-joshua/scripts/generate_results.py

    # single model
    bash containers/run.sh paper-data \\
        python papers/tg-roger-joshua/scripts/generate_results.py \\
        --model qwen2.5:14b-instruct
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:
    print(f"ERROR: missing dep ({e}).", file=sys.stderr)
    sys.exit(2)

PAPER       = Path(__file__).resolve().parent.parent
VERDICTS    = PAPER / "verdicts"
CORPUS_XLSX = PAPER / "sections" / "_auto" / "corpus.xlsx"
OUT_DIR     = PAPER / "sections" / "_auto"

# 23-term IEEE Thesaurus curated list, kept in sync with evaluate.py and
# prompts/classify_v2.md. Used as the display order for distribution tables.
PRIMARY_ORDER = [
    "Information systems",
    "Database management systems",
    "Enterprise information systems",
    "Web applications",
    "Mobile applications",
    "Software engineering",
    "Artificial intelligence",
    "Machine learning",
    "Natural language processing",
    "Image processing",
    "Computer vision",
    "Recommender systems",
    "Data mining",
    "Ontologies",
    "Multi-agent systems",
    "Cybersecurity",
    "Constraint programming",
    "Complex networks",
    "Internet of Things",
    "Gamification",
    "Serious games",
    "E-learning",
    "Other",
]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def slug_to_model(slug: str) -> str:
    """Reverse the model-slug escaping used by classify.py."""
    return slug.replace("_", ":", 1)


def load_corpus() -> pd.DataFrame:
    df = pd.read_excel(CORPUS_XLSX)
    df["id"] = df["id"].astype(str).str.strip()
    return df.set_index("id")


def load_verdicts(model_dir: Path) -> dict[str, dict]:
    verdicts: dict[str, dict] = {}
    for p in model_dir.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            pid = rec.get("project_id") or p.stem
            verdicts[pid] = rec
        except Exception:
            pass
    return verdicts


# -----------------------------------------------------------------------
# Build result DataFrame  (mirrors tgIITulua/final_results.xlsx columns)
# -----------------------------------------------------------------------

def build_results(corpus: pd.DataFrame, verdicts: dict[str, dict],
                  model_tag: str) -> pd.DataFrame:
    rows = []
    for pid, meta in corpus.iterrows():
        rec = verdicts.get(pid, {})
        v = rec.get("verdict", {})
        rows.append({
            # --- tgIITulua-compatible columns -------------------------
            "ID":                    pid,
            "NOMBRE DEL TRABAJO":    meta.get("title", ""),
            "PALABRAS CLAVE":        meta.get("keywords", ""),
            "RESUMEN":               meta.get("abstract", ""),
            "input_source":          rec.get("input_source", ""),
            # --- LLM output (replaces NMF Topic) ----------------------
            "LLM_primary_term":      v.get("primary_term", ""),
            "LLM_secondary_term":    v.get("secondary_term", "") or "",
            "LLM_confidence":        v.get("confidence", ""),
            "LLM_justification":     v.get("justification", ""),
            "LLM_keywords_cited":    ", ".join(v.get("keywords_cited", [])),
            # --- Ground truth (gold set only) -------------------------
            "expert_category":       meta.get("expert_category", ""),
            # --- Provenance -------------------------------------------
            "source":                meta.get("source", ""),
            "model":                 model_tag,
            "generated_at":          rec.get("generated_at", ""),
        })
    return pd.DataFrame(rows)


def build_distribution(results: pd.DataFrame) -> pd.DataFrame:
    classified = results[results["LLM_primary_term"] != ""]
    total = len(classified)
    counts = (classified["LLM_primary_term"]
              .value_counts()
              .reindex(PRIMARY_ORDER, fill_value=0))
    dist = pd.DataFrame({
        "Category":   counts.index,
        "# Projects": counts.values,
        "Percentage": (counts.values / total * 100).round(1) if total else 0,
    })
    total_row = pd.DataFrame([{
        "Category": "Total",
        "# Projects": total,
        "Percentage": 100.0,
    }])
    return pd.concat([dist, total_row], ignore_index=True)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def process_model(model_dir: Path, corpus: pd.DataFrame) -> None:
    slug = model_dir.name
    model_tag = slug_to_model(slug)
    verdicts = load_verdicts(model_dir)

    if not verdicts:
        print(f"  {slug}: no verdicts found — skipping")
        return

    results = build_results(corpus, verdicts, model_tag)
    dist    = build_distribution(results)

    out_results = OUT_DIR / f"llm_results_{slug}.xlsx"
    out_dist    = OUT_DIR / f"llm_distribution_{slug}.xlsx"

    results.to_excel(out_results, index=False)
    dist.to_excel(out_dist, index=False)

    classified = (results["LLM_primary_term"] != "").sum()
    gold_ok    = ((results["expert_category"] != "") &
                  (results["LLM_primary_term"] != "")).sum()
    print(f"  {slug}: {classified}/{len(results)} classified "
          f"({gold_ok} with gold label)  → {out_results.name}")
    print(f"    distribution:")
    for _, r in dist.iterrows():
        print(f"      {r['Category']}: {r['# Projects']} ({r['Percentage']:.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="Limit to one model slug (e.g. qwen2.5:14b-instruct). "
                         "Default: all sub-dirs of verdicts/.")
    args = ap.parse_args()

    if not CORPUS_XLSX.exists():
        print(f"ERROR: {CORPUS_XLSX} not found — run build_corpus.py first",
              file=sys.stderr)
        return 2
    if not VERDICTS.exists():
        print(f"ERROR: {VERDICTS} not found — run classify.py first",
              file=sys.stderr)
        return 2

    corpus = load_corpus()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        slug = args.model.replace(":", "_").replace("/", "_")
        dirs = [VERDICTS / slug]
    else:
        dirs = [d for d in VERDICTS.iterdir()
                if d.is_dir() and not d.name.startswith("_")]

    if not dirs:
        print("No verdict directories found.", file=sys.stderr)
        return 2

    for d in sorted(dirs):
        if not d.exists():
            print(f"  WARNING: {d} not found", file=sys.stderr)
            continue
        process_model(d, corpus)

    return 0


if __name__ == "__main__":
    sys.exit(main())
