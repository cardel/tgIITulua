#!/usr/bin/env python3
"""
evaluate.py — Compare LLM verdicts against the expert gold set.

Inputs:
    papers/tg-roger-joshua/verdicts/<model_tag>/*.json
        — classify.py output (one JSON per project_id, segregated per model).
        If no --model is given, the script picks the single subdirectory
        that exists; if multiple are present the caller must disambiguate.
    sources/tgIITulua/processed-data/datosTG.xlsx
        — gold: the `CATEGORIA REVISA` column holds the expert-assigned
          category for the subset that was manually reviewed.

What it does:
    1. Joins verdicts with gold by project_id.
    2. Normalizes gold labels to the three primary categories
       (Information Systems / Computer Science / Gamification).
    3. Reports accuracy, macro-precision/recall/F1, and a confusion matrix.
    4. Writes:
        - papers/tg-roger-joshua/sections/_auto/metrics.tex
          (small booktabs table for the paper)
        - papers/tg-roger-joshua/sections/_auto/metrics.json
          (raw numbers, consumed by render_confusion.py)

USAGE
    bash containers/run.sh paper-data \\
        python papers/tg-roger-joshua/scripts/evaluate.py

    # specific model when several have been run
    python papers/tg-roger-joshua/scripts/evaluate.py --model qwen2.5:14b-instruct

EXIT CODES
    0 = metrics produced
    2 = fatal error (missing inputs)
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PAPER = Path(__file__).resolve().parent.parent
VERDICTS_DIR = PAPER / "verdicts"
DATOS_XLSX = PAPER / "sources" / "tgIITulua" / "processed-data" / "datosTG.xlsx"
OUT_DIR = PAPER / "sections" / "_auto"
METRICS_TEX = OUT_DIR / "metrics.tex"
METRICS_JSON = OUT_DIR / "metrics.json"

# 23-term IEEE Thesaurus curated list (mirrors prompts/classify_v2.md and
# schemas/verdict.schema.json). Order is used for confusion-matrix rendering.
PRIMARY_LABELS = [
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


# Canonical normalization: map every observed variant of CATEGORIA REVISA to
# one of PRIMARY_LABELS (IEEE Thesaurus term). Unknown variants return None
# and are reported as unmapped for the caller to decide what to do.
GOLD_NORMALIZATION = {
    # Information systems family
    "information systems": "Information systems",
    "information system": "Information systems",
    "sistemas de informacion": "Information systems",
    "sistemas de información": "Information systems",
    "sistema de informacion": "Information systems",
    "sistema de información": "Information systems",
    "si": "Information systems",
    "bases de datos": "Database management systems",
    "base de datos": "Database management systems",
    "desarrollo de software": "Software engineering",
    # Artificial intelligence family
    "inteligencia artificial": "Artificial intelligence",
    "inteligencia arificial": "Artificial intelligence",  # typo in datosTG
    "sistemas de recomendacion": "Recommender systems",
    "sistemas de recomendación": "Recommender systems",
    "procesamiento digital de imagenes": "Image processing",
    "procesamiento digital de imágenes": "Image processing",
    "procesamiento de lenguaje natural": "Natural language processing",
    "mineria de datos": "Data mining",
    "minería de datos": "Data mining",
    "ontologias": "Ontologies",
    "ontologías": "Ontologies",
    "agentes de software": "Multi-agent systems",
    "seguridad": "Cybersecurity",
    "programacion por restricciones": "Constraint programming",
    "programación por restricciones": "Constraint programming",
    "redes complejas": "Complex networks",
    # Gamification / serious games / e-learning family
    "gamification": "Gamification",
    "gamificacion": "Gamification",
    "gamificación": "Gamification",
    "serious games": "Serious games",
    "juego serio": "Serious games",
    "juegos serios": "Serious games",
    "objeto virtual de aprendizaje": "E-learning",
    "objetos virtuales de aprendizaje": "E-learning",
    "ova": "E-learning",
}


def normalize_gold(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    return GOLD_NORMALIZATION.get(key)


def resolve_model_dir(model_tag: str | None) -> Path:
    if not VERDICTS_DIR.exists():
        raise FileNotFoundError(f"missing {VERDICTS_DIR} (run classify.py first)")
    if model_tag:
        safe = model_tag.replace(":", "_").replace("/", "_")
        candidate = VERDICTS_DIR / safe
        if not candidate.exists():
            raise FileNotFoundError(
                f"no verdicts for model {model_tag!r} under {candidate}"
            )
        return candidate
    subdirs = [p for p in VERDICTS_DIR.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    if len(subdirs) == 0:
        raise FileNotFoundError(
            f"{VERDICTS_DIR} contains no per-model subdirectories; "
            f"run classify.py first."
        )
    names = ", ".join(sorted(p.name for p in subdirs))
    raise RuntimeError(
        f"multiple models found under {VERDICTS_DIR}: {names}. "
        f"Pass --model <tag> to choose one."
    )


def load_verdicts(model_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(model_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: could not parse {p.name}: {e}", file=sys.stderr)
            continue
        rows.append({
            "id": data.get("project_id"),
            "llm_primary": data.get("verdict", {}).get("primary_term"),
            "llm_confidence": data.get("verdict", {}).get("confidence"),
            "input_source": data.get("input_source"),
            "model_id": data.get("model_id"),
        })
    return pd.DataFrame(rows)


def load_gold() -> pd.DataFrame:
    if not DATOS_XLSX.exists():
        raise FileNotFoundError(f"missing {DATOS_XLSX}")
    df = pd.read_excel(DATOS_XLSX)
    df.columns = [c.strip() for c in df.columns]
    if "CATEGORIA REVISA" not in df.columns:
        raise KeyError("datosTG.xlsx is missing column 'CATEGORIA REVISA'")
    df = df[["ID", "CATEGORIA REVISA"]].rename(
        columns={"ID": "id", "CATEGORIA REVISA": "gold_raw"}
    )
    df["id"] = df["id"].astype(str).str.strip()
    df["gold_primary"] = df["gold_raw"].apply(normalize_gold)
    return df


def compute_metrics(verdicts: pd.DataFrame, gold: pd.DataFrame,
                    *, verbose: bool = False) -> dict:
    """Join verdicts with gold and compute accuracy / macro-PRF1 / confusion.

    Raises ValueError when the intersection of valid verdicts and gold is empty.
    """
    joined = verdicts.merge(gold, on="id", how="left")
    unmapped_gold = joined[joined["gold_raw"].notna() & joined["gold_primary"].isna()]
    if verbose and not unmapped_gold.empty:
        samples = Counter(unmapped_gold["gold_raw"]).most_common(10)
        print("WARNING: unmapped gold labels (add to GOLD_NORMALIZATION):",
              file=sys.stderr)
        for label, n in samples:
            print(f"  - {label!r}  ({n})", file=sys.stderr)

    eval_set = joined.dropna(subset=["gold_primary", "llm_primary"]).copy()
    eval_set = eval_set[eval_set["gold_primary"].isin(PRIMARY_LABELS)]
    eval_set = eval_set[eval_set["llm_primary"].isin(PRIMARY_LABELS)]

    if eval_set.empty:
        raise ValueError("no overlap between gold set and LLM verdicts.")

    y_true = eval_set["gold_primary"].tolist()
    y_pred = eval_set["llm_primary"].tolist()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=PRIMARY_LABELS, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_n = precision_recall_fscore_support(
        y_true, y_pred, labels=PRIMARY_LABELS, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=PRIMARY_LABELS).tolist()
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=PRIMARY_LABELS))

    # Bootstrap 95% CI on macro-F1: resample (y_true, y_pred) pairs B times.
    rng = np.random.default_rng(0)
    n = len(y_true)
    f1_boot = np.empty(1000, dtype=float)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    for b in range(1000):
        idx = rng.integers(0, n, size=n)
        f1_boot[b] = f1_score(
            yt[idx], yp[idx],
            labels=PRIMARY_LABELS, average="macro", zero_division=0,
        )
    f1_lo, f1_hi = np.percentile(f1_boot, [2.5, 97.5])

    return {
        "n": len(eval_set),
        "labels": PRIMARY_LABELS,
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "macro_f1_ci_lower": float(f1_lo),
        "macro_f1_ci_upper": float(f1_hi),
        "cohen_kappa": kappa,
        "per_class": {
            label: {
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "f1": float(per_class_f1[i]),
                "support": int(per_class_n[i]),
            }
            for i, label in enumerate(PRIMARY_LABELS)
        },
        "confusion_matrix": cm,
        "n_verdicts": len(verdicts),
        "n_gold": int(gold["gold_primary"].notna().sum()),
        "n_unmapped_gold": int(len(unmapped_gold)),
    }


def render_metrics_tex(metrics: dict) -> str:
    rows = []
    rows.append(f"    Accuracy        & {metrics['accuracy']*100:.1f}\\,\\% \\\\")
    rows.append(f"    Macro precision & {metrics['macro_precision']*100:.1f}\\,\\% \\\\")
    rows.append(f"    Macro recall    & {metrics['macro_recall']*100:.1f}\\,\\% \\\\")
    rows.append(
        f"    Macro F1 (95\\% CI) & "
        f"{metrics['macro_f1']*100:.1f}\\,\\% "
        f"[{metrics['macro_f1_ci_lower']*100:.1f}, "
        f"{metrics['macro_f1_ci_upper']*100:.1f}] \\\\"
    )
    rows.append(f"    Cohen's $\\kappa$  & {metrics['cohen_kappa']:.3f} \\\\")
    body = "\n".join(rows)
    return f"""% Auto-generated by scripts/evaluate.py — do not edit.
\\begin{{table}}[t]
  \\centering
  \\caption{{LLM classifier performance on the expert gold set ({metrics['n']} projects).}}
  \\label{{tab:llm-metrics}}
  \\begin{{tabular}}{{lr}}
    \\toprule
    Metric & Value \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="Ollama model tag to evaluate; inferred when only one exists.")
    args = ap.parse_args()

    try:
        model_dir = resolve_model_dir(args.model)
        verdicts = load_verdicts(model_dir)
        gold = load_gold()
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Evaluating verdicts from {model_dir.relative_to(PAPER.parent.parent)}")

    if verdicts.empty:
        print("ERROR: no verdict JSON files found. Run classify.py first.",
              file=sys.stderr)
        return 2

    try:
        metrics = compute_metrics(verdicts, gold, verbose=True)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    METRICS_TEX.write_text(render_metrics_tex(metrics), encoding="utf-8")

    print(f"Evaluated on N={metrics['n']} projects "
          f"(gold labeled: {metrics['n_gold']}, unmapped: {metrics['n_unmapped_gold']})")
    print(f"  Accuracy        : {metrics['accuracy']*100:.1f}%")
    print(f"  Macro precision : {metrics['macro_precision']*100:.1f}%")
    print(f"  Macro recall    : {metrics['macro_recall']*100:.1f}%")
    print(f"  Macro F1        : {metrics['macro_f1']*100:.1f}%")
    print()
    # Per-class breakdown, restricted to labels with non-zero support in the
    # gold set (with 23 IEEE terms most classes are empty on the 111-project
    # gold subset, so printing the full matrix would be noise).
    print("Per-class (support > 0):")
    print(f"  {'Label':32} {'P':>6} {'R':>6} {'F1':>6} {'N':>4}")
    for label, stats in metrics["per_class"].items():
        if stats["support"] == 0:
            continue
        print(f"  {label[:32]:32} {stats['precision']*100:6.1f} "
              f"{stats['recall']*100:6.1f} {stats['f1']*100:6.1f} "
              f"{stats['support']:>4}")
    print()
    print(f"Wrote {METRICS_TEX.relative_to(PAPER.parent.parent)}")
    print(f"Wrote {METRICS_JSON.relative_to(PAPER.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
