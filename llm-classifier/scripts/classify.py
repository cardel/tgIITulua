#!/usr/bin/env python3
"""
classify.py — Local-LLM classifier for the degree-project corpus.

Reads metadata from tgIITulua/processed-data/datosTG.xlsx (via the
sources/tgIITulua symlink) by default, or from a pre-built corpus
file when --input-xlsx is supplied (see build_corpus.py).
Full-text .txt files are read from tgIITulua/extracted-text/ when
available.

OUTPUT
    papers/tg-roger-joshua/verdicts/<project_id>.json
        — validates against scripts/schemas/verdict.schema.json
    .cache/classify-cache.json
        — keyed by input_hash + model_tag so re-runs are cheap.

REQUIREMENTS
    - Ollama running locally (default: http://127.0.0.1:11434).
    - The selected model pulled once, e.g.:
        ollama pull qwen2.5:14b-instruct
        ollama pull llama3.1:8b
        ollama pull mistral:7b-instruct

USAGE
    # gold set only (default)
    bash containers/run.sh paper-data \\
        python papers/tg-roger-joshua/scripts/classify.py

    # full corpus (datosTG + Seguimiento 2019-2025)
    bash containers/run.sh paper-data \\
        python papers/tg-roger-joshua/scripts/classify.py \\
        --input-xlsx papers/tg-roger-joshua/sections/_auto/corpus.xlsx

    # specific model
    python papers/tg-roger-joshua/scripts/classify.py --model llama3.1:8b

    # only a subset
    python papers/tg-roger-joshua/scripts/classify.py --only 2018A3 2019B1

    # force re-classification of existing verdicts
    python papers/tg-roger-joshua/scripts/classify.py --force

EXIT CODES
    0 = every project produced a valid verdict (or was skipped when cached)
    1 = at least one project failed
    2 = fatal error (missing deps / Ollama unreachable / inputs)
"""

from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib import request as _urlreq
from urllib import error as _urlerr

try:
    import pandas as pd
except ImportError as e:
    print(f"ERROR: missing dep ({e}).", file=sys.stderr)
    sys.exit(2)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

PAPER = Path(__file__).resolve().parent.parent
SCRIPTS = PAPER / "scripts"
PROMPT_FILE = SCRIPTS / "prompts" / "classify_v2.md"
SCHEMA_FILE = SCRIPTS / "schemas" / "verdict.schema.json"
VERDICTS_DIR = PAPER / "verdicts"

TGII = PAPER / "sources" / "tgIITulua"
DATOS_XLSX = TGII / "processed-data" / "datosTG.xlsx"
TEXTS_DIR = TGII / "extracted-text"

CACHE_FILE = Path(".cache") / "classify-cache.json"

PROMPT_VERSION = "v2"
DEFAULT_MODEL = "qwen2.5:14b-instruct"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
NUM_PREDICT = 800
MAX_FULLTEXT_CHARS = 8000
REQUEST_TIMEOUT = 600  # seconds; per-chunk timeout in streaming mode

# 23-term IEEE Thesaurus curated list; kept in sync with prompts/classify_v2.md
# and schemas/verdict.schema.json.
ALLOWED_TERMS = {
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
}


# ----------------------------------------------------------------------
# Metadata loading
# ----------------------------------------------------------------------

def load_metadata(input_xlsx: Path | None = None) -> pd.DataFrame:
    """Load project metadata.

    If *input_xlsx* is given (corpus built by build_corpus.py) its
    columns are already normalised (id, title, keywords, abstract …).
    Otherwise the raw datosTG.xlsx format is expected.
    """
    src = input_xlsx if input_xlsx is not None else DATOS_XLSX
    if not src.exists():
        raise FileNotFoundError(f"missing {src}")
    df = pd.read_excel(src)
    df.columns = [c.strip() for c in df.columns]

    if input_xlsx is not None:
        # corpus.xlsx is already normalised
        for col in ("title", "keywords", "abstract"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        df = df.dropna(subset=["id"]).reset_index(drop=True)
        df["id"] = df["id"].astype(str).str.strip()
    else:
        wanted = {
            "ID": "id",
            "NOMBRE DEL TRABAJO": "title",
            "PALABRAS CLAVE": "keywords",
            "RESUMEN": "abstract",
            "CATEGORÍA": "manual_category",
            "CATEGORIA REVISA": "reviewer_category",
        }
        present = {k: v for k, v in wanted.items() if k in df.columns}
        df = df[list(present)].rename(columns=present)
        df = df.dropna(subset=["id"]).reset_index(drop=True)
        df["id"] = df["id"].astype(str).str.strip()
        for col in ("title", "keywords", "abstract"):
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
    return df


def load_fulltext(project_id: str) -> str | None:
    p = TEXTS_DIR / f"{project_id}.txt"
    if not p.exists():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    return text[:MAX_FULLTEXT_CHARS]


# ----------------------------------------------------------------------
# Prompt assembly and hashing
# ----------------------------------------------------------------------

def build_input_blob(row: pd.Series, fulltext: str | None) -> str:
    parts = [
        f"project_id: {row['id']}",
        f"title: {row.get('title', '')}",
        f"keywords: {row.get('keywords', '')}",
        f"abstract: {row.get('abstract', '')}",
    ]
    if fulltext:
        parts.append(f"full_text_excerpt:\n{fulltext}")
    else:
        parts.append("full_text_excerpt: (not available)")
    return "\n".join(parts)


def hash_input(project_id: str, blob: str, model_tag: str) -> str:
    h = hashlib.sha256()
    for chunk in (project_id, PROMPT_VERSION, model_tag, blob):
        h.update(chunk.encode("utf-8"))
        h.update(b"||")
    return h.hexdigest()[:24]


# ----------------------------------------------------------------------
# Ollama call
# ----------------------------------------------------------------------

def load_prompt_template() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"missing {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def ollama_health(base_url: str) -> None:
    try:
        with _urlreq.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"{base_url}: HTTP {resp.status}")
    except (_urlerr.URLError, OSError) as e:
        raise RuntimeError(
            f"cannot reach Ollama at {base_url} ({e}). "
            f"Start it with `ollama serve`."
        ) from e


def call_ollama(base_url: str, model: str,
                system_prompt: str, user_blob: str) -> dict:
    """Call the Ollama /api/chat endpoint and return the parsed JSON verdict.

    Uses stream=True so REQUEST_TIMEOUT applies per-chunk (not total response),
    which prevents false timeouts during slow CPU-only inference.
    """
    payload = {
        "model": model,
        "stream": True,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": NUM_PREDICT,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_blob},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = _urlreq.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    content = ""
    with _urlreq.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            content += (chunk.get("message") or {}).get("content", "")
            if chunk.get("done"):
                break
    raw = content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def validate_verdict(v: dict) -> list[str]:
    errs = []
    # Some local LLMs emit "Term | null" or "Term, null" in place of a clean
    # term. Strip the trailing null marker before validating.
    for key in ("primary_term", "secondary_term"):
        val = v.get(key)
        if isinstance(val, str):
            stripped = re.sub(r"\s*[|,/;]\s*null\s*$", "", val, flags=re.IGNORECASE).strip()
            if stripped.lower() in ("", "null", "none"):
                v[key] = None if key == "secondary_term" else val
            else:
                v[key] = stripped
    primary = v.get("primary_term")
    if primary not in ALLOWED_TERMS:
        errs.append(f"primary_term not in allowed set: {primary!r}")
    secondary = v.get("secondary_term")
    if secondary is not None and secondary not in ALLOWED_TERMS:
        # Drop out-of-vocabulary secondaries instead of failing the verdict;
        # the primary classification is what feeds the aggregate analyses.
        v["secondary_term"] = None
        v.setdefault("notes", {})
        if isinstance(v.get("notes"), dict):
            v["notes"]["secondary_dropped"] = secondary
    conf = v.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        errs.append(f"confidence out of range: {conf!r}")
    justification = v.get("justification", "")
    if not isinstance(justification, str):
        v["justification"] = ""
    elif not justification.strip():
        v["justification"] = ""
    return errs


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                          encoding="utf-8")


# ----------------------------------------------------------------------
# Output layout
# ----------------------------------------------------------------------

def verdict_path(model_tag: str, project_id: str) -> Path:
    # Verdicts are segregated per model so a multi-model comparison does not
    # overwrite earlier runs. `qwen2.5:14b-instruct` -> `qwen2.5_14b-instruct`.
    safe = model_tag.replace(":", "_").replace("/", "_")
    return VERDICTS_DIR / safe / f"{project_id}.json"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=[],
                    help="Only process these project IDs (space-separated)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA,
                    help=f"Ollama base URL (default: {DEFAULT_OLLAMA})")
    ap.add_argument("--force", action="store_true",
                    help="Re-classify even if verdict JSON already exists")
    ap.add_argument("--no-cache", action="store_true",
                    help="Bypass the in-memory cache (still writes verdicts to disk)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build inputs and hashes, but skip model calls")
    ap.add_argument("--input-xlsx", default=None, metavar="PATH",
                    help="Pre-built corpus xlsx (from build_corpus.py). "
                         "Default: tgIITulua/processed-data/datosTG.xlsx")
    ap.add_argument("--metadata-only", action="store_true",
                    help="Skip loading extracted full text; classify on "
                         "title+keywords+abstract only (uniform inputs).")
    args = ap.parse_args()

    input_xlsx = Path(args.input_xlsx) if args.input_xlsx else None
    try:
        meta = load_metadata(input_xlsx)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.only:
        meta = meta[meta["id"].isin(args.only)]
        if meta.empty:
            print(f"ERROR: none of --only {args.only} match the ID column",
                  file=sys.stderr)
            return 2

    system_prompt = load_prompt_template()
    cache = load_cache()
    use_cache = not args.no_cache

    if not args.dry_run:
        try:
            ollama_health(args.ollama_url)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    failed: list[str] = []
    ok = 0
    skipped = 0
    print(f"Model: {args.model}  |  Ollama: {args.ollama_url}")
    for i, row in meta.iterrows():
        pid = row["id"]
        out_path = verdict_path(args.model, pid)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fulltext = None if args.metadata_only else load_fulltext(pid)
        blob = build_input_blob(row, fulltext)
        h = hash_input(pid, blob, args.model)

        if out_path.exists() and not args.force:
            print(f"  [{i+1:>3}] SKIP  {pid} (verdict exists)")
            skipped += 1
            continue

        if use_cache and h in cache and not args.force:
            verdict = cache[h]
            tag = " (cache)"
        elif args.dry_run:
            print(f"  [{i+1:>3}] DRY   {pid}  hash={h}  src="
                  f"{'full' if fulltext else 'meta'}")
            continue
        else:
            try:
                verdict = call_ollama(args.ollama_url, args.model,
                                       system_prompt, blob)
            except (json.JSONDecodeError, _urlerr.URLError, OSError) as e:
                print(f"  [{i+1:>3}] FAIL  {pid}: {e}", file=sys.stderr)
                failed.append(pid)
                continue
            cache[h] = verdict
            tag = ""
            if (i + 1) % 10 == 0:
                save_cache(cache)

        errs = validate_verdict(verdict)
        if errs:
            print(f"  [{i+1:>3}] INVALID {pid}: {'; '.join(errs)}", file=sys.stderr)
            failed.append(pid)
            continue

        record = {
            "project_id": pid,
            "model_id": args.model,
            "prompt_version": PROMPT_VERSION,
            "input_hash": h,
            "input_source": "full_text" if fulltext else "metadata_only",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "verdict": verdict,
        }
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"  [{i+1:>3}] OK    {pid} -> {verdict['primary_term']}{tag}")
        ok += 1

    if not args.dry_run:
        save_cache(cache)

    print()
    print("=" * 60)
    print(f"  OK:      {ok}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {len(failed)}")
    print("=" * 60)
    if failed:
        for pid in failed:
            print(f"  - {pid}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
