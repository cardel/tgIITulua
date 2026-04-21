#!/usr/bin/env python3
"""
extract_batch.py — Resumable batch OCR for the 2018–2024 corpus.

Walks a directory of PDFs, converts each to per-page JPEGs with pdf2image,
runs Tesseract (Spanish) via pyocr, concatenates the page text, and writes
<ID>.txt into the output directory. Skips PDFs whose .txt already exists.

This is a directory-aware, resumable version of the legacy ocrtest.py.

USAGE
    python3 extract_batch.py \\
        --pdfs raw-data/textos-2018-2024/ \\
        --out  extracted-text/ \\
        --dpi  500

    # resume after interruption — skips PDFs whose .txt is already there
    python3 extract_batch.py --pdfs raw-data/textos-2018-2024/ --out extracted-text/

    # process a single file (backwards-compatible with ocrtest.py)
    python3 extract_batch.py path/to/single.pdf --out extracted-text/

EXIT CODES
    0 = all PDFs processed or already present
    1 = at least one PDF failed
    2 = fatal error (missing deps, no input)

DEPENDENCIES
    system: tesseract-ocr with 'spa' traineddata, poppler-utils (pdftoppm)
    python: pyocr, pdf2image, Pillow
"""

from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

try:
    import pyocr
    import pyocr.builders
    from pdf2image import convert_from_path
    import PIL.Image
except ImportError as e:
    print(f"ERROR: missing dep ({e}). Try: pip install pyocr pdf2image Pillow",
          file=sys.stderr)
    sys.exit(2)


# ----------------------------------------------------------------------
# OCR tool bootstrap
# ----------------------------------------------------------------------

def get_ocr_tool() -> "pyocr.libtesseract":
    tools = pyocr.get_available_tools()
    if not tools:
        print("ERROR: no OCR tool available (install tesseract-ocr).",
              file=sys.stderr)
        sys.exit(2)
    tool = tools[0]
    langs = tool.get_available_languages()
    if "spa" not in langs:
        print(f"ERROR: Spanish language pack not installed for {tool.get_name()}. "
              f"Got: {langs}", file=sys.stderr)
        sys.exit(2)
    return tool


# ----------------------------------------------------------------------
# Single-PDF processing
# ----------------------------------------------------------------------

def ocr_pdf(pdf_path: Path, out_txt: Path, tool, dpi: int) -> bool:
    """Convert pdf_path to text via OCR. Returns True on success."""
    pages_dir = Path(tempfile.mkdtemp(prefix="extract_batch_"))
    try:
        try:
            pages = convert_from_path(str(pdf_path), dpi=dpi)
        except Exception as e:
            print(f"    [!] pdf2image failed: {e}", file=sys.stderr)
            return False

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with out_txt.open("w", encoding="utf-8") as fh:
            for i, page in enumerate(pages, 1):
                img_path = pages_dir / f"p{i:04d}.jpg"
                page.save(img_path, "JPEG")
                try:
                    text = tool.image_to_string(
                        PIL.Image.open(img_path),
                        lang="spa",
                        builder=pyocr.builders.TextBuilder(),
                    )
                except Exception as e:
                    print(f"    [!] OCR failed on page {i}: {e}", file=sys.stderr)
                    continue
                text = text.replace("-\n", "")
                fh.write(text)
                fh.write("\n")
                print(f"    page {i}/{len(pages)}: {len(text)} chars", end="\r")
            print()
        return True
    finally:
        shutil.rmtree(pages_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------

def collect_pdfs(pdfs_arg: list[Path]) -> list[Path]:
    """Expand directories to .pdf files, keep plain files as-is."""
    out: list[Path] = []
    for p in pdfs_arg:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pdf")))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            out.append(p)
        else:
            print(f"WARNING: ignoring {p} (not a pdf or dir)", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pdfs", nargs="*", type=Path,
                    help="PDF files or directories to process (alt: --pdfs)")
    ap.add_argument("--pdfs", dest="pdfs_opt", action="append", type=Path,
                    default=[], help="Additional PDF files or directories")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for .txt files")
    ap.add_argument("--dpi", type=int, default=500,
                    help="Raster DPI for the PDF → image step (default: 500)")
    ap.add_argument("--force", action="store_true",
                    help="Re-OCR even when the .txt already exists")
    args = ap.parse_args()

    inputs: list[Path] = list(args.pdfs) + list(args.pdfs_opt)
    if not inputs:
        ap.error("provide at least one PDF or directory (positional or via --pdfs)")

    pdfs = collect_pdfs(inputs)
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    tool = get_ocr_tool()

    print(f"Tool:   {tool.get_name()}")
    print(f"PDFs:   {len(pdfs)}")
    print(f"Out:    {args.out}")
    print(f"DPI:    {args.dpi}")
    print(f"Force:  {args.force}")
    print()

    ok = 0
    skipped = 0
    failed: list[Path] = []
    t0 = time.time()
    for i, pdf in enumerate(pdfs, 1):
        out_txt = args.out / (pdf.stem + ".txt")
        tag = f"[{i:>3}/{len(pdfs)}]"
        if out_txt.exists() and not args.force:
            print(f"{tag} SKIP  {pdf.name} (already in {out_txt})")
            skipped += 1
            continue
        print(f"{tag} OCR   {pdf.name}  →  {out_txt.name}")
        if ocr_pdf(pdf, out_txt, tool, args.dpi):
            ok += 1
        else:
            failed.append(pdf)
            if out_txt.exists() and out_txt.stat().st_size == 0:
                out_txt.unlink()

    dt = time.time() - t0
    print()
    print("=" * 60)
    print(f"  OK:      {ok}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {len(failed)}")
    print(f"  Elapsed: {dt:.1f}s")
    print("=" * 60)
    if failed:
        print("Failed PDFs:")
        for p in failed:
            print(f"  - {p}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
