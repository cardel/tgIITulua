# LLM-assisted classifier — degree projects of Universidad del Valle, Tuluá

This subdirectory hosts the LLM-based classification pipeline used to label the
277 degree projects (2012–2025) of the Systems Engineering program at
Universidad del Valle, Tuluá, against a curated 23-term subset of the
**IEEE Thesaurus**. It is the reproducibility and audit-trail companion of the
paper *Classification and Impact of the Degree Projects of a Professional
Program in Systems Engineering: An LLM-Assisted Reappraisal*.

The legacy TF-IDF + NMF pipeline (`process-data.py`, `modeloTG.csv`,
`final_results.xlsx`) lives at the repo root and is **unchanged** so both
pipelines remain reproducible side by side, as required by the paper's
methodology section.

## What is in this directory

```
llm-classifier/
├── prompts/
│   ├── classify_v1.md     # 3-bucket taxonomy used in earlier drafts
│   └── classify_v2.md     # IEEE Thesaurus 23-term taxonomy (active)
├── schemas/
│   └── verdict.schema.json # JSON schema validated for every verdict
├── scripts/
│   ├── classify.py         # talks to a local Ollama, writes one verdict per project
│   ├── evaluate.py         # accuracy / macro-F1 / confusion vs. expert gold set
│   ├── compare_models.py   # cross-model comparison table
│   ├── generate_results.py # consolidates verdicts to xlsx
│   ├── build_corpus.py     # builds the per-project metadata corpus
│   ├── render_confusion.py # renders the confusion-matrix TikZ figure
│   ├── build_eval_figures.py # renders per-class F1 and precision/recall figures
│   ├── run_all_models.sh   # orchestrates the full sweep over the three models
│   └── requirements.txt    # Python dependencies (the runtime is Ollama)
├── verdicts/
│   ├── qwen2.5_14b-instruct/        # 186 v2 verdicts (current taxonomy)
│   ├── llama3.1_8b/                 # 227 v2 verdicts
│   ├── mistral_7b-instruct/         # 224 verdicts (139 v2 + earlier runs)
│   ├── qwen2.5_14b-instruct_v1_backup/  # 198 historical v1 verdicts
│   ├── llama3.1_8b_v1_backup/       # 204 historical v1 verdicts
│   └── _smoke/                      # 1 sanity-check verdict
├── results/
│   ├── corpus.xlsx                  # input corpus (project_id, title, keywords, abstract, ...)
│   ├── corpus_stats.txt             # record counts per source
│   ├── llm_results_<model>.xlsx     # per-project verdicts, one xlsx per model
│   ├── llm_distribution_<model>.xlsx # IEEE-term counts and percentages per model
│   ├── metrics_<model>.json         # accuracy, macro-precision, macro-recall, macro-F1, confusion
│   ├── metrics_<model>.tex          # rendered LaTeX summary table
│   ├── models_comparison.json       # cross-model headline metrics + per-class details
│   └── models_comparison.tex        # cross-model comparison table (LaTeX, used in the paper)
└── figures/
    ├── confusion_matrix.tex         # TikZ confusion matrix for the best model (qwen2.5)
    ├── per_class_f1_support.tex     # per-class F1 with gold support annotated
    └── precision_recall_gap.tex     # per-class precision vs. recall (granularity-gap signature)
```

## Models evaluated

| Model tag                     | Parameters | 4-bit footprint | Macro-F1 (gold N=109–111) |
|-------------------------------|-----------:|----------------:|--------------------------:|
| `qwen2.5:14b-instruct`        | 14.7 B     | ≈ 9 GB          | 34.2 %                    |
| `llama3.1:8b`                 | 8.0 B      | ≈ 5 GB          | 34.1 %                    |
| `mistral:7b-instruct`         | 7.2 B      | ≈ 4 GB          | 33.4 %                    |

Headline numbers and per-class breakdowns are in `results/models_comparison.json`
and `results/metrics_<model>.json`. The figures under `figures/` are the same
ones rendered in the paper.

## Reproducing a run

The pipeline runs locally on a host with [Ollama](https://ollama.com) installed.
Python dependencies are listed in `scripts/requirements.txt`.

```bash
pip install -r llm-classifier/scripts/requirements.txt

# one-time: pull the three Ollama models
ollama pull qwen2.5:14b-instruct
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct

# run all three models on the corpus, then evaluate
bash llm-classifier/scripts/run_all_models.sh
```

The verdict cache makes re-runs cheap: any project whose verdict JSON already
exists in `verdicts/<model>/` is skipped on the next pass. To force a fresh
classification (for example after editing a prompt), move the affected
verdict directory aside:

```bash
mv llm-classifier/verdicts/qwen2.5_14b-instruct \
   llm-classifier/verdicts/qwen2.5_14b-instruct_v2_backup
```

## Verdict format

Every JSON record in `verdicts/<model>/<project_id>.json` carries:

- `project_id` — the degree-project identifier (e.g., `2018A1`).
- `model_id` — the Ollama tag used.
- `prompt_version` — `v1` for the 3-bucket taxonomy, `v2` for the IEEE
  Thesaurus subset.
- `input_hash` — SHA-256 over `(project_id, prompt_version, input_blob)`,
  used as the cache key.
- `input_source` — `full_text` when the extracted PDF text was available,
  `metadata_only` when the model saw only title + keywords + abstract.
- `generated_at` — UTC timestamp.
- `verdict` — the structured response: `primary_term`, `secondary_term`,
  `confidence`, `justification`, `keywords_cited`. The schema lives in
  `schemas/verdict.schema.json`.

## Gold set

Expert labels live at `raw-data/datosTG.xlsx` in the column `CATEGORIA REVISA`,
covering 111 projects from 2012–2018. The mapping from the 16 fine-grained
expert labels to the 23 IEEE Thesaurus terms is in
`scripts/evaluate.py::GOLD_NORMALIZATION`. The gold set is preserved in this
repo; it is the same one used by the legacy NMF pipeline.

## Caveats

- `evaluate.py` reports macro-averaged metrics; eight of the sixteen
  gold-set classes have support N ≤ 2, so a single misclassification drops
  the class F1 to zero and pulls the macro average down. When the macro
  average is restricted to classes with support ≥ 3, qwen2.5:14b-instruct
  rises from 34.2 % to about 57 %.
- The IEEE taxonomy is finer than the 16-label expert vocabulary, so projects
  the experts placed under `Information Systems` are reassigned by the model
  to legitimate children of that family (`Enterprise information systems`,
  `Web applications`, `Mobile applications`). This shows up as high precision
  with low recall on the parent classes; the figure
  `figures/precision_recall_gap.tex` makes the pattern explicit.
- Schema-failure rates collapse to under 4 % across the three models once
  classification runs on uniform metadata-only inputs. The validator drops
  out-of-vocabulary `secondary_term` values (e.g. `Semantic web`) and the
  `"Term | null"` artifact occasionally emitted by Qwen, keeping the primary
  classification.

## License

MIT, the same as the rest of this repository.
