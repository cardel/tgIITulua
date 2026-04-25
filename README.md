# Process final work degree Universidad del Valle

This work is related to the extraction of data and categorization of the final works from the Universidad del Valle 2012-2018

## Required libraries

* nltk
* gensim
* sklearn

## Stop Words

* Spanish dictionary of nltk
* elaboración propia
* ilustración
* presente proyecto
* varchar
* consultar
* usuario
* permisos
* descripción general
* grado
* trabajo
* universidad 
* valle
* tulua
* inglés
* clínica
* municipio
* sede
* contenido
* tabla
* figura

## Usage

```bash
python process-data.py
```


## LLM-assisted classifier (extension)

The subdirectory [`llm-classifier/`](llm-classifier/) contains the
LLM-based classification pipeline that complements the NMF baseline above.
It introduces:

- a curated 23-term subset of the **IEEE Thesaurus** as the target taxonomy;
- a structured-JSON prompt (`llm-classifier/prompts/classify_v2.md`) and
  schema validator (`llm-classifier/schemas/verdict.schema.json`);
- per-project verdicts under `llm-classifier/verdicts/<model>/`;
- consolidated results, metrics, and cross-model comparison under
  `llm-classifier/results/`;
- TikZ figures under `llm-classifier/figures/` (confusion matrix, per-class
  F1 with support, precision-vs-recall gap).

Three locally executed open-weight models are evaluated against the same
expert gold set used by the NMF baseline (`raw-data/datosTG.xlsx`,
column `CATEGORIA REVISA`): `qwen2.5:14b-instruct`, `llama3.1:8b`, and
`mistral:7b-instruct`. See [`llm-classifier/README.md`](llm-classifier/README.md)
for layout, reproducibility instructions, and caveats.

The legacy NMF pipeline (`process-data.py`, `modeloTG.csv`,
`final_results.xlsx`) remains unchanged so the two pipelines stay
reproducible side by side.

## License
[MIT](https://choosealicense.com/licenses/mit/)