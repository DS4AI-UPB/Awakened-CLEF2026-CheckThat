# CT26 Task 1 Final Notebook

The notebook retrieves the scientific paper implicitly referenced by a social media post in English, German, or French from a collection of 10,000 English scientific papers.

## Pipeline

The notebook implements the final multi-stage retrieval system:

1. Load the CheckThat! Task 1 collection and language splits from Hugging Face.
2. Translate German and French queries to English using cached/precomputed translations.
3. Generate document-side synthetic tweet-style claims with `gpt-4o-mini`.
4. Append synthetic claims to paper text for dense first-stage retrieval.
5. Fine-tune `intfloat/multilingual-e5-large` with two rounds of iterative hard-negative mining.
6. Retrieve candidates with the fine-tuned dense bi-encoder.
7. Fine-tune `BAAI/bge-reranker-v2-m3` as a cross-encoder reranker.
8. Rerank top candidates with the cross-encoder.
9. Combine normalized cross-encoder and first-stage scores with:

```text
final_score = CE_score_norm + 0.15 * first_stage_score_norm
```

10. Export dev/test prediction TSV files and zip archives.

## Requirements

Recommended runtime:

- Python 3.10+
- CUDA GPU strongly recommended; the original experiments were run on GPU/Colab-style hardware.
- Hugging Face account with access to `sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims`
- OpenAI API key for translation/doc-side augmentation if caches are missing

The notebook installs its Python dependencies:

```python
!pip install -q transformers datasets sentence-transformers rank-bm25 openai tenacity huggingface_hub
```

Main libraries:

- `datasets`
- `transformers`
- `sentence-transformers`
- `rank-bm25`
- `openai`
- `tenacity`
- `torch`
- `numpy`
- `pandas`

## Required Files

Run the notebook from the `task1/` directory.

Expected local translation files:

```text
train_en_queries_de.json
train_en_queries_fr.json
train_en_queries_en.json
dev_en_queries_de.json
dev_en_queries_fr.json
dev_en_queries_en.json
test_en_queries_de.json
test_en_queries_fr.json
test_en_queries_en.json
```

The train/dev translation files are present in this repository. The test translation files may need to be generated first. In the notebook, the GPT-based test translation block is present but commented out; uncomment it if the `test_en_queries_*.json` files are missing.

## Configuration

Important notebook configuration values:

```python
OPENAI_MODEL_FAST = "gpt-4o-mini"
DENSE_MODEL = "intfloat/multilingual-e5-large"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LANGS = ["de", "fr", "en"]

N_SYNTH_TWEETS_PER_PAPER = 2
N_HARD_NEGS = 5
HARD_NEG_POOL = 50
N_FT_ROUNDS = 2
FT_EPOCHS = 2
FT_BATCH_SIZE = 32
RERANK_FT_EPOCHS = 1
RERANK_FT_BATCH = 16
FIRST_STAGE_TOPK = 100
CE_RERANK_TOPK = 20
INTERP_ALPHA = 0.15
```

The notebook creates:

```text
v5_cache/
v5_models/
v5_out/
```

`v5_cache/` stores OpenAI generations/translations so repeated runs do not pay for the same prompts again.

## How To Run

1. Open `CT26_Task1_final.ipynb`.
2. Set `OPENAI_API_KEY` in the config cell, or provide it through the environment before running.
3. Authenticate with Hugging Face when prompted by `notebook_login()`.
4. Make sure all `*_en_queries_*.json` files exist, especially the test translation files.
5. Run cells from top to bottom through the dev evaluation/export cells.
6. For test submission, run the test inference/export cell that uses:

```python
run_full_pipeline(query_orig, query_en)
```

This is the version that applies dense retrieval, cross-encoder reranking, and weighted score combination.

## Outputs

Development predictions:

```text
v5_out/dev_predictions_de.tsv
v5_out/dev_predictions_fr.tsv
v5_out/dev_predictions_en.tsv
v5_out/dev_predictions.zip
```

Test predictions:

```text
v5_out/predictions_de.tsv
v5_out/predictions_fr.tsv
v5_out/predictions_en.tsv
v5_out/test_predictions.zip
```

Each TSV contains:

```text
index    preds
```

where `preds` is the list of top-5 predicted publication IDs.
