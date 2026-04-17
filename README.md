# Fake News Detection (Multimodal V2) - B.Tech Final Year Project

This version implements the requested research upgrades:

1. RoBERTa text embeddings (replacing TF-IDF)
2. CLIP image branch + explicit text-image similarity
3. SHAP global/local explanation plots
4. Social-propagation GNN branch (FakeNewsNet-style graph JSON input)

## Architecture

- Text encoder: `roberta-base` CLS embedding
- Image encoder: `openai/clip-vit-base-patch32` image embedding
- Cross-modal signal: CLIP cosine similarity (`text_emb dot image_emb`)
- URL branch: domain credibility + lexical URL risk features
- Social branch: custom 2-layer GCN for propagation graph classification
- Fusion classifier: Logistic Regression on concatenated dense features
- Explainability:
  - SHAP global beeswarm
  - SHAP local waterfall
  - token-level leave-one-out importance

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- data/
|   |-- sample_fake_real_news.csv
|   `-- graphs/
|       |-- real_graph_1.json
|       |-- real_graph_2.json
|       |-- fake_graph_1.json
|       `-- fake_graph_2.json
|-- docs/
|   `-- research_and_datasets.md
`-- src/
    |-- evaluate.py
    |-- model.py
    |-- predict.py
    |-- train.py
    `-- utils.py
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note:
- First run downloads RoBERTa and CLIP weights from Hugging Face.
- Use `--device cuda` if GPU is available.

## Dataset Format

Recommended columns:

- `title`
- `text`
- `url`
- `image_path` (optional, can be blank)
- `graph_path` (optional, FakeNewsNet-like JSON path)
- `label` (`fake/real` or `0/1`)

Auto-detected alternatives are supported for URL/image/graph column names.

## Train

```powershell
python -m src.train --data data/sample_fake_real_news.csv --image-base-dir data --graph-base-dir data --device cpu
```

Outputs:

- `artifacts/fake_news_multimodal_v2.joblib`
- `artifacts/metrics_v2.json`
- `artifacts/shap_global.png` (if SHAP generation succeeds)

## Predict (CLI)

```powershell
python -m src.predict --title "Viral claim" --text "..." --url "http://example.xyz/claim" --image-path "D:\images\news.jpg" --graph-path "D:\graphs\propagation.json"
```

Output includes:
- final fake/real probability
- SHAP local top features
- token-level local importance
- local SHAP image path

## Evaluate (For Report Appendix)

```powershell
python -m src.evaluate --data data/sample_fake_real_news.csv --artifact artifacts/fake_news_multimodal_v2.joblib --image-base-dir data --graph-base-dir data
```

Generated outputs:
- `artifacts/eval_metrics.json`
- `artifacts/eval_report.txt`
- `artifacts/eval_predictions.csv`
- `artifacts/confusion_matrix.png`

## Streamlit Demo

```powershell
streamlit run app.py
```

App supports:
- headline + article text input
- URL input
- image upload
- propagation graph JSON upload
- modality strength display
- SHAP feature table + local waterfall plot

## FakeNewsNet Graph JSON (Supported Shape)

Minimal example:

```json
{
  "root": "u0",
  "nodes": [
    {"id": "u0", "timestamp": 0},
    {"id": "u1", "timestamp": 2}
  ],
  "edges": [["u0", "u1"]]
}
```

Also supports edge dict objects:

```json
{"source":"u0","target":"u1"}
```

## Important Notes

- Missing image or graph is handled gracefully (fallback features used).
- For small datasets, deep components may overfit; use this as project baseline and report limitations.
- For final report, include ablation:
  - text only
  - text + URL
  - text + URL + CLIP
  - text + URL + CLIP + GNN
