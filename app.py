from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils import clean_text, load_artifact


ARTIFACT_PATH = Path("artifacts/fake_news_multimodal_v2.joblib")
TMP_DIR = Path("artifacts/tmp_uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Multimodal Fake News Detector", page_icon="N", layout="wide")
st.title("Multimodal Fake News Detector (RoBERTa + CLIP + URL + GNN + SHAP)")
st.caption("B.Tech CSE Final Year Project")

if not ARTIFACT_PATH.exists():
    st.error(
        "Model artifact not found. Train first:\n"
        "`python src/train.py --data data/your_dataset.csv --image-base-dir data --graph-base-dir data`"
    )
    st.stop()

model = load_artifact(ARTIFACT_PATH)

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Headline / Title")
    url = st.text_input("Source URL", placeholder="https://example.com/news/article")
with col2:
    image_file = st.file_uploader("News Image", type=["png", "jpg", "jpeg", "webp"])
    graph_file = st.file_uploader("Propagation Graph JSON", type=["json"])

text = st.text_area("News Content", height=220, placeholder="Paste full article text...")
top_k = st.slider("Top LOO tokens", min_value=5, max_value=20, value=12, step=1)

if st.button("Detect", type="primary"):
    if not text.strip():
        st.warning("Please enter article text.")
        st.stop()

    image_path = ""
    graph_path = ""
    if image_file is not None:
        suffix = Path(image_file.name).suffix if Path(image_file.name).suffix else ".jpg"
        p = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
        p.write_bytes(image_file.getbuffer())
        image_path = str(p.resolve())
    if graph_file is not None:
        p = TMP_DIR / f"{uuid.uuid4().hex}.json"
        p.write_bytes(graph_file.getbuffer())
        graph_path = str(p.resolve())

    combined = clean_text(f"{title} {text}")
    prob_real = float(
        model.predict_proba(
            text=[combined],
            image_paths=[image_path],
            urls=[url],
            graph_paths=[graph_path],
        )[0, 1]
    )
    prob_fake = 1.0 - prob_real
    pred = "REAL" if prob_real >= 0.5 else "FAKE"
    if pred == "FAKE":
        st.error(f"Prediction: {pred}")
    else:
        st.success(f"Prediction: {pred}")
    st.write({"fake_probability": round(prob_fake, 4), "real_probability": round(prob_real, 4)})

    local = model.explain_local(
        text=combined,
        image_path=image_path,
        url=url,
        graph_path=graph_path,
        top_k_tokens=top_k,
    )
    st.subheader("Modality Strength (from SHAP)")
    st.write(local.get("modality_strength", {}))

    shap_items = local.get("shap_top_features", [])
    if shap_items:
        st.subheader("Top SHAP Features")
        st.dataframe(pd.DataFrame(shap_items), use_container_width=True)

    tokens = local.get("token_importance_loo", [])
    if tokens:
        st.subheader("Local Token Importance (Leave-One-Out)")
        st.dataframe(pd.DataFrame(tokens), use_container_width=True)

    shap_local_path = Path("artifacts/shap_local_app.png")
    try:
        out = model.save_shap_local_plot(
            out_path=shap_local_path,
            text=combined,
            image_path=image_path,
            url=url,
            graph_path=graph_path,
        )
        st.subheader("Local SHAP Waterfall")
        st.image(out, caption="Local explanation")
    except Exception as exc:
        st.info(f"Local SHAP plot could not be generated: {exc}")

global_plot = Path("artifacts/shap_global.png")
if global_plot.exists():
    st.subheader("Global SHAP Plot (Training Background)")
    st.image(str(global_plot.resolve()), caption="Global feature impact")
