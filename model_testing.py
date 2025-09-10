import os
import re
import random
import logging
from typing import List, Dict, Tuple

import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient


# -------------------------
# Streamlit Page Settings
# -------------------------
st.set_page_config(page_title="Mining Law Compliance Chatbot (RAG)", page_icon="⚖️", layout="wide")
logging.basicConfig(level=logging.INFO)


# -------------------------
# Utilities
# -------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,;'""():%/-]", '', text)
    return text.strip()


def generate_response_with_llm(query: str, retrieved_docs: List[Dict]) -> Tuple[str, float, str]:
    """
    Simulated LLM analysis, ported from test.py.
    Returns (formatted_markdown_response, risk_score, recommendation)
    """
    if not retrieved_docs:
        return (
            "No relevant documents found to process your query.",
            0.0,
            "No specific recommendations.",
        )

    context_str = "\n\n".join(
        [
            f"--- Document: {doc['filename']} (Similarity: {doc['score']:.4f}) ---\n{doc['text']}"
            for doc in retrieved_docs
        ]
    )

    # Simulated logic (keyword heuristics)
    if "prohibit" in query.lower() and any("allowed" in doc['text'].lower() for doc in retrieved_docs):
        risk_score = 0.7 + random.uniform(0, 0.25)
        response = f"""
**Summary of Legal Provisions:**
Document 1 generally prohibits activities related to {query}. Document 2, however, contains clauses that allow similar activities under certain conditions.

**Identified Obligations and Prohibitions:**
From Document 1: "No mining activities are permitted within 500 meters of a protected forest."
From Document 2: "Mining is allowed within 200 meters of a designated ecological zone, provided specific conditions are met."

**Compliance Risk Assessment:** High - {risk_score:.2f}

**Reasoning for Risk:**
There is a direct contradiction regarding mining proximity to protected areas. Document 1 states "no mining activities are permitted within 500 meters," while Document 2 states "mining is allowed within 200 meters." This creates significant ambiguity and a high risk of non-compliance if not properly addressed.

**Legal Recommendations:**
1. Immediately clarify the specific legal distances and conditions for mining near protected areas with the relevant regulatory body.
2. Identify which law takes precedence (e.g., more recent law, specific versus general).
3. Adopt the most stringent requirement to ensure full compliance (i.e., the 500-meter prohibition).
4. Update internal compliance protocols and training materials accordingly.
"""
        return response, risk_score, (
            "Clarify conflicting distances, identify precedence, and adopt the most stringent requirement (500 meters)."
        )

    elif "report" in query.lower() and "quarterly" in context_str and "annual" in context_str:
        risk_score = 0.4 + random.uniform(0, 0.25)
        response = f"""
**Summary of Legal Provisions:**
One document mandates quarterly environmental compliance reports, while another suggests annual environmental impact assessments. Both relate to environmental reporting.

**Identified Obligations and Prohibitions:**
From Document A: "This regulation mandates that all mining operations shall submit a quarterly environmental compliance report."
From Document B: "This guideline suggests that mining companies should submit an annual environmental impact assessment."

**Compliance Risk Assessment:** Medium - {risk_score:.2f}

**Reasoning for Risk:**
While not a direct prohibition vs. allowance, there is a difference in reporting frequency (quarterly vs. annual) and type (compliance report vs. impact assessment). This could lead to confusion or overlooking a requirement. "Shall" implies a stronger obligation than "suggests."

**Legal Recommendations:**
1. Ensure adherence to the more frequent (quarterly) reporting requirement.
2. Submit both quarterly compliance reports and annual impact assessments if both are deemed applicable.
3. Seek clarification from the environmental regulatory body regarding the scope and frequency of required reports.
"""
        return response, risk_score, (
            "Adhere to quarterly reporting, consider submitting both, and seek clarification from regulators."
        )

    else:
        risk_score = 0.1 + random.uniform(0, 0.2)
        response = f"""
**Summary of Legal Provisions:**
The retrieved documents provide general guidance on {query.lower()}. They emphasize responsible practices and adherence to established regulations.

**Identified Obligations and Prohibitions:**
No explicit, direct contradictions or highly specific obligations/prohibitions were identified across the top relevant documents for this general query.

**Compliance Risk Assessment:** Low - {risk_score:.2f}

**Reasoning for Risk:**
Based on the provided context, the legal provisions related to your query appear broadly consistent and do not present immediate, obvious conflicts. The risk is considered low, indicating general clarity.

**Legal Recommendations:**
1. Continue to monitor for any new regulations or amendments related to {query.lower()}.
2. Ensure regular training for personnel on general compliance principles.
3. Periodically review internal policies to align with best practices.
"""
        return response, risk_score, (
            "Monitor new regulations, conduct regular training, and review internal policies."
        )


# -------------------------
# Cached Loaders
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> SentenceTransformer:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    return SentenceTransformer(model_path)


@st.cache_resource(show_spinner=False)
def load_pdf_data(mongo_uri: str, db_name: str, collection_name: str) -> List[Dict]:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    pdf_data_collection = db[collection_name]
    docs = list(pdf_data_collection.find({"text": {"$exists": True}}))
    client.close()
    return docs


@st.cache_resource(show_spinner=False)
def compute_embeddings(model_path: str, texts: List[str]):
    model = load_model(model_path)
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.title("⚖️ Settings")

with st.sidebar:
    st.markdown("**Model & Data**")
    model_path = st.text_input(
        "SentenceTransformer model path",
        value="trained_sbert_mininglaw_risk_aware_with_negatives",
    )
    mongo_uri = st.text_input("MongoDB URI", value="mongodb://localhost:27017/")
    db_name = st.text_input("Database", value="mining_law_db")
    collection_name = st.text_input("Collection", value="pdf_data")

    st.markdown("**Retrieval**")
    top_k = st.slider("Top K Documents", 1, 10, 3)
    min_relevance_threshold = st.slider("Min Relevance Threshold", 0.0, 1.0, 0.125, 0.005)
    excerpt_len = st.slider("Excerpt length (chars)", 100, 1200, 300, 50)

    st.markdown("**Behavior**")
    random_seed = st.number_input("Random seed", value=42, step=1)
    st.caption("Used only by the simulated LLM for reproducibility")

    trigger_load = st.button("Load / Refresh Data", type="primary")


# -------------------------
# Data Loading & Embedding
# -------------------------
if "app_state" not in st.session_state:
    st.session_state.app_state = {
        "pdf_data": None,
        "pdf_embeddings": None,
        "all_texts": None,
        "loaded_signature": None,
    }

app_state = st.session_state.app_state

# Define a signature that determines whether reload is necessary
current_signature = (model_path, mongo_uri, db_name, collection_name)

if trigger_load or app_state.get("loaded_signature") != current_signature:
    with st.spinner("Loading model and documents, computing embeddings..."):
        try:
            random.seed(int(random_seed))

            pdf_data = load_pdf_data(mongo_uri, db_name, collection_name)
            if not pdf_data:
                st.warning("No data found in MongoDB collection. Please ensure your PDFs are processed.")
                app_state.update({
                    "pdf_data": [],
                    "pdf_embeddings": None,
                    "all_texts": [],
                    "loaded_signature": current_signature,
                })
            else:
                all_texts = [clean_text(doc.get("text", "")) for doc in pdf_data]
                embeddings = compute_embeddings(model_path, all_texts)
                app_state.update({
                    "pdf_data": pdf_data,
                    "pdf_embeddings": embeddings,
                    "all_texts": all_texts,
                    "loaded_signature": current_signature,
                })
                st.success(f"Loaded {len(pdf_data)} documents and computed embeddings.")
        except Exception as e:
            st.error(f"Failed to load resources: {e}")
            app_state.update({
                "pdf_data": [],
                "pdf_embeddings": None,
                "all_texts": [],
                "loaded_signature": current_signature,
            })


# -------------------------
# Main UI
# -------------------------
st.markdown("""
<h2 style='text-align: center;'>⚖️ Mining Law Compliance Chatbot (RAG)</h2>
<p style='text-align: center;'>Ask a question about mining laws. The app retrieves the most relevant documents from MongoDB using your SentenceTransformer model and generates a simulated LLM analysis.</p>
""", unsafe_allow_html=True)

with st.form("query_form"):
    user_query = st.text_input("Your mining law question:", placeholder="e.g., Are there prohibitions on mining near protected forests?")
    submitted = st.form_submit_button("Search")

if submitted:
    if not user_query.strip():
        st.warning("Please enter a query.")
    elif app_state.get("pdf_embeddings") is None or not app_state.get("all_texts"):
        st.warning("Please load data first from the sidebar.")
    else:
        try:
            model = load_model(model_path)
            query_embedding = model.encode(clean_text(user_query), convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(query_embedding, app_state["pdf_embeddings"])[0]

            values, indices = torch.topk(similarities, k=min(top_k, similarities.shape[0]))
            top_scores = values.tolist()
            top_indices = indices.tolist()

            if not top_scores or top_scores[0] < float(min_relevance_threshold):
                st.error("Irrelevant question. The query does not match any relevant legal text with sufficient confidence.")
            else:
                retrieved_docs_for_llm: List[Dict] = []

                st.subheader("Retrieved Documents")
                for rank, idx in enumerate(top_indices, start=1):
                    doc = app_state["pdf_data"][idx]
                    filename = doc.get("filename", "Unknown")
                    text = clean_text(doc.get("text", ""))
                    score = float(top_scores[rank - 1])

                    retrieved_docs_for_llm.append({
                        "filename": filename,
                        "text": text,
                        "score": score,
                    })

                    with st.expander(f"{rank}. {filename} — similarity {score:.4f}"):
                        st.caption("Excerpt")
                        st.write(text[:excerpt_len] + ("..." if len(text) > excerpt_len else ""))

                st.markdown("---")
                st.subheader("LLM Response (Simulated)")
                full_response_md, risk_score, recommendation = generate_response_with_llm(
                    user_query, retrieved_docs_for_llm
                )
                st.markdown(full_response_md)
                st.info(f"Overall LLM-Assessed Risk Score: {risk_score:.2f}")
                st.success(f"Final Legal Recommendation: {recommendation}")
                st.markdown("\n")
        except Exception as e:
            st.error(f"Error processing query: {e}")


# -------------------------
# Footer
# -------------------------
st.markdown("""
<hr />
<small>Tip: Adjust thresholds and Top K from the sidebar. This demo uses a simulated LLM to illustrate the full RAG flow.</small>
""", unsafe_allow_html=True)


