from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
import torch
import logging
import sys
import os
import re
import random
import tkinter as tk
from tkinter import scrolledtext, Button, Entry, Label, PhotoImage, font, messagebox

# Logging setup
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.StreamHandler(stream=sys.stdout)])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;\'"():%/-]', '', text)
    return text.strip()

def assess_compliance_risk(law1_text, law2_text, query=""):
    risk_score = 0.0
    explanation = "No significant conflict detected."

    if "shall not" in law1_text and "shall" in law2_text:
        risk_score = 0.6 + random.uniform(0, 0.2)
        explanation = f"⚠ Conflict regarding '{query}': One law prohibits what another mandates. Recommendation: Adhere to the stricter clause."
    elif "must" in law1_text and "may not" in law2_text:
        risk_score = 0.6 + random.uniform(0, 0.2)
        explanation = f"⚠ Conflict regarding '{query}': One law requires what another forbids. Recommendation: Clarify the intent with regulatory counsel."
    elif "is prohibited" in law1_text and "is allowed" in law2_text:
        risk_score = 0.7 + random.uniform(0, 0.2)
        explanation = f"🚨 Direct contradiction on '{query}': One law allows what another prohibits. Recommendation: Follow the most recent or stricter law."
    elif "should" in law1_text and "is not required" in law2_text:
        risk_score = 0.35
        explanation = f"⚠ Potential ambiguity in guidance for '{query}'. Recommendation: Treat 'should' as 'must' for better compliance posture."

    return risk_score, explanation

def main():
    try:
        model_path = "trained_sbert_mininglaw_risk_aware_with_negatives"
        if not os.path.exists(model_path):
            raise ValueError(f"Trained model not found at {model_path}")
        model = SentenceTransformer(model_path)
        logging.info(f"✅ Loaded trained model from '{model_path}'")
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        return

    client = MongoClient("mongodb://localhost:27017/")
    db = client["mining_law_db"]
    pdf_data_collection = db["pdf_data"]

    try:
        pdf_data = list(pdf_data_collection.find({"text": {"$exists": True}}))
        if not pdf_data:
            logging.warning("⚠ No data found in 'pdf_data'")
            client.close()
            return
        logging.info(f"📄 Loaded {len(pdf_data)} documents from MongoDB")
    except Exception as e:
        logging.error(f"❌ Failed to load data: {e}")
        client.close()
        return

    try:
        all_texts = [doc["text"] for doc in pdf_data]
        pdf_embeddings = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=True)
        logging.info("✅ Embedded all law documents")
    except Exception as e:
        logging.error(f"❌ Error embedding documents: {e}")
        client.close()
        return

    def search_laws(query):
        try:
            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(query_embedding, pdf_embeddings)[0]
            top_results = torch.topk(similarities, 3)

            results_text.delete(1.0, tk.END)

            top_indices = top_results.indices.tolist()
            top_scores = top_results.values.tolist()

            for i, idx in enumerate(top_indices):
                doc = pdf_data[idx]
                filename = doc.get("filename", "Unknown")
                excerpt = doc.get("text", "")[:300].strip()
                score = top_scores[i]
                
                if score < 0.15:  # Check for similarity score below 0.2
                    results_text.insert(tk.END, "\n❌ Irrelevant question. The query does not match any relevant legal text.\n")
                    return  # Exit the function if the score is too low

                results_text.insert(tk.END, f"\n🔹 Match {i+1} for query: '{query}'\n")
                results_text.insert(tk.END, f"📄 Document: {filename}\n")
                results_text.insert(tk.END, f"📝 Content Excerpt: {excerpt}...\n")
                results_text.insert(tk.END, f"🔗 Similarity Score: {score:.4f}\n")
                results_text.insert(tk.END, "-" * 60 + "\n")

            results_text.insert(tk.END, "\n⚖️ Compliance Risk Evaluation\n")
            contradiction_flag = False

            for i in range(len(top_indices)):
                for j in range(i + 1, len(top_indices)):
                    law1 = pdf_data[top_indices[i]].get("text", "")
                    law2 = pdf_data[top_indices[j]].get("text", "")
                    risk_score, explanation = assess_compliance_risk(law1, law2, query)

                    if risk_score > 0.0:
                        contradiction_flag = True
                        results_text.insert(tk.END, f"\n🚨 Contradiction Detected Between Match {i+1} and Match {j+1}\n")
                        results_text.insert(tk.END, f"🔺 Risk Score: {risk_score:.2f}\n")
                        results_text.insert(tk.END, f"📌 Legal Recommendation: {explanation}\n")

                        # ➕ Suggest an alternative non-conflicting document
                        excluded = {top_indices[i], top_indices[j]}
                        for k, doc in enumerate(pdf_data):
                            if k in excluded:
                                continue
                            alt_text = doc.get("text", "")
                            alt_score = float(util.pytorch_cos_sim(query_embedding, model.encode(alt_text, convert_to_tensor=True))[0])
                            if alt_score > 0.5:
                                results_text.insert(tk.END, f"\n🧭 Suggested Alternative Law: {doc.get('filename', 'Unnamed')}\n")
                                results_text.insert(tk.END, f"📑 Excerpt: {alt_text[:200]}...\n")
                                results_text.insert(tk.END, f"🔗 Similarity: {alt_score:.4f}\n")
                                break

                    results_text.insert(tk.END, "-" * 60 + "\n")

            if not contradiction_flag:
                results_text.insert(tk.END, "\n✅ No contradictions detected. The legal provisions appear consistent.\n")
                results_text.insert(tk.END, "-" * 60 + "\n")

        except Exception as e:
            logging.error(f"❌ Query error: {e}")
            results_text.insert(tk.END, f"\n❌ Error processing query: {e}")

    def on_search_button_click():
        query = query_entry.get().strip()
        if query:
            search_laws(query)

    root = tk.Tk()
    root.title("Mining Law Chatbot")

    # --- Styling ---
    bg_color = '#E6F2FF'  # Light blue
    text_color = '#2E3B55'  # Dark blue-gray
    button_color = '#66B2FF'  # Medium blue
    highlight_color = '#A0D6FF' # Lighter blue for highlights
    font_family = "Arial"  # Change this to your desired font family
    font_size = 11
    font_style = font.Font(family=font_family, size=font_size)
    bold_font = font.Font(family=font_family, size=font_size, weight="bold")

    root.configure(bg=bg_color)

    # --- Header ---
    header_font = font.Font(family=font_family, size=18, weight="bold")
    header_label = Label(root, text="Mining Law Compliance Chatbot", bg=bg_color, fg=text_color, font=header_font)
    header_label.pack(pady=(10, 20))

    # --- Query Input ---
    query_label = Label(root, text="🔍 Enter your mining law question:", bg=bg_color, fg=text_color, font=bold_font)
    query_label.pack(pady=(0, 5))

    query_entry = Entry(root, width=60, font=font_style, bg='white', fg=text_color)
    query_entry.pack(pady=(0, 10))

    search_button = Button(root, text="Search", command=on_search_button_click, bg=button_color, fg='white', font=bold_font, relief=tk.RAISED, borderwidth=2)
    search_button.pack(pady=(0, 15))
    search_button.config(highlightbackground=bg_color)  # Remove the border

    # --- Results Output ---
    results_label = Label(root, text="📋 Results, Contradictions & Recommendations:", bg=bg_color, fg=text_color, font=bold_font)
    results_label.pack()

    results_text = scrolledtext.ScrolledText(root, width=90, height=30, bg='#F9F9F9', fg=text_color, font=font_style)
    results_text.pack(pady=(0, 20))

    # --- Center Window ---
    root.eval('tk::PlaceWindow . center')

    # --- Run Main Loop ---
    root.mainloop()
    client.close()

if __name__ == "__main__":
    main()
