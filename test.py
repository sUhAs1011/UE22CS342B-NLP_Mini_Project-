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
from typing import List, Dict, Tuple

# --- LLM Integration Placeholder ---
# In a real application, you would replace this with an actual LLM API call
# For example, using Google's Gemini API:
# from google.generativeai import GenerativeModel
# model = GenerativeModel('gemini-pro')
# response = model.generate_content(prompt)
# Or for OpenAI:
# from openai import OpenAI
# client = OpenAI()
# response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[...])

def generate_response_with_llm(query: str, retrieved_docs: List[Dict]) -> Tuple[str, float, str]:
    """
    Simulates interaction with an LLM to generate a response, assess risk,
    and provide recommendations based on retrieved legal documents.

    Args:
        query (str): The user's original query.
        retrieved_docs (List[Dict]): A list of dictionaries, where each dict
                                    contains 'filename', 'text', and 'score' for retrieved documents.

    Returns:
        Tuple[str, float, str]: (LLM's generated response, risk_score, legal_recommendation)
    """
    if not retrieved_docs:
        return "No relevant documents found to process your query.", 0.0, "No specific recommendations."

    context_str = "\n\n".join([f"--- Document: {doc['filename']} (Similarity: {doc['score']:.4f}) ---\n{doc['text']}" for doc in retrieved_docs])

    # Construct a detailed prompt for the LLM
    prompt = f"""
    You are an AI assistant specializing in mining law compliance. Your task is to analyze legal texts, identify obligations, prohibitions, and potential conflicts, and provide a compliance risk score and actionable recommendations.

    Here is the user's question: "{query}"

    Here are the relevant legal documents:
    {context_str}

    Based on the provided documents and the user's question, please perform the following:
    1.  **Summarize the key legal provisions** directly relevant to the user's query from *all* provided documents.
    2.  **Identify specific obligations and prohibitions**. Quote the exact clauses where possible.
    3.  **Assess the compliance risk (Low, Medium, High)**.
        * **Low Risk:** Provisions are clear, consistent, and easy to comply with.
        * **Medium Risk:** Some ambiguities, potential for varying interpretations, or minor inconsistencies.
        * **High Risk:** Direct contradictions, unclear or impossible obligations, significant gaps, or high potential for non-compliance.
    4.  **Explain the reasoning** for the assigned risk level, citing specific contradictory or ambiguous clauses if they exist.
    5.  **Provide clear, actionable legal recommendations** for a mining company to ensure compliance.

    Format your response as follows:
    **Summary of Legal Provisions:**
    [Your summary here]

    **Identified Obligations and Prohibitions:**
    [List specific obligations and prohibitions, quoting relevant text]

    **Compliance Risk Assessment:** [Low/Medium/High] - [Numerical Score e.g., 0.1-0.3 for Low, 0.4-0.6 for Medium, 0.7-1.0 for High]

    **Reasoning for Risk:**
    [Your explanation of why this risk level was assigned, citing specific conflicting or ambiguous parts of the text]

    **Legal Recommendations:**
    [Actionable steps for compliance]
    """

    # --- Placeholder for actual LLM API call ---
    # In a real scenario, you would send 'prompt' to your LLM API
    # and parse its response.
    # For now, we'll simulate a response.
    logging.info("Simulating LLM response for RAG query...")

    simulated_llm_response = ""
    simulated_risk_score = 0.0
    simulated_recommendation = ""

    # Simple simulation based on keywords in query and retrieved docs
    # This is *not* a true LLM; it's a stand-in for demonstration.
    if "prohibit" in query.lower() and any("allowed" in doc['text'].lower() for doc in retrieved_docs):
        simulated_llm_response = f"""
        **Summary of Legal Provisions:**
        Document 1 generally prohibits activities related to {query}. Document 2, however, contains clauses that allow similar activities under certain conditions.

        **Identified Obligations and Prohibitions:**
        From Document 1: "No mining activities are permitted within 500 meters of a protected forest."
        From Document 2: "Mining is allowed within 200 meters of a designated ecological zone, provided specific conditions are met."

        **Compliance Risk Assessment:** High - {0.7 + random.uniform(0, 0.25):.2f}

        **Reasoning for Risk:**
        There is a direct contradiction regarding mining proximity to protected areas. Document 1 states "no mining activities are permitted within 500 meters," while Document 2 states "mining is allowed within 200 meters." This creates significant ambiguity and a high risk of non-compliance if not properly addressed.

        **Legal Recommendations:**
        1.  Immediately clarify the specific legal distances and conditions for mining near protected areas with the relevant regulatory body.
        2.  Identify which law takes precedence (e.g., more recent law, specific versus general).
        3.  Adopt the most stringent requirement to ensure full compliance (i.e., the 500-meter prohibition).
        4.  Update internal compliance protocols and training materials accordingly.
        """
        simulated_risk_score = 0.7 + random.uniform(0, 0.25)
        simulated_recommendation = "Clarify conflicting distances, identify precedence, and adopt the most stringent requirement (500 meters)."
    elif "report" in query.lower() and "quarterly" in context_str and "annual" in context_str:
        simulated_llm_response = f"""
        **Summary of Legal Provisions:**
        One document mandates quarterly environmental compliance reports, while another suggests annual environmental impact assessments. Both relate to environmental reporting.

        **Identified Obligations and Prohibitions:**
        From Document A: "This regulation mandates that all mining operations shall submit a quarterly environmental compliance report."
        From Document B: "This guideline suggests that mining companies should submit an annual environmental impact assessment."

        **Compliance Risk Assessment:** Medium - {0.4 + random.uniform(0, 0.25):.2f}

        **Reasoning for Risk:**
        While not a direct prohibition vs. allowance, there is a difference in reporting frequency (quarterly vs. annual) and type (compliance report vs. impact assessment). This could lead to confusion or overlooking a requirement. "Shall" implies a stronger obligation than "suggests."

        **Legal Recommendations:**
        1.  Ensure adherence to the more frequent (quarterly) reporting requirement.
        2.  Submit both quarterly compliance reports and annual impact assessments if both are deemed applicable.
        3.  Seek clarification from the environmental regulatory body regarding the scope and frequency of required reports.
        """
        simulated_risk_score = 0.4 + random.uniform(0, 0.25)
        simulated_recommendation = "Adhere to quarterly reporting, consider submitting both, and seek clarification from regulators."
    else:
        simulated_llm_response = f"""
        **Summary of Legal Provisions:**
        The retrieved documents provide general guidance on {query.lower()}. They emphasize responsible practices and adherence to established regulations.

        **Identified Obligations and Prohibitions:**
        No explicit, direct contradictions or highly specific obligations/prohibitions were identified across the top relevant documents for this general query.

        **Compliance Risk Assessment:** Low - {0.1 + random.uniform(0, 0.2):.2f}

        **Reasoning for Risk:**
        Based on the provided context, the legal provisions related to your query appear broadly consistent and do not present immediate, obvious conflicts. The risk is considered low, indicating general clarity.

        **Legal Recommendations:**
        1.  Continue to monitor for any new regulations or amendments related to {query.lower()}.
        2.  Ensure regular training for personnel on general compliance principles.
        3.  Periodically review internal policies to align with best practices.
        """
        simulated_risk_score = 0.1 + random.uniform(0, 0.2)
        simulated_recommendation = "Monitor new regulations, conduct regular training, and review internal policies."

    # --- End of Placeholder Simulation ---

    return simulated_llm_response, simulated_risk_score, simulated_recommendation

# Logging setup
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;\'"():%/-]', '', text)
    return text.strip()

# --- Removed the old assess_compliance_risk function, as RAG will handle this ---
# def assess_compliance_risk(law1_text, law2_text, query=""):
#     # ... (old keyword-based logic) ...
#     pass

def main():
    try:
        model_path = "trained_sbert_mininglaw_risk_aware_with_negatives_enhanced"
        if not os.path.exists(model_path):
            raise ValueError(f"Trained model not found at {model_path}")
        model = SentenceTransformer(model_path)
        logging.info(f"✅ Loaded trained model from '{model_path}'")
    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        messagebox.showerror("Model Error", f"Failed to load the Sentence Transformer model: {e}\nPlease ensure 'train.py' has been run successfully.")
        return

    client = MongoClient("mongodb://localhost:27017/")
    db = client["mining_law_db"]
    pdf_data_collection = db["pdf_data"]

    try:
        pdf_data = list(pdf_data_collection.find({"text": {"$exists": True}}))
        if not pdf_data:
            logging.warning("⚠ No data found in 'pdf_data'")
            messagebox.showwarning("Database Warning", "No data found in MongoDB collection 'pdf_data'. Please ensure your PDFs are processed.")
            client.close()
            return
        logging.info(f"📄 Loaded {len(pdf_data)} documents from MongoDB")
    except Exception as e:
        logging.error(f"❌ Failed to load data: {e}")
        messagebox.showerror("Database Error", f"Failed to load data from MongoDB: {e}")
        client.close()
        return

    try:
        all_texts = [clean_text(doc["text"]) for doc in pdf_data] # Clean text before embedding
        pdf_embeddings = model.encode(all_texts, convert_to_tensor=True, show_progress_bar=True)
        logging.info("✅ Embedded all law documents")
    except Exception as e:
        logging.error(f"❌ Error embedding documents: {e}")
        messagebox.showerror("Embedding Error", f"Failed to embed documents: {e}")
        client.close()
        return

    def search_laws_and_rag(query):
        try:
            query_embedding = model.encode(clean_text(query), convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(query_embedding, pdf_embeddings)[0]
            top_results_torch = torch.topk(similarities, 3) # Retrieve top 3 documents

            results_text.delete(1.0, tk.END) # Clear previous results first

            top_scores = top_results_torch.values.tolist()
            top_indices = top_results_torch.indices.tolist()

            MIN_RELEVANCE_THRESHOLD = 0.125 # Increased threshold

            # If no results, or the top score is below the threshold, consider the query irrelevant.
            if not top_scores or top_scores[0] < MIN_RELEVANCE_THRESHOLD:
                results_text.insert(tk.END, "\n❌ Irrelevant question. The query does not match any relevant legal text with sufficient confidence.\n")
                return

            # If we reach here, the query is considered relevant enough to show retrieved documents.
            retrieved_docs_for_llm = []

            for i, idx in enumerate(top_indices):
                doc = pdf_data[idx]
                filename = doc.get("filename", "Unknown")
                text = clean_text(doc.get("text", ""))
                score = top_scores[i] # Score for the current document

                # We no longer need an inner irrelevance check for the overall query decision here,
                # as it was made based on top_scores[0]. We display all initially retrieved top_k documents.

                retrieved_docs_for_llm.append({
                    "filename": filename,
                    "text": text,
                    "score": score
                })
                results_text.insert(tk.END, f"\n🔹 Retrieved Document {i+1}:\n")
                results_text.insert(tk.END, f"📄 Document: {filename}\n")
                results_text.insert(tk.END, f"🔗 Similarity Score: {score:.4f}\n")
                results_text.insert(tk.END, f"📝 Excerpt (first 300 chars): {text[:300]}...\n")
                results_text.insert(tk.END, "-" * 60 + "\n")

            # The following check for empty retrieved_docs_for_llm is now largely redundant 
            # due to the upfront check on top_scores, but kept for safety against unexpected states.
            if not retrieved_docs_for_llm:
                # This case should ideally not be reached if top_scores[0] was valid.
                results_text.insert(tk.END, "\n❌ No documents were processed, though the query seemed initially relevant. Please check logs.\n")
                return

            results_text.insert(tk.END, "\n--- Generating LLM Response (RAG) ---\n")
            # Call the LLM with the query and retrieved context
            llm_full_response, llm_risk_score, llm_recommendation = generate_response_with_llm(query, retrieved_docs_for_llm)

            results_text.insert(tk.END, llm_full_response)
            results_text.insert(tk.END, f"\n\nOverall LLM-Assessed Risk Score: {llm_risk_score:.2f}\n")
            results_text.insert(tk.END, f"Final Legal Recommendation from LLM: {llm_recommendation}\n")
            results_text.insert(tk.END, "\n" + "=" * 80 + "\n")


        except Exception as e:
            logging.error(f"❌ RAG query error: {e}")
            results_text.insert(tk.END, f"\n❌ Error processing query with RAG: {e}\n")

    def on_search_button_click():
        query = query_entry.get().strip()
        if query:
            search_laws_and_rag(query)
        else:
            messagebox.showwarning("Input Error", "Please enter a query.")


    root = tk.Tk()
    root.title("Mining Law Compliance Chatbot (RAG)")

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
    header_label = Label(root, text="Mining Law Compliance Chatbot (RAG)", bg=bg_color, fg=text_color, font=header_font)
    header_label.pack(pady=(10, 20))

    # --- Query Input ---
    query_label = Label(root, text="🔍 Enter your mining law question:", bg=bg_color, fg=text_color, font=bold_font)
    query_label.pack(pady=(0, 5))

    query_entry = Entry(root, width=60, font=font_style, bg='white', fg=text_color)
    query_entry.pack(pady=(0, 10))
    query_entry.bind("<Return>", lambda event: on_search_button_click()) # Bind Enter key

    search_button = Button(root, text="Search", command=on_search_button_click, bg=button_color, fg='white', font=bold_font, relief=tk.RAISED, borderwidth=2)
    search_button.pack(pady=(0, 15))
    search_button.config(highlightbackground=bg_color)

    # --- Results Output ---
    results_label = Label(root, text="📋 RAG Response (Retrieval & LLM Generation):", bg=bg_color, fg=text_color, font=bold_font)
    results_label.pack()

    results_text = scrolledtext.ScrolledText(root, width=90, height=30, bg='#F9F9F9', fg=text_color, font=font_style, wrap=tk.WORD) # wrap=tk.WORD for better text display
    results_text.pack(pady=(0, 20))

    # --- Center Window ---
    root.eval('tk::PlaceWindow . center')

    # --- Run Main Loop ---
    root.mainloop()
    client.close()

if __name__ == "__main__":
    main()
