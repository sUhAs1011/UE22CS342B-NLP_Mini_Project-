from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
import torch
import logging
import sys
import os
import re
import random
import tkinter as tk
from tkinter import scrolledtext, Button, Entry, Label

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)])

def clean_text(text):
    """
    Cleans the text by normalizing whitespace and removing special characters
    while preserving those relevant to legal text.
    """
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s.,;\'"():%/-]', '', text)  # Remove special characters, keep legal symbols
    return text.strip()

def assess_compliance_risk(law1_text, law2_text, query=""):
    """
    Assesses the compliance risk based on potential contradictions between two laws.
    """
    risk_score = 0.0
    explanation = "No significant conflict detected."

    if "shall not" in law1_text and "shall" in law2_text:
        risk_score = 0.6 + random.uniform(0, 0.2)
        explanation = f"Potential conflict regarding '{query}': One law prohibits an action that the other permits."
    elif "must" in law1_text and "may not" in law2_text:
        risk_score = 0.6 + random.uniform(0, 0.2)
        explanation = f"Potential conflict regarding '{query}': One law mandates an action that the other forbids."
    elif "is prohibited" in law1_text and "is allowed" in law2_text:
        risk_score = 0.9
        explanation = f"High conflict regarding '{query}': Direct contradiction between prohibition and allowance."
    elif "should" in law1_text and "is not required" in law2_text:
        risk_score = 0.3 + random.uniform(0, 0.2)
        explanation = f"Potential conflict regarding '{query}': One law recommends an action that the other does not require"

    return risk_score, explanation

def main():
    """
    Main function to load the trained SBERT model, connect to MongoDB,
    and provide a search interface for mining laws.
    """
    # Step 1: Load trained SBERT model
    try:
        model_path = "trained_sbert_mininglaw"
        if not os.path.exists(model_path):
            raise ValueError(f"Trained model not found at {model_path}.  Make sure you have trained the model and that the path is correct.")
        model = SentenceTransformer(model_path)
        logging.info(f"Loaded trained SBERT model from '{model_path}'")
    except Exception as e:
        logging.error(f"Error loading the trained SBERT model: {e}")
        return

    # Step 2: Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["mining_law_db"]
    pdf_data_collection = db["pdf_data"]

    # Step 3: Load data from MongoDB
    try:
        pdf_data = list(pdf_data_collection.find())
        if not pdf_data:
            logging.warning("No data found in the 'pdf_data' collection in MongoDB.")
            client.close()
            return
        logging.info(f"Loaded {len(pdf_data)} documents from MongoDB")
    except Exception as e:
        logging.error(f"Error loading data from MongoDB: {e}")
        client.close()
        return

    # Step 4: Encode data
    try:
        pdf_embeddings = model.encode(
            [item["text"] for item in pdf_data],
            convert_to_tensor=True,
            show_progress_bar=True
        )
        logging.info("Encoded data from MongoDB successfully.")
    except Exception as e:
        logging.error(f"Error encoding data: {e}")
        client.close()
        return

    # Step 5: Search function
    def search_laws(query):
        """
        Searches for the most similar document in the database based on a query,
        and also checks for potential contradictions with other retrieved laws.
        """
        try:
            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(query_embedding, pdf_embeddings)[0]
            top_results = torch.topk(similarities, 3)
            top_indices = top_results.indices.tolist()
            top_scores = top_results.values.tolist()

            results_text.delete(1.0, tk.END)  # Clear previous results

            for i, index in enumerate(top_indices):
                filename = pdf_data[index].get('filename', "Filename not found")
                text = pdf_data[index].get('text', "Text not found")
                score = top_scores[i]
                logging.info(f"Match {i+1} found in file: {filename} with score: {score:.4f}")
                results_text.insert(tk.END, f"\n🔍 Match {i+1} for your query: '{query}'\n")
                results_text.insert(tk.END, f"📄 Filename: {filename}\n")
                results_text.insert(tk.END, f"📝 Excerpt: {text[:200]}...\n")
                results_text.insert(tk.END, f"🔗 Similarity Score: {score:.4f}\n")
                results_text.insert(tk.END, "-" * 60 + "\n")

            if len(top_indices) > 1:
                contradictions_found = False
                for i in range(len(top_indices)):
                    for j in range(i + 1, len(top_indices)):
                        law1_text = pdf_data[top_indices[i]]['text']
                        law2_text = pdf_data[top_indices[j]]['text']
                        risk_score, explanation = assess_compliance_risk(law1_text, law2_text, query)
                        if risk_score > 0.0:
                            contradictions_found = True
                            logging.warning(
                                f"Potential contradiction detected between documents: {pdf_data[top_indices[i]].get('filename')} and {pdf_data[top_indices[j]].get('filename')}")
                            results_text.insert(tk.END, f"\n🚨 Potential Compliance Risk Detected:\n")
                            results_text.insert(tk.END, f"  Risk Score: {risk_score:.2f}\n")
                            results_text.insert(tk.END, f"  Explanation: {explanation}\n")
                            results_text.insert(tk.END, "-" * 60 + "\n")
                if not contradictions_found:
                    results_text.insert(tk.END, "\n✅ No contradictions detected among the top matching documents.\n")
                    results_text.insert(tk.END, "-" * 60 + "\n")

        except Exception as e:
            logging.error(f"Error during search: {e}")
            results_text.insert(tk.END, f"Error during search: {e}\n")

    # Step 6: Create GUI
    def on_search_button_click():
        query = query_entry.get()
        search_laws(query)

    root = tk.Tk()
    root.title("Mining Law Search")

    query_label = Label(root, text="Enter your mining law question:")
    query_label.pack(pady=10)

    query_entry = Entry(root, width=50)
    query_entry.pack(pady=10)

    search_button = Button(root, text="Search", command=on_search_button_click)
    search_button.pack(pady=10)

    results_label = Label(root, text="Search Results:")
    results_label.pack()

    results_text = scrolledtext.ScrolledText(root, width=60, height=20)
    results_text.pack(pady=10)

    #centering the main window
    root.eval('tk::PlaceWindow . center')

    root.mainloop()
    client.close()

if __name__ == "__main__":
    main()

