from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
import torch
import logging
import sys
import os
import re
import random  # Import the random module

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
    This is a simplified example; a real-world implementation would require a much
    more sophisticated rule-based or ML-driven approach.  The function is made more
    dynamic by incorporating the query into the explanation and adjusting the
    risk score range.

    Args:
        law1_text (str): The text of the first law.
        law2_text (str): The text of the second law.
        query (str, optional): The user's query.  This is used to provide more context
            in the explanation. Defaults to "".

    Returns:
        tuple: (risk_score, explanation).  Risk score is a float between 0 (no risk) and 1 (high risk).
    """
    risk_score = 0.0
    explanation = "No significant conflict detected."

    # Very basic contradiction detection (for demonstration purposes)
    if "shall not" in law1_text and "shall" in law2_text:
        risk_score = 0.6 + random.uniform(0, 0.2)  # 0.6 to 0.8
        explanation = f"Potential conflict regarding '{query}': One law prohibits an action that the other permits."
    elif "must" in law1_text and "may not" in law2_text:
        risk_score = 0.6 + random.uniform(0, 0.2)  # 0.6 to 0.8
        explanation = f"Potential conflict regarding '{query}': One law mandates an action that the other forbids."
    elif "is prohibited" in law1_text and "is allowed" in law2_text:
        risk_score = 0.9
        explanation = f"High conflict regarding '{query}': Direct contradiction between prohibition and allowance."
    elif "should" in law1_text and "is not required" in law2_text:
        risk_score = 0.3 + random.uniform(0, 0.2)  # 0.3 to 0.5
        explanation = f"Potential conflict regarding '{query}': One law recommends an action that the other does not require"

    return risk_score, explanation

def main():
    """
    Main function to load the trained SBERT model, connect to MongoDB,
    and provide a search interface for mining laws, including contradiction
    detection and compliance risk assessment.
    """
    # Step 1: Load trained SBERT model
    try:
        model_path = "trained_sbert_mininglaw"  # Path to the saved model
        if not os.path.exists(model_path):
            raise ValueError(f"Trained model not found at {model_path}.  Make sure you have trained the model and that the path is correct.")
        model = SentenceTransformer(model_path)
        logging.info(f"Loaded trained SBERT model from '{model_path}'")
    except Exception as e:
        logging.error(f"Error loading the trained SBERT model: {e}")
        return

    # Step 2: Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")  # Or your MongoDB connection string
    db = client["mining_law_db"]
    pdf_data_collection = db["pdf_data"]  # Use the collection where you stored the PDF data

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
            [item["text"] for item in pdf_data],  # Encode the 'text' field from each document
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

        Args:
            query (str): The query string.
        """
        try:
            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(query_embedding, pdf_embeddings)[0]
            # Get the top 3 most relevant documents
            top_results = torch.topk(similarities, 3)  # Get top 3 results and their indices
            top_indices = top_results.indices.tolist()
            top_scores = top_results.values.tolist()

            for i, index in enumerate(top_indices):
                filename = pdf_data[index].get('filename', "Filename not found")
                text = pdf_data[index].get('text', "Text not found")
                score = top_scores[i]
                logging.info(f"Match {i+1} found in file: {filename} with score: {score:.4f}")
                print(f"\n🔍 Match {i+1} for your query: '{query}'\n")
                print(f"📄 Filename: {filename}")
                print(f"📝 Excerpt: {text[:200]}...")
                print(f"🔗 Similarity Score: {score:.4f}")
                print("-" * 60)

            if len(top_indices) > 1:  # Check for contradictions only if multiple documents are retrieved
                contradictions_found = False # added to track if any contradictions were found
                for i in range(len(top_indices)):
                    for j in range(i + 1, len(top_indices)):
                        law1_text = pdf_data[top_indices[i]]['text']
                        law2_text = pdf_data[top_indices[j]]['text']
                        risk_score, explanation = assess_compliance_risk(law1_text, law2_text, query)  # Pass the query
                        if risk_score > 0.0:
                            contradictions_found = True # set the flag
                            logging.warning(
                                f"Potential contradiction detected between documents: {pdf_data[top_indices[i]].get('filename')} and {pdf_data[top_indices[j]].get('filename')}")
                            print(f"\n🚨 Potential Compliance Risk Detected:")
                            print(f"  Risk Score: {risk_score:.2f}")
                            print(f"  Explanation: {explanation}")
                            print("-" * 60)
                if not contradictions_found: # print this message if no contradictions were found.
                    print("\n✅ No contradictions detected among the top matching documents.")
                    print("-" * 60)

        except Exception as e:
            logging.error(f"Error during search: {e}")

    # Step 6: Interactive search loop
    try:
        while True:
            query = input("\n💬 Ask a mining law question (or type 'exit'): ")
            if query.lower() in ["exit", "quit"]:
                break
            search_laws(query)
    except KeyboardInterrupt:
        logging.info("Search interrupted by user.")
    finally:
        client.close()  # Close the connection

if __name__ == "__main__":
    main()
