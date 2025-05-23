from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import logging
import sys
import warnings
import torch
import random
import re
import numpy as np
import spacy
from typing import List, Dict, Tuple

# Load a spaCy model (you might need to download one, e.g., 'en_core_web_sm')
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("✅ spaCy model loaded successfully.")
except OSError:
    logging.error("❌ Could not load spaCy model. Downloading 'en_core_web_sm'...")
    try:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        logging.info("✅ spaCy model downloaded and loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Error downloading or loading spaCy model: {e}")
        nlp = None  # Handle the case where spaCy fails to load
except Exception as e:
    logging.error(f"❌ Error loading spaCy model: {e}")
    nlp = None

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)])

def clean_text(text: str) -> str:
    """Cleans text by removing extra whitespace and some special characters."""
    text = re.sub(r'\s+', ' ', text).strip()
    # You might want to be more selective with char removal depending on legal text specifics
    # text = re.sub(r'[^\w\s.,;\'"():%/-]', '', text)
    return text

def create_training_examples(collection_name="pdf_data", question_templates=None, num_negative_samples=3) -> List[InputExample]:
    """
    Creates InputExample pairs from data in a MongoDB collection, focusing on passage-level
    chunking and generating diverse positive and negative examples for RAG.

    Args:
        collection_name (str): The name of the MongoDB collection.
        question_templates (dict): A dictionary of question templates for different risk levels.
        num_negative_samples (int): The number of negative examples to generate for each positive example.

    Returns:
        List[InputExample]: A list of InputExample objects, or an empty list on error.
    """
    examples = []
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["mining_law_db"]
        collection = db[collection_name]

        all_documents_data = list(collection.find({"text": {"$exists": True}, "filename": {"$exists": True}}))
        logging.info(f"Loaded {len(all_documents_data)} documents from MongoDB collection '{collection_name}'")

        if not all_documents_data:
            logging.warning("No documents found in the database. Exiting training example creation.")
            return []

        # Default question templates (can be customized)
        if question_templates is None:
            question_templates = {
                "high_risk": "What are the potential compliance risks associated with {topic} in the document '{filename}'?",
                "medium_risk": "Analyze {topic} in document '{filename}' for varying interpretations.",
                "low_risk": "Summarize the key provisions related to {topic} in document '{filename}'.",
                "simple": "What is the main purpose of the legal text regarding {topic} in '{filename}'?",
                "irrelevant": [
                    "What is the current stock market trend?",
                    "Tell me about the history of space travel.",
                    "How do I cook pasta?",
                    "What's the best movie of all time?",
                    "Explain quantum physics simply.",
                ]
            }

        all_chunks = []
        for doc in all_documents_data:
            filename = doc["filename"]
            full_text = clean_text(doc["text"])
            
            # Simple paragraph-based chunking
            # You could also implement token-based chunking with overlap for more control
            # using libraries like langchain's RecursiveCharacterTextSplitter
            chunks = [clean_text(para) for para in full_text.split('\n\n') if clean_text(para)]
            for i, chunk in enumerate(chunks):
                all_chunks.append({"filename": filename, "chunk_text": chunk, "doc_idx": all_documents_data.index(doc), "chunk_idx_in_doc": i})

        if not all_chunks:
            logging.warning("No valid chunks created from documents. Exiting training example creation.")
            return []

        for i, current_chunk_info in enumerate(all_chunks):
            filename = current_chunk_info["filename"]
            chunk_text = current_chunk_info["chunk_text"]
            doc_idx = current_chunk_info["doc_idx"]
            chunk_idx_in_doc = current_chunk_info["chunk_idx_in_doc"]

            # Heuristic to assign a 'topic' or general subject to the chunk
            # This is a very basic example; a real-world scenario might use keyword extraction
            # or a more sophisticated NLP model to identify the core topic.
            topic = "mining regulations"
            if "environmental" in chunk_text.lower() or "emission" in chunk_text.lower():
                topic = "environmental protection"
            elif "license" in chunk_text.lower() or "permit" in chunk_text.lower():
                topic = "licensing procedures"
            elif "safety" in chunk_text.lower() or "accident" in chunk_text.lower():
                topic = "mine safety"
            elif "tax" in chunk_text.lower() or "royalty" in chunk_text.lower():
                topic = "taxation and royalties"

            # Assign a risk level based on keywords in the chunk
            risk_level = "simple"
            if any(k in chunk_text.lower() for k in ["shall not", "must not", "prohibited", "illegal", "violation"]):
                risk_level = "high_risk"
            elif any(k in chunk_text.lower() for k in ["except", "provided that", "may", "should", "guideline"]):
                risk_level = "medium_risk"
            elif any(k in chunk_text.lower() for k in ["procedure", "define", "policy", "framework"]):
                risk_level = "low_risk"

            # Create positive example: question with the relevant chunk
            question_template = question_templates.get(risk_level, question_templates["simple"])
            question = question_template.format(topic=topic, filename=filename)
            examples.append(InputExample(texts=[question, chunk_text], label=1.0))

            # --- Generate Negative Examples ---
            num_neg_per_type = num_negative_samples // 2 if num_negative_samples > 1 else num_negative_samples

            # 1. Negative samples from OTHER DOCUMENTS (random chunks)
            other_doc_chunks = [
                c for c in all_chunks
                if c["doc_idx"] != doc_idx # Exclude chunks from the same document
            ]
            if other_doc_chunks:
                negative_other_doc_chunks = random.sample(other_doc_chunks, min(num_neg_per_type, len(other_doc_chunks)))
                for neg_chunk_info in negative_other_doc_chunks:
                    examples.append(InputExample(texts=[question, neg_chunk_info["chunk_text"]], label=0.0))

            # 2. Negative samples from THE SAME DOCUMENT (different chunks)
            same_doc_other_chunks = [
                c for c in all_chunks
                if c["doc_idx"] == doc_idx and c["chunk_idx_in_doc"] != chunk_idx_in_doc # Exclude the current chunk
            ]
            if same_doc_other_chunks and num_neg_per_type > 0:
                negative_same_doc_chunks = random.sample(same_doc_other_chunks, min(num_neg_per_type, len(same_doc_other_chunks)))
                for neg_chunk_info in negative_same_doc_chunks:
                    examples.append(InputExample(texts=[question, neg_chunk_info["chunk_text"]], label=0.0))
            
            # 3. Irrelevant questions paired with the current positive chunk
            if question_templates.get("irrelevant"):
                irrelevant_question = random.choice(question_templates["irrelevant"])
                examples.append(InputExample(texts=[irrelevant_question, chunk_text], label=0.0))

        return examples

    except Exception as e:
        logging.error(f"Error creating training examples: {e}", exc_info=True)
        return []
    finally:
        if 'client' in locals() and client:
            client.close()

def main():
    """
    Main function to load data from MongoDB, create training examples (including negative samples),
    train a SentenceTransformer model, and save it.
    """
    # Step 1: Load data from MongoDB and create training examples (with negative samples)
    # The num_negative_samples here applies to the sum of 'other doc' and 'same doc' chunks,
    # plus an additional irrelevant question.
    train_examples = create_training_examples(collection_name="pdf_data", num_negative_samples=8)

    if not train_examples:
        logging.warning("No training examples generated. Check your MongoDB data and the create_training_examples function.")
        return

    logging.info(f"Generated {len(train_examples)} training examples.")

    # Step 2: Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Step 3: Define model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 4: Define loss (using MultipleNegativesRankingLoss)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Step 5: Train the model
    logging.info("Starting model training...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=175,
            warmup_steps=len(train_dataloader) // 10,
            show_progress_bar=True,
            output_path="trained_sbert_m ininglaw_risk_aware_with_negatives", # Model is saved automatically here
            evaluation_steps=0, # Disabled periodic evaluation
            save_best_model=False, # Save final model, not "best" based on eval
            # Use a small development set for evaluation during training if possible
            # evaluator=None # You could add an evaluator here if you have a dev set
        )
    logging.info("✅ Model training complete.")

    # Model is automatically saved by model.fit if output_path and save_best_model=True are used.
    # No need for an explicit save() call unless you want to save to a different path or always.
    logging.info("Model saved to 'trained_sbert_mininglaw_risk_aware_with_negatives' (best model during training).")


if __name__ == "__main__":
    main()
