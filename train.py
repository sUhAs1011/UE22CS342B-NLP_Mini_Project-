from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler, util
from torch.utils.data import DataLoader
import logging
import sys
import warnings
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard
import torch
from sklearn.metrics import accuracy_score  # Import accuracy_score
from typing import List, Dict, Tuple
import re
import random

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)])

def create_training_examples_original(collection_name="pdf_data", question_templates=None, num_negative_samples=2) -> List[InputExample]:
    """
    Creates InputExample pairs from data in a MongoDB collection.
    """
    examples = []
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/")  # Adjust connection string if needed
        db = client["mining_law_db"]  # Use your database name
        collection = db[collection_name]

        # Fetch all documents from the collection
        all_documents = list(collection.find())
        logging.info(f"Loaded {len(all_documents)} documents from MongoDB collection '{collection_name}'")

        # Default question templates
        if question_templates is None:
            question_templates = {
                "high_risk": "What are the potential compliance risks associated with the content in the document '{filename}'?",
                "medium_risk": "Analyze the document '{filename}' for compliance requirements.",
                "low_risk": "Provide a summary of the key provisions in the document '{filename}' related to [specific topic].",
                "simple": "What is the main purpose of the document '{filename}'?",
            }

        for i, doc in enumerate(all_documents):
            filename = doc.get("filename")
            text = doc.get("text")
            if filename and text:
                # Assign a risk level (Original logic)
                if "shall not" in text.lower() or "must not" in text.lower() or "prohibited" in text.lower():
                    risk_level = "high_risk"
                elif "except" in text.lower() or "provided that" in text.lower():
                    risk_level = "medium_risk"
                elif "procedure" in text.lower() or "define" in text.lower():
                    risk_level = "low_risk"
                else:
                    risk_level = "simple"

                # Create positive example
                question = question_templates.get(risk_level, question_templates["simple"]).format(filename=filename)
                examples.append(InputExample(texts=[question, text], label=1.0))

                # Generate negative examples by pairing the question with text from other documents
                other_documents = [d for j, d in enumerate(all_documents) if i != j and d.get("text")]
                if other_documents:
                    negative_samples = random.sample(other_documents, min(num_negative_samples, len(other_documents)))
                    for neg_doc in negative_samples:
                        negative_text = neg_doc["text"]
                        examples.append(InputExample(texts=[question, negative_text], label=0.0))
                else:
                    logging.warning(f"Not enough other documents to generate negative samples for '{filename}'.")
            else:
                logging.warning(f"Skipping document with missing filename or text: {doc}")
        return examples

    except Exception as e:
        logging.error(f"Error loading data from MongoDB: {e}")
        return []  # Return an empty list on error
    finally:
        if 'client' in locals() and client:
            client.close()

def create_training_examples_enhanced(collection_name="pdf_data", question_templates=None, num_negative_samples=3) -> List[InputExample]:
    """
    Creates InputExample pairs from data in a MongoDB collection, with enhanced negative sampling
    and irrelevant question generation.
    """
    examples = []
    try:
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/")  # Adjust connection string if needed
        db = client["mining_law_db"]  # Use your database name
        collection = db[collection_name]

        # Fetch all documents from the collection
        all_documents = list(collection.find({"text": {"$exists": True}, "filename": {"$exists": True}}))
        logging.info(f"Loaded {len(all_documents)} valid documents from MongoDB collection '{collection_name}'")

        if not all_documents:
            logging.warning("No valid documents found in the collection.")
            return []

        # Default question templates (expanded)
        if question_templates is None:
            question_templates = {
                "high_risk": "What are the critical compliance obligations and potential risks outlined in the document '{filename}', considering recent legal updates?",
                "medium_risk": "Explain the key regulatory requirements and interpretations within the document '{filename}' that mining companies must adhere to.",
                "low_risk": "Summarize the specific guidelines and procedures detailed in '{filename}' concerning [specific topic, e.g., environmental permits].",
                "simple": "What is the primary subject or purpose of the legal document '{filename}'?",
                "irrelevant": [
                    "What is the weather forecast for tomorrow?",
                    "Who won the last football match?",
                    "What is the capital of France?",
                    "Tell me a joke about mining.",
                    "How to bake a cake?",
                ]
            }

        for i, doc in enumerate(all_documents):
            filename = doc["filename"]
            text = doc["text"]

            # Assign a risk level (Improved logic)
            if "shall not" in text.lower() or "must not" in text.lower() or "prohibited" in text.lower():
                risk_level = "high_risk"
            elif "except" in text.lower() or "provided that" in text.lower() or "means" in text.lower():
                risk_level = "medium_risk"
            elif "procedure" in text.lower() or "define" in text.lower() or "policy" in text.lower() :
                risk_level = "low_risk"
            else:
                risk_level = "simple"

            # Create positive example
            question = question_templates.get(risk_level, question_templates["simple"]).format(filename=filename)
            examples.append(InputExample(texts=[question, text], label=1.0))

            # Generate negative examples from other documents
            other_relevant_docs = [d for j, d in enumerate(all_documents) if i != j]
            if other_relevant_docs:
                negative_samples = random.sample(other_relevant_docs, min(num_negative_samples, len(other_relevant_docs)))
                for neg_doc in negative_samples:
                    negative_text = neg_doc["text"]
                    examples.append(InputExample(texts=[question, negative_text], label=0.0))
            else:
                logging.warning(f"Not enough other relevant documents for negative sampling for '{filename}'.")

            # Generate negative examples with irrelevant questions
            num_irrelevant = 1  # Generate a few irrelevant questions per document
            for _ in range(num_irrelevant):
                irrelevant_question = random.choice(question_templates["irrelevant"])
                examples.append(InputExample(texts=[irrelevant_question, text], label=0.0))

        return examples

    except Exception as e:
        logging.error(f"Error loading data from MongoDB: {e}")
        return []  # Return an empty list on error
    finally:
        if 'client' in locals() and client:
            client.close()


def assess_compliance_risk(law1_text, law2_text, query=""):
    """
    Assesses the compliance risk based on potential contradictions between two laws.
    (Retaining the original assess_compliance_risk function)
    """
    risk_score = 0.0
    explanation = "No significant conflict detected."
    contradiction_keywords = [
        ("shall not", "shall"),
        ("must not", "must"),
        ("is prohibited", "is allowed"),
        ("should", "is not required"),
        ("cannot", "can"),
        ("no person may", "any person may"),
        ("it is illegal", "it is legal"),
        ("mandatory", "optional"),
        ("required", "not required"),
        ("prohibits", "permits"),
    ]

    # 1. Keyword-based contradiction detection with context
    for phrase1, phrase2 in contradiction_keywords:
        if phrase1 in law1_text.lower() and phrase2 in law2_text.lower():
            # Check if they are in the same sentence or nearby sentences.
            sentences1 = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+', law1_text)
            sentences2 = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+', law2_text)

            for sent1 in sentences1:
                if phrase1 in sent1.lower():
                    for sent2 in sentences2:
                        if phrase2 in sent2.lower():
                            if abs(sentences1.index(sent1) - sentences2.index(sent2)) <= 2:
                                risk_score = 0.7 + random.uniform(0, 0.3)  # 0.7 to 1.0
                                explanation = f"High potential conflict regarding '{query}': Direct conflict between '{phrase1}' and '{phrase2}' within close proximity. Legal recommendation: Mining companies should adhere to the more stringent requirement."
                                return risk_score, explanation

    # 2. Check for conflicting conditions or exceptions
    if "except" in law1_text.lower() and "provided that" in law2_text.lower():
        risk_score = 0.5 + random.uniform(0, 0.2)  # 0.5 to 0.7
        explanation = f"Moderate potential conflict regarding '{query}': One law states a general rule, while the other provides an exception or condition that may contradict it. Legal recommendation: Mining companies should carefully examine the specific conditions and ensure they meet both the general rule and the exception."
        return risk_score, explanation

    # 3. Check for conflicting definitions
    if "means" in law1_text.lower() and "means" in law2_text.lower():
        # Extract the word/phrase being defined
        defined_word1 = law1_text.lower().split("means")[0].strip().split()[-1]
        defined_word2 = law2_text.lower().split("means")[0].strip().split()[-1]
        if defined_word1 == defined_word2:
            risk_score = 0.4 + random.uniform(0, 0.3)
            explanation = f"Moderate potential conflict regarding '{query}': Conflicting definitions of the term '{defined_word1}'. Legal recommendation: Mining companies should seek clarification from regulatory authorities to determine the legally binding definition."
            return risk_score, explanation

    # 4. Check for temporal conflicts (e.g., amendments)
    if "amendment" in law1_text.lower() and "original act" in law2_text.lower():
        risk_score = 0.3 + random.uniform(0, 0.2)
        explanation = f"Low potential conflict regarding '{query}': One law is an amendment to the other, which may lead to conflicts in application. Legal recommendation: Mining companies should prioritize compliance with the most recent amendment."
        return risk_score, explanation

    return risk_score, explanation


def main():
    """
    Main function to load data from MongoDB, create enhanced training examples (including hard negatives
    and irrelevant questions), train a SentenceTransformer model, and save it.
    """
    # Step 1: Load data from MongoDB and create enhanced training examples
    train_examples = create_training_examples_enhanced(collection_name="pdf_data", num_negative_samples=3)

    if not train_examples:
        logging.warning("No training examples generated. Check the data in your MongoDB collection.")
        return

    # Step 2: Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32) # Increased batch size

    # Step 3: Define model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 4: Define loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Step 5: Train the model
    num_epochs = 200
    warmup_steps = len(train_dataloader) // 10

    logging.info(f"Starting training for {num_epochs} epochs with {len(train_dataloader)} steps per epoch and {warmup_steps} warmup steps.")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True
        )

    # Step 6: Save the model
    try:
        model.save("trained_sbert_mininglaw_risk_aware_with_negatives_enhanced")
        logging.info("✅ Enhanced model saved to 'trained_sbert_mininglaw_risk_aware_with_negatives_enhanced'")
    except Exception as e:
        logging.error(f"Error saving enhanced model: {e}")


if __name__ == "__main__":
    main()
