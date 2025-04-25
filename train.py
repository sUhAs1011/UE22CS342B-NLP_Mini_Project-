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
import numpy as np  # Import numpy
import spacy  # Import spaCy

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

def create_training_examples(collection_name="pdf_data", question_templates=None, num_negative_samples=2) -> List[InputExample]:
    """
    Creates InputExample pairs from data in a MongoDB collection, addressing
    deficiencies in handling simple and complex questions, compliance risks,
    and importantly, generating negative examples for irrelevant questions.

    Args:
        collection_name (str, optional): The name of the MongoDB collection.
            Defaults to "pdf_data".
        question_templates (dict, optional): A dictionary of question templates
            for different risk levels. If None, default templates are used.
        num_negative_samples (int, optional): The number of negative examples
            to generate for each positive example. Defaults to 2.

    Returns:
        List[InputExample]: A list of InputExample objects, or an empty list on error.
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
                "high_risk": "What are the potential compliance risks associated with the content in the document '{filename}', and what specific recommendations can ensure full compliance, including adherence to recent amendments and legal precedents?",
                "medium_risk": "Analyze the document '{filename}' for compliance requirements.  Identify areas where interpretations may vary, and outline steps to mitigate potential conflicts between different legal interpretations.",
                "low_risk": "Provide a summary of the key provisions in the document '{filename}' related to [specific topic, e.g., environmental regulations, licensing procedures].",
                "simple": "What is the main purpose of the document '{filename}'?",
            }

        for i, doc in enumerate(all_documents):
            filename = doc.get("filename")
            text = doc.get("text")
            if filename and text:
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


def assess_compliance_risk(law1_text, law2_text, query="", model=None):
    """
    Assesses the compliance risk based on potential contradictions between two laws,
    incorporating semantic similarity.

    Args:
        law1_text (str): The text of the first law.
        law2_text (str): The text of the second law.
        query (str, optional): The user's query.
        model (SentenceTransformer, optional):  Pre-trained SentenceTransformer model.

    Returns:
        tuple: (risk_score, explanation).
    """
    risk_score = 0.0
    explanation = "No significant conflict detected."

    if model is None:
        logging.warning("No SentenceTransformer model provided.  Using basic keyword checks only.")
        return assess_compliance_risk_keyword_based(law1_text, law2_text, query)

    # 1. Semantic Similarity
    embedding1 = model.encode(law1_text, convert_to_tensor=True)
    embedding2 = model.encode(law2_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()

    if similarity < 0.7:  # Adjust threshold as needed
        risk_score = 0.6 + (0.7 - similarity) * 0.4  # Scale risk score
        explanation = f"Moderate to high potential conflict regarding '{query}': Semantic analysis indicates significant differences between the two laws (similarity: {similarity:.2f}). Legal recommendation: Conduct a thorough review of both laws to identify specific areas of conflict."
        return risk_score, explanation

    # 2. Keyword-based check (as a fallback and to refine the analysis)
    keyword_risk, keyword_explanation = assess_compliance_risk_keyword_based(law1_text, law2_text, query)
    if keyword_risk > 0:
        risk_score = max(risk_score, keyword_risk)
        explanation = keyword_explanation + " (Further refined by keyword analysis.)"

    return risk_score, explanation


def assess_compliance_risk_keyword_based(law1_text, law2_text, query=""):
    """
    Assesses the compliance risk based on potential contradictions between two laws.
    This is a simplified example; a real-world implementation would require a much
    more sophisticated rule-based or ML-driven approach. The function is made more
    dynamic by incorporating the query into the explanation and adjusting the
    risk score range.

    Args:
        law1_text (str): The text of the first law.
        law2_text (str): The text of the second law.
        query (str, optional): The user's query. This is used to provide more context
            in the explanation. Defaults to "".

    Returns:
        tuple: (risk_score, explanation). Risk score is a float between 0 (no risk) and 1 (high risk).
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

def extract_entities(text):
    """
    Extracts named entities from the given text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of named entities.  Each entity is a tuple (text, label).
              Returns an empty list if spaCy is not loaded.
    """
    if nlp is None:
        logging.warning("spaCy model not loaded. NER will not be performed.")
        return []

    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def main():
    """
    Main function to load data from MongoDB, create training examples (including negative samples),
    train a SentenceTransformer model, and save it.
    """
    # Step 1: Load data from MongoDB and create training examples (with negative samples)
    train_examples = create_training_examples(collection_name="pdf_data", num_negative_samples=3) # Increased negative samples

    if not train_examples:
        logging.warning("No training examples generated. Check the data in your MongoDB collection.")
        return

    # Step 2: Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16) # Increased batch size for efficiency

    # Step 3: Define model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 4: Define loss (using MultipleNegativesRankingLoss - corrected name)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Step 5: Train the model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=175,
            warmup_steps=len(train_dataloader) // 10,
            show_progress_bar=True
        )

    # Step 6: Save the model
    try:
        model.save("trained_sbert_mininglaw_risk_aware_with_negatives")
        logging.info("✅ Model saved to 'trained_sbert_mininglaw_risk_aware_with_negatives'")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


    # Example usage (after training and saving the model):
    loaded_model = SentenceTransformer("trained_sbert_mininglaw_risk_aware_with_negatives")
    law1 = "This law prohibits mining within 100 meters of a river."
    law2 = "This law allows mining within 50 meters of a stream."
    query = "Mining near water bodies"
    risk_score, explanation = assess_compliance_risk(law1, law2, query, loaded_model)
    print(f"Risk Score: {risk_score:.2f}")
    print(f"Explanation: {explanation}")

    # NER Example
    example_text = "The MMDR Act applies to mining operations in Rajasthan and Gujarat."
    entities = extract_entities(example_text)
    print("\nNamed Entities:")
    for entity, label in entities:
        print(f"{entity} ({label})")


if __name__ == "__main__":
    main()
