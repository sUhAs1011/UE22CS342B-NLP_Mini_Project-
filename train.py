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

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)])

def create_examples_from_mongodb(collection_name="pdf_data") -> List[InputExample]:
    """
    Creates InputExample pairs from data in a MongoDB collection.
    The collection is expected to have documents with 'filename' and 'text' fields.
    Each document's text is paired with a question about its content.

    Args:
        collection_name (str, optional): The name of the MongoDB collection to load data from.
            Defaults to "pdf_data".

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
        documents = list(collection.find())
        logging.info(f"Loaded {len(documents)} documents from MongoDB collection '{collection_name}'")

        for doc in documents:
            filename = doc.get("filename")
            text = doc.get("text")
            if filename and text:
                # Create a question and answer pair for each document.
                question = f"What is the content of the document '{filename}'?  What are the compliance risks associated with this document?"
                examples.append(InputExample(texts=[question, text], label=1.0))
            else:
                logging.warning(f"Skipping document with missing filename or text: {doc}")
        return examples

    except Exception as e:
        logging.error(f"Error loading data from MongoDB: {e}")
        return []  # Return an empty list on error to avoid crashing the training.
    finally:
        client.close()



def main():
    """
    Main function to load data from MongoDB, create training examples,
    train a SentenceTransformer model, evaluate its ability to identify contradicting information,
    and save it.
    """
    # Step 1: Load data from MongoDB
    train_examples = create_examples_from_mongodb(collection_name="pdf_data")

    if not train_examples:
        logging.warning("No training examples generated.  Check the data in your MongoDB collection.")
        return

    # Step 2: Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)  # Changed batch size to 1

    # Step 3: Define model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 4: Define loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Step 5: Train the model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=20,
            warmup_steps=len(train_dataloader) // 10,
            show_progress_bar=True
        )

    # Step 6: Save the model
    try:
        model.save("trained_sbert_mininglaw")
        logging.info("✅ Model saved to 'trained_sbert_mininglaw'")
    except Exception as e:
        logging.error(f"Error saving model: {e}")



if __name__ == "__main__":
    main()
