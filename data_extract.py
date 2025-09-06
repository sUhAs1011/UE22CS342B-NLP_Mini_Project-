import os
import PyPDF2
import logging
import sys
from pymongo import MongoClient  # Import the MongoDB client

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(stream=sys.stdout)])

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text, or None on error.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""  # Handle None extracted text
        logging.info(f"Successfully extracted text from {pdf_path}")
        return text
    except FileNotFoundError:
        logging.error(f"File not found: {pdf_path}")
        return None
    except PyPDF2.errors.PdfReadError:
        logging.error(f"Error reading PDF: {pdf_path}.  It might be corrupted or not a valid PDF.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while processing {pdf_path}: {e}")
        return None

def extract_text_from_all_pdfs(directory):
    """
    Extracts text from all PDF files in a directory.

    Args:
        directory (str): The path to the directory containing the PDF files.

    Returns:
        dict: A dictionary where keys are PDF file names and values are the
              extracted text, or None if there was an error.
    """
    pdf_texts = {}
    if not os.path.exists(directory):
        logging.error(f"Directory not found: {directory}")
        return {}

    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            if text is not None:  # Only add if extraction was successful
                pdf_texts[filename] = text
    return pdf_texts

def store_text_in_mongodb(data, collection_name="pdf_data"):
    """
    Stores the extracted text from PDF files into a MongoDB collection.

    Args:
        data (dict): A dictionary where keys are PDF file names and values are the extracted text.
        collection_name (str, optional): The name of the MongoDB collection to use.
            Defaults to "pdf_data".
    """
    try:
        # Connect to MongoDB (adjust the connection string as needed)
        client = MongoClient("mongodb://localhost:27017/")
        db = client["mining_law_db"]  # Use your database name
        collection = db[collection_name]

        # Prepare data for insertion
        documents = [{"filename": filename, "text": text} for filename, text in data.items()]

        if documents:
            # Insert the documents into the collection
            result = collection.insert_many(documents)
            logging.info(f"Successfully inserted {len(result.inserted_ids)} documents into MongoDB collection '{collection_name}'")
        else:
            logging.info("No data to insert into MongoDB.")

    except Exception as e:
        logging.error(f"An error occurred while connecting to or writing to MongoDB: {e}")
    finally:
        client.close()  # Ensure the connection is closed

def main():
    """
    Main function to extract text from PDF files in a specified directory
    and store the results in a MongoDB database.
    """
    pdf_directory = r"C:\Users\Admin\Downloads\Legalbot-master_1\datasets"  # Changed to raw string
    extracted_texts = extract_text_from_all_pdfs(pdf_directory)

    if extracted_texts:
        store_text_in_mongodb(extracted_texts) # Store to MongoDB
        for filename, text in extracted_texts.items():
            print(f"\n--- Extracted text from {filename} ---")
            print(text[:500] + ("..." if len(text) > 500 else ""))
    else:
        logging.info("No PDF files found in the directory or error occurred during processing.")

if __name__ == "__main__":
    main()
