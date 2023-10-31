import logging
import os

from docx import Document
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

# Set up logging
logging.basicConfig(filename='./logs/qdrant_vectorstore.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def read_docx_from_folder(folder_path):
    """
    Reads all .docx files in the specified folder and concatenates their text.

    Each document's text is separated by a series of dashes for clarity.

    :param folder_path: Path to the folder containing .docx files.
    :return: A single string containing the combined text of all .docx files.
    """
    combined_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            logging.info(f"Processing file: {filename}")
            document = Document(file_path)
            for para in document.paragraphs:
                combined_text += para.text + "\n"
            combined_text += "\n" + "-" * 40 + "\n"
    return combined_text

def get_text_chunks(text):
    """
    Splits the provided text into smaller chunks based on the specified chunk size and overlap.

    :param text: The text to be split into chunks.
    :return: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        # length_function=lambda x: len(x.split())
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def partition_and_chunk_by_title(folder_path):
    """
    Partitions and chunks .docx documents in the specified folder.

    This function processes each .docx file within the given folder. It first partitions
    the document into sections based on titles or headings. Each section is then
    chunked into smaller parts. The function aggregates these chunks from all
    documents into a single list, suitable for uploading to a vector database.

    :param folder_path: Path to the folder containing .docx files.
    :return: A list of text chunks from all the partitioned and chunked documents.
    """
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            logging.info(f"Processing file: {filename}")
            elements = partition(filename=file_path)
            document_chunks = chunk_by_title(elements)
            chunks.extend(str(d) for d in document_chunks)

    return chunks

        
def get_vector_store():
    """
    Initializes and returns a Qdrant vector store client.

    Configuration for the client is pulled from environment variables.

    :return: Initialized Qdrant vector store client.
    """
    client = QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store

def reset_collection(vector_store, collection_name):
    """
    Deletes and recreates a collection in Qdrant to ensure it's empty before adding new data.

    :param vector_store: The Qdrant vector store client.
    :param collection_name: The name of the collection to reset.
    """
    # Delete the existing collection
    vector_store.client.delete_collection(collection_name)
    logging.info(f"Deleted collection '{collection_name}'.")

    # Recreate the collection
    vector_store.client.create_collection(
        collection_name,
        vectors_config=models.VectorParams(
        size=1536,  # For OpenAI Embeddings
        distance=models.Distance.COSINE
    ))
    logging.info(f"Recreated collection '{collection_name}'.")

def main():
    """
    Main function to process .docx files and add their text to a Qdrant vector store.

    The function reads all .docx files from a specified directory, splits the text into chunks,
    and adds these chunks as texts to a Qdrant vector store. The collection is cleared before
    adding new texts to avoid duplication.
    """
    load_dotenv()

    path_to_documents = "./documents/"

    # Naive chunking...
    # all_document_texts = read_docx_from_folder(path_to_documents)
    # text_chunks = get_text_chunks(all_document_texts)

    # Much smarter chunking and reads Tables!
    text_chunks = partition_and_chunk_by_title(path_to_documents)

    vector_store = get_vector_store()

    # Clear the collection before adding new documents
    reset_collection(vector_store, os.getenv("QDRANT_COLLECTION_NAME"))
    logging.info(f"Cleared the collection '{os.getenv('QDRANT_COLLECTION_NAME')}'.")

    if text_chunks:
        vector_store.add_texts(text_chunks)
        logging.info(f"Added {len(text_chunks)} new text chunks to the vector store in collection '{os.getenv('QDRANT_COLLECTION_NAME')}'.")
    else:
        logging.info("No new text chunks to add.")

if __name__ == "__main__":
    main()
