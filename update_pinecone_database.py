import logging
import os

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
import pinecone 
import json 
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='./logs/pinecone.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load the saved modification times
try:
    with open('last_modification_times.json', 'r') as f:
        last_modification_times = json.load(f)
except FileNotFoundError:
    last_modification_times = {}


def get_modification_time(file_path):
    """
    Retrieves the last modification time of the specified file.

    :param file_path: Path to the file.
    :return: Last modification time of the file.
    """
    return os.path.getmtime(file_path)


def partition_and_chunk_by_title(folder_path, last_modification_times):
    """
    Partitions and chunks .docx documents by title in the specified folder. 
    Only processes new or modified files based on their last modification times.

    :param folder_path: Path to the folder containing .docx files.
    :param last_modification_times: Dictionary containing last modification times of files.
    :return: A dictionary with filenames as keys and their corresponding chunks as values.
    """
    chunks = {}
    for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
        if filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            current_modification_time = get_modification_time(file_path)
            # Check if file is new or modified
            if filename not in last_modification_times or last_modification_times[filename] != current_modification_time:
                logging.info(f"Processing modified or new file: {filename}")

                if filename not in chunks:
                    chunks[filename] = []
                elements = partition(filename=file_path)
                document_chunks = chunk_by_title(elements)
                chunks[filename].extend((str(d) for d in document_chunks))
                last_modification_times[filename] = current_modification_time  # Update the last modification time
            else:
                logging.info(f"Skipping unchanged file: {filename}")
    return chunks

        
def get_vectorstore():
    """
    Connects and returns a Pinecone index instance using the index name from environment variables.

    :return: Pinecone index instance.
    """
    pinecone.init(      
	    api_key=os.getenv('PINECONE_API_KEY'),      
	    environment='gcp-starter'      
    )   
    vectorstore = pinecone.Index(index_name=os.getenv('PINECONE_INDEX_NAME'))
    return vectorstore


def upsert_vectors(vectorstore, embeddings, text_chunks):
    """
    Upserts the vectors to Pinecone based on changes in the files.

    :param vectorstore: Pinecone index instance.
    :param embeddings: Embedding model instance.
    :param text_chunks: Dictionary of text chunks.
    """

    for filename, lines in text_chunks.items():
        # If the file has been modified or is new
        ids_to_delete = [f"{filename}_{i}" for i in range(len(lines))]
        logging.info(f"Deleting vectors associated with: {filename}")
        vectorstore.delete(ids=ids_to_delete)

        # Now upsert the new chunks
        for i, line in enumerate(lines):
            vector_id = f"{filename}_{i}"
            vector = embeddings.embed_documents([line])[0]
            meta = {'document_name': filename, 'text': line}
            logging.info(f"Upserting vector with ID: {vector_id}")
            vectorstore.upsert(vectors=[(vector_id, vector, meta)])



def main():
    load_dotenv()

    path_to_documents = "./documents/"

    text_chunks = partition_and_chunk_by_title(path_to_documents, last_modification_times)

    vectorstore = get_vectorstore()

    embeddings = OpenAIEmbeddings()

    upsert_vectors(vectorstore, embeddings, text_chunks, last_modification_times)

    # Save the updated modification times
    with open('last_modification_times.json', 'w') as f:
        json.dump(last_modification_times, f)

if __name__ == "__main__":
    main()
