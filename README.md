# Chat with your documents

## About
The [**My Documents Chat**](https://github.com/mmilunovic-mdcs/my-documents-chat) is a Python-based project focused on creating chat applications that interact with users and retrieve information from documents. It utilizes several key technologies and libraries:

- **[Qdrant](https://github.com/qdrant/qdrant)** 🗃️ - A vector search engine for storing and retrieving document vectors.

- **[LangChain](https://github.com/LangChain/langchain)** 🦜️🔗 - Used for building conversational agents with language models like GPT-4.

- **[Unstructured](https://github.com/unstructuredai/unstructured)** 📑 - Handles unstructured data, aiding in document processing.

- **[Streamlit](https://github.com/streamlit/streamlit)** 🌟 - Powers the user interface, making Python scripts interactive and web-friendly.

The project uses **CohereRerank** for better document retrieval, based on ideas from the [**CohereRerank paper**](https://txt.cohere.com/rerank/). This improves how accurately and relevantly documents match user questions.

The chat apps in this repository are also shaped by ideas from the [**ReAct paper**](https://arxiv.org/abs/2210.03629). These ideas help make the chats respond more relevantly and accurately.


## Installation and Requirements
To get started with the applications in this repository, follow these steps:

1. **Clone the Repository**:  
   Clone the repository to your local machine using:
   ```sh
   git clone https://github.com/mmilunovic-mdcs/my-documents-chat.git
   ```

2. **Install Dependencies**:  
   Navigate to the cloned repository directory and install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   Key libraries include `langchain`, `streamlit`, `python-dotenv`, and `qdrant_client`.

## Running the Update Qdrant Database Script
To update the Qdrant vector store with new document chunks, run the `update_qdrant_database.py` script. This script processes `.docx` files, partitions, and chunks the documents for vector storage.

1. **Set Environment Variables**:  
   Ensure that the necessary environment variables are set in your environment or a `.env` file. These include `QDRANT_HOST`, `QDRANT_API_KEY`,`QDRANT_COLLECTION_NAME`, `OPENAI_API_KEY` and `COHERE_API_KEY`.

2. **Run the Script**:  
   Execute the script by running:
   ```sh
   python update_qdrant_database.py
   ```
   The script will read `.docx` files from a specified directory, process them, and update the Qdrant vector store.

## Running the Agent Chat App with Streamlit
To run the `agent_chat_app.py` using Streamlit:

1. **Start the Streamlit Server**:  
   Run the following command in your terminal:
   ```sh
   streamlit run agent_chat_app.py
   ```

2. **Interact with the App**:  
   Once the Streamlit server is running, it will open a web interface in your default browser. Here, you can interact with the chat application, asking questions and receiving responses based on the multimodality documents in your Qdrant vector store.

## Next Steps

Here's a checklist of upcoming enhancements and features for the **My Documents Chat** project:

- [ ] **Images as Input**: Implement functionality to accept and process image-based inputs in the chat applications.

- [ ] **Daily Script Run**: Set up a system to automatically run certain scripts daily, ensuring up-to-date data and functionality.

- [ ] **Smarter Database Update with Hashing**: Improve the database update process by incorporating hashing techniques. This will help in identifying and managing changes more efficiently.

- [ ] **Caching for LLM Responses**: Implement caching mechanisms for responses from large language models (LLMs) to speed up response times and reduce API calls.