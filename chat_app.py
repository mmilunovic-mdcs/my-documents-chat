import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import QdrantClient
import os
import time

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

def get_conversation_chain(vectorstore):
    """
    Creates and returns a conversational retrieval chain.

    This function sets up a conversational retrieval chain using the ChatOpenAI model (specifically GPT-4),
    along with a memory buffer for storing and returning chat history. The retrieval chain is configured
    to use the provided vector store as its retriever.

    :param vectorstore: The vector store to be used as the retriever in the conversation chain.
    :return: A ConversationalRetrievalChain instance configured with a language model, retriever, and memory.
    """
    llm = ChatOpenAI(model="gpt-4")

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={'k': 20}
        ),
        memory=memory,
        verbose=True
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multimodality documents 📊📚",)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Chat with Multimodality documents :books:")

    vectorstore = get_vector_store()
    st.session_state.conversation = get_conversation_chain(vectorstore)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    user_question = st.chat_input("Ask a question about Multimodality documents:")
    if user_question:
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Get response from conversation chain
        response = st.session_state.conversation({'question': user_question})
        assistant_answer = response['answer']

        # Add bot response to chat history and display
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_answer})

if __name__ == '__main__':
    main()