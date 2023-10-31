import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent, create_retriever_tool)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema.messages import SystemMessage
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.agents import initialize_agent, AgentType

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

def get_conversation_agent(vectorstore):

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=CohereRerank(),
        base_retriever=vectorstore.as_retriever(search_kwargs={'k': 20})
    )
    qdrant_mm_retrival_tool = create_retriever_tool(
        compression_retriever,
        "search_multimodality",
        "Searches and returns documents regarding multimodality.",
    )
    tools = [qdrant_mm_retrival_tool]

    llm = ChatOpenAI(model="gpt-4")
    # agent_executor = create_conversational_retrieval_agent(
    #     llm,
    #     tools,
    #     system_message=SystemMessage(
    #         content="""
    #             You are a conversational assistant that has access to the tool called search_multimodality, this tool allows you to search and retrieve documents related to multimodality or anything that you're not sure about.
    #             If you don't have and answer, try invoking this tool. If user asks for something that seems related to multimodality always first use the tool."""),
    #     remember_intermediate_steps=True,
    #     verbose=True
    # )

    react_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return react_agent

    

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multimodality documents 📊📚",)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Chat with Multimodality documents :books:")

    vectorstore = get_vector_store()
    st.session_state.conversation_agent = get_conversation_agent(vectorstore)

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
        response = st.session_state.conversation_agent({'input': user_question})
        assistant_answer = response['output']

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