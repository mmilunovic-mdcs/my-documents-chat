import os
import re
import time
from PIL import Image

import streamlit as st
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.vectorstores import Pinecone
from langchain.agents import initialize_agent, AgentType
from redisvl.llmcache.semantic import SemanticCache
import pinecone

from tools import TransformToTableTool, MermaidDiagramTool

def get_vector_store():
    """
    Initializes and returns a Qdrant vector store client.

    Configuration for the client is pulled from environment variables.

    :return: Initialized Qdrant vector store client.
    """
    embeddings = OpenAIEmbeddings()

    pinecone.init(      
	    api_key=os.getenv('PINECONE_API_KEY'),      
	    environment='gcp-starter'      
    )   
    index = pinecone.Index(index_name=os.getenv('PINECONE_INDEX_NAME'))
    
    vectorstore = Pinecone(index, embeddings, "text")

    return vectorstore

def get_conversation_agent(vectorstore):

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=CohereRerank(),
        base_retriever=vectorstore.as_retriever(search_kwargs={'k': 20})
    )
    vector_database_retrival_tool = create_retriever_tool(
        compression_retriever,
        "search_multimodality",
        "Searches and returns documents regarding multimodality.",
    )

    text2table_tool = TransformToTableTool()
    text2diagram_tool = MermaidDiagramTool()

    tools = [vector_database_retrival_tool, text2table_tool, text2diagram_tool]

    llm = ChatOpenAI(model="gpt-4")

    conversational_memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True,
        k=10
    )

    system_message = """
        You are a helpful assistant.
        You hava access to my work documents through tool called: search_multimodality
        You can use tool TransformToTable to transform some text into a markdown table
        You can use tool TransformToDiagram to transform some text into a diagram.
        After you use TransformToDiagram tool your final response to the user should be diagram.png

        You chat with me about my work and about other general topics.
        You can ask clarifying questions to help you understand my question.
    """

    react_agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        llm=llm,
        tools=tools, 
        memory=conversational_memory,
        agent_kwargs={"system_message": system_message},
        handle_parsing_errors=True,
        verbose=True
    )


    return react_agent

def get_cache():
    cache = SemanticCache(
        redis_url="redis://localhost:6379",
        threshold=0.7, # semantic similarity threshold
        ttl=86_400, # Time-to-Live in seconds
    )
    return cache
    

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multimodality documents 📊📚",)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Chat with Multimodality documents :books:")

    vectorstore = get_vector_store()
    st.session_state.conversation_agent = get_conversation_agent(vectorstore)
    st.session_state.cache = get_cache()

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

        # response = st.session_state.cache.check(user_question)
        response = False
        if response:
            assistant_answer = response[0]
        else:
            response = st.session_state.conversation_agent({'input': user_question})
            assistant_answer = response['output']
            # st.session_state.cache.store(user_question, assistant_answer)

        # Add bot response to chat history and display
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            diagrams_dir = './diagrams'
            diagram_files = [f for f in os.listdir(diagrams_dir) if f.endswith('.png')]

            if diagram_files:
                # Assuming you want to display the first .png file
                image_path = os.path.join(diagrams_dir, diagram_files[0])
                image = Image.open(image_path)
                st.image(image, caption="diagram")
                os.remove(image_path)
            else:
                full_response = ""
                # Simulate stream of response with milliseconds delay
                for chunk in assistant_answer.split(" "):
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_answer})

if __name__ == '__main__':
    main()