from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

def get_available_device():
    """
    Detect and return the best available device for model inference.
    Returns:
        str: 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' for CPU
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_chatbot_response(client, model_name, messages, temperature=0):
    # Initialize LangChain chat model with Groq
    chat_model = ChatGroq(
        model_name=os.getenv("GROQ_MODEL_NAME"),
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Convert messages to LangChain format
    lc_messages = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" 
        else AIMessage(content=msg["content"])
        for msg in messages
    ]
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coffee shop assistant. Provide clear and concise responses."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create chain
    chain = LLMChain(
        llm=chat_model,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    
    # Get response
    response = chain.predict(input=messages[-1]["content"])
    return response

def get_embedding(model_name, text_input):
    """
    Get embeddings using LangChain's HuggingFace embeddings.
    Automatically uses the best available device (GPU, MPS, or CPU).
    
    Args:
        model_name (str): Name of the HuggingFace model to use
        text_input (str or list): Text or list of texts to embed
    
    Returns:
        list: List of embeddings for the input text(s)
    """
    # Get the best available device
    device = get_available_device()
    print(f"Using device: {device}")
    
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Handle both single string and list of strings
    if isinstance(text_input, str):
        text_input = [text_input]
    
    # Get embeddings
    embeddings_result = embeddings.embed_documents(text_input)
    return embeddings_result

def double_check_json_output(client, model_name, json_string):
    # Initialize LangChain chat model with Groq
    chat_model = ChatGroq(
        model_name=os.getenv("GROQ_MODEL_NAME"),
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a JSON validation expert. Your task is to:
        1. Check the provided JSON string for validity
        2. Correct any mistakes that would make it invalid
        3. Return ONLY the corrected JSON string
        4. Ensure all keys are enclosed in double quotes
        5. Start with an opening curly brace and end with a closing curly brace"""),
        ("human", "Please validate and correct this JSON string:\n{json_string}")
    ])
    
    chain = LLMChain(
        llm=chat_model,
        prompt=prompt,
        verbose=True
    )
    
    response = chain.predict(json_string=json_string)
    return response.replace("'", "")