import os
import json
import dotenv
import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.chat_models import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables
dotenv.load_dotenv()

# Get embedding model name from environment variables
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")  # Default to "nomic-embed-text"
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "llama3.2")  # Change to your Ollama model name

# Initialize LangChain LLM for Ollama
llm = ChatOllama(model=LLM_MODEL, temperature=0.7)


def get_embedding(text_input):
    """
    Generate embeddings using Ollama.
    Returns a 2D NumPy array.
    """
    if isinstance(text_input, str):
        text_input = [text_input]  # Convert string to list

    embeddings = [ollama.embeddings(model=MODEL_NAME, prompt=text)["embedding"] for text in text_input]

    return np.array(embeddings)  # Ensure 2D output


def double_check_json_output(json_string, max_retries=3):
    """
    Validates and corrects a JSON string using LangChain's LLM.
    Retries correction up to `max_retries` times if invalid JSON is received.
    """
    system_prompt = """
    You are an AI assistant that validates and corrects JSON strings.
    - Ensure the JSON is correctly formatted with double-quoted keys.
    - If the JSON is valid, return it as is.
    - If it's invalid, fix errors and return only the corrected JSON.
    - Do not add any extra text or comments.
    """

    for attempt in range(max_retries):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Check and fix this JSON: ```{json_string}```"),
        ]
        response = llm(messages).content  # Get response from LangChain LLM
        
        try:
            json_data = json.loads(response)  # Validate JSON
            return json.dumps(json_data, indent=4)  # Ensure valid output
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON received (Attempt {attempt + 1}/{max_retries})")

    print("Error: Could not fix the JSON after multiple attempts.")
    return None  # Return None if JSON is still invalid


# Example Usage:
if __name__ == "__main__":
    sample_json = '{"name": "John", age: 30, "city": "New York"}'  # Incorrect JSON (missing quotes around "age")
    
    corrected_json = double_check_json_output(sample_json)
    
    if corrected_json:
        print("Corrected JSON:", corrected_json)
    else:
        print("Failed to correct JSON.")





# import ollama
# import json
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import dotenv
# import os

# # Load environment variables
# dotenv.load_dotenv()

# # Get embedding model name from environment variable
# MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")  # Default to "nomic-embed-text"


# def get_chatbot_response(model_name, messages):
#     """
#     Function to interact with the chatbot using Ollama.
#     """
#     input_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

#     response = ollama.chat(model=model_name, messages=input_messages)
    
#     return response.get("message", {}).get("content", "").strip()


# def get_embedding(text_input):
#     """
#     Generate embeddings using Ollama instead of SentenceTransformer.
#     Returns a 2D NumPy array.
#     """
#     if isinstance(text_input, str):
#         text_input = [text_input]  # Convert string to list

#     embeddings = [ollama.embeddings(model=MODEL_NAME, prompt=text)["embedding"] for text in text_input]

#     return np.array(embeddings)  # Ensure 2D output


# def double_check_json_output(model_name, json_string, max_retries=3):
#     """
#     Validates and corrects a JSON string using a chatbot.
#     Retries correction up to `max_retries` times if invalid JSON is received.
#     """
#     prompt = f"""
#     You will check this JSON string and correct any mistakes that will make it invalid.
#     Then, return only the corrected JSON string—nothing else.
    
#     If the JSON is correct, return it as is.
#     If there is extra text before or after the JSON string, remove it.
#     Do NOT return anything outside of the JSON string.
#     Ensure all keys are enclosed in double quotes.

#     Here is the JSON to check:
#     ```
#     {json_string}
#     ```
#     """

#     for attempt in range(max_retries):
#         messages = [{"role": "user", "content": prompt}]
#         response = get_chatbot_response(model_name, messages)
        
#         # Attempt to load the JSON
#         try:
#             json_data = json.loads(response)  # Validate JSON
#             return json.dumps(json_data, indent=4)  # Ensure valid output
#         except json.JSONDecodeError:
#             print(f"Warning: Invalid JSON received (Attempt {attempt + 1}/{max_retries})")

#     print("Error: Could not fix the JSON after multiple attempts.")
#     return None  # Return None if JSON is still invalid


# # Example Usage:
# if __name__ == "__main__":
#     sample_json = '{"name": "John", age: 30, "city": "New York"}'  # Incorrect JSON (missing quotes around "age")
    
#     corrected_json = double_check_json_output("your_chatbot_model", sample_json)
    
#     if corrected_json:
#         print("Corrected JSON:", corrected_json)
#     else:
#         print("Failed to correct JSON.")