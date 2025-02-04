import ollama
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import dotenv
dotenv.load_dotenv()
import os

embd_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME"))



def get_chatbot_response(model_name , messages):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"] , "content": message["content"]})
    
    response = ollama.chat(
        model = model_name , 
        messages=input_messages , 
    )

    return response["message"]["content"]




def get_embedding(model_name, text_input):
    """
    Generate an embedding similar to OpenAI's structure.
    Ensures output is a 2D array.
    """
    embeddings = embd_model.encode(text_input)

    # Ensure the output is a 2D array (1 sample, N features)
    return np.array(embeddings).reshape(1, -1)

def double_check_json_output(model_name,json_string):
    prompt = f""" You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.
    
    if there is any text before order after the json string , remove it.
    Do NOT return a single letter outside of the json string.
    Make sure that each key is enclosed in double quotes.
    The first thing you write should be open curly brace of the json and the last letter you write should be the closing curly brace. 

    You should check the json string for the following text between triple backtics.
    '''
    {json_string}
    '''
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_chatbot_response(model_name,messages)
    response = response.replace("'" , "")
    try:
        json.loads(response)
    except json.JSONDecodeError:
        # Attempt to correct the JSON format
        corrected_output = double_check_json_output(model_name , response)
        return corrected_output
    return response

    

