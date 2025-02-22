import os
import json
import dotenv
import ollama
from copy import deepcopy
from pinecone import Pinecone
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .utils import get_chatbot_response, get_embedding

dotenv.load_dotenv()

class DetailsAgent:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)
        
        # Initialize LangChain LLM
        self.llm = ChatOllama(model=self.model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""
            Using the following contexts, answer the query:
            
            Context:
            {context}
            
            Query: {query}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def get_closest_results(self, input_embeddings, top_k=2):
        try:
            results = self.index.query(
                namespace='ns1',
                vector=input_embeddings.tolist(),
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )
            return results['matches']
        except Exception as e:
            print(f"[Error] Pinecone query failed: {str(e)}")
            return []
    
    def get_response(self, messages):
        messages = deepcopy(messages)
        user_message = messages[-1]["content"]
        
        # Get embeddings
        embeddings = get_embedding(self.embedding_model, user_message)[0]
        
        # Retrieve relevant context
        results = self.get_closest_results(embeddings)
        context = "\n".join([x['metadata']['text'].strip() for x in results]) if results else ""
        
        # Prepare structured prompt
        input_data = {"context": context, "query": user_message}
        chatbot_output = self.chain.run(input_data)
        
        return self.postprocess(chatbot_output)
    
    def postprocess(self, output):
        return {
            "role": "assistant",
            "content": output,
            "memory": {"agent": "details_agent"}
        }













# import os
# import ollama
# import json
# from copy import deepcopy
# from .utils import get_chatbot_response , get_embedding
# import dotenv
# dotenv.load_dotenv()
# from pinecone import Pinecone


# class DetailsAgent():
#     def __init__(self):
#         self.model_name = os.getenv("MODEL_NAME")
#         self.pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
#         self.index_name = os.getenv("PINECONE_INDEX_NAME")
#         self.embedding_client = os.getenv("EMBEDDING_MODEL_NAME")
    
#     def get_closest_results(self , index_name , input_embeddings , top_k = 2):
#         index = self.pc.Index(index_name)

#         results = index.query(
#             namespace = 'ns1' ,
#             vector = input_embeddings.tolist() , 
#             top_k = top_k , 
#             include_values = False , 
#             include_metadata = True
#         )

#         return results
    
#     def get_response(self , messages):
#         messages = deepcopy(messages)

#         user_message = messages[-1]["content"]
#         embeddings = get_embedding(self.embedding_client , user_message)[0]

#         result = self.get_closest_results(self.index_name , embeddings)

#         source_knowledge = "\n".join([x['metadata']['text'].strip()+'\n' for x in result['matches']])

#         prompt = f"""
#         Using the contexts below, answer the query.

#         Contexts:
#         {source_knowledge}

#         Query: {user_message}
#         """


#         # system_prompt = """ You are a customer support agent for a coffee shop called Merry's way.You should answer every question as if you are waiter and provide the neccessary information to the user regarding their orders """
#         system_prompt = """
#         You are a friendly and helpful customer support agent for a coffee shop called **Merry's Way**.  
#         Respond to every question as if you are a polite, cheerful waiter eager to assist customers with their orders and inquiries.  

#         ### 🎯 **Your Responsibilities:**  
#         1. **Menu Assistance:** Provide clear, concise details about menu items, ingredients, prices, and specials.  
#         2. **Order Support:** Help users place, modify, or inquire about orders efficiently.  
#         3. **Recommendations:** Suggest drinks, pastries, or combos based on customer preferences.  
#         4. **General Inquiries:** Answer questions about Merry's Way, such as working hours, location, and services.

#         ---

#         ### ☕ **Tone and Style Guidelines:**  
#         - Be **friendly**, **polite**, and **enthusiastic**, like a welcoming waiter.  
#         - Keep responses **concise** but **informative**.  
#         - Use **positive language** to create a warm, inviting experience.  

#         ---

#         ### ✅ **Few-Shot Examples:**  

#         #### **Example 1 - Menu Inquiry**
#         **User:** "What’s on your menu?"  
#         **Assistant:**  
#         "Hi there! 😊 At Merry's Way, we offer a variety of delicious drinks like cappuccinos, lattes, and iced coffees, along with pastries such as croissants and muffins. Would you like to know more about any specific item?"  

#         #### **Example 2 - Order Inquiry**
#         **User:** "Can I order a large caramel latte?"  
#         **Assistant:**  
#         "Of course! ☕ I've noted your large caramel latte. Would you like to add anything else to your order?"  

#         #### **Example 3 - Recommendation Request**
#         **User:** "What do you recommend with a cappuccino?"  
#         **Assistant:**  
#         "Great choice! A fresh almond croissant pairs wonderfully with a cappuccino. Would you like to try it with your coffee? 😊"  

#         #### **Example 4 - Working Hours**
#         **User:** "What time do you open on Sundays?"  
#         **Assistant:**  
#         "Good question! ⏰ Merry's Way is open from 8 AM to 10 PM on Sundays. Let me know if I can assist you with anything else!"  

#         ---

#         ### ⚠️ **Important Notes:**  
#         - **Stay relevant:** Only answer questions related to Merry's Way and its services.  
#         - **Politeness matters:** Even if you can’t fulfill a request, respond kindly (e.g., "I'm sorry, but we don’t offer that. Can I assist you with something else?").  
#         """ 
    
        
#         messages[-1]["content"] = prompt
#         input_messages = [{"role":"system" , "content":system_prompt}] + messages[-3:]

#         chatbot_output = get_chatbot_response(self.model_name , input_messages)

#         output = self.postprocess(chatbot_output)

#         return output
    
#     def postprocess(self, output):
#         output = {
#             "role": "assistant",
#             "content": output,
#             "memory": {"agent":"details_agent"}
#         }
#         return output





