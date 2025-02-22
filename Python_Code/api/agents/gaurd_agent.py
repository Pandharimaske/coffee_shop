import os
import json
from copy import deepcopy
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import dotenv

dotenv.load_dotenv()

class GuardAgent():
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "llama3.2")  # Default to llama3.2
        self.chatbot = ChatOllama(model=self.model_name)

    def get_response(self, messages):
        messages = deepcopy(messages)

        system_prompt = """
        You are an AI assistant for a coffee shop application that serves drinks and pastries. 
        Your task is to analyze user inputs, classify them based on relevance to the coffee shop, and respond in a strictly structured JSON format.

        ### Allowed User Requests:
        1. Inquiries about the coffee shop (location, working hours, menu items, and related questions).
        2. Questions about menu items (ingredients, nutritional details, and descriptions).
        3. Placing an order for drinks, pastries, or other items on the menu.
        4. Requests for recommendations on what to buy.

        ### Not Allowed User Requests:
        1. Questions unrelated to the coffee shop (e.g., personal advice, unrelated topics).
        2. Questions about the staff or how to prepare specific menu items.

        ### Instructions:
        - Strictly output valid JSON without any extra text or explanations.
        - Ensure all keys and values are strings, with consistent formatting.
        - Carefully analyze the user input to ensure accurate classification.

        ### JSON Response Format:
        {
            "chain_of_thought": "Step-by-step reasoning based on the allowed/not allowed categories to justify the decision.",
            "decision": "allowed" or "not allowed",
            "message": "" if "decision" is "allowed", otherwise "Sorry, I can't help with that. Can I help you with your order?"
        }
        """
        
        input_messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=messages[-1]["content"])]
        chatbot_output = self.chatbot.invoke(input_messages).content
        chatbot_output = self.postprocess(chatbot_output)

        return chatbot_output

    def postprocess(self, output):
        # Ensure output is in proper JSON format
        output = json.loads(output)

        dict_output = {
            "role": "assistant",
            "content": output["message"],
            "memory": {
                "agent": "guard_agent",
                "guard_decision": output["decision"]
            }
        }
        return dict_output












# import os
# import json
# from copy import deepcopy
# from .utils import get_chatbot_response, double_check_json_output
# import dotenv
# dotenv.load_dotenv()

# class GaurdAgent():
#     def __init__(self):
#         self.model_name = os.getenv("MODEL_NAME")

#     def get_response(self, messages):
#         messages = deepcopy(messages)

#         system_prompt = """
#         You are an AI assistant for a coffee shop application that serves drinks and pastries. 
#         Your task is to analyze user inputs, classify them based on relevance to the coffee shop, and respond in a strictly structured JSON format.

#         ### Allowed User Requests:
#         1. Inquiries about the coffee shop (location, working hours, menu items, and related questions).
#         2. Questions about menu items (ingredients, nutritional details, and descriptions).
#         3. Placing an order for drinks, pastries, or other items on the menu.
#         4. Requests for recommendations on what to buy.

#         ### Not Allowed User Requests:
#         1. Questions unrelated to the coffee shop (e.g., personal advice, unrelated topics).
#         2. Questions about the staff or how to prepare specific menu items.

#         ### Instructions:
#         - Strictly output valid JSON without any extra text or explanations.
#         - Ensure all keys and values are strings, with consistent formatting.
#         - Carefully analyze the user input to ensure accurate classification.

#         ### JSON Response Format:
#         {
#             "chain_of_thought": "Step-by-step reasoning based on the allowed/not allowed categories to justify the decision.",
#             "decision": "allowed" or "not allowed",
#             "message": "" if "decision" is "allowed", otherwise "Sorry, I can't help with that. Can I help you with your order?"
#         }

#         ### Example 1:
#         User: "What time do you close today?"
#         Response:
#         {
#             "chain_of_thought": "The question is about the coffee shop's working hours, which is allowed.",
#             "decision": "allowed",
#             "message": ""
#         }

#         ### Example 2:
#         User: "How do I make a cappuccino at home?"
#         Response:
#         {
#             "chain_of_thought": "The user is asking how to prepare a menu item, which is not allowed.",
#             "decision": "not allowed",
#             "message": "Sorry, I can't help with that. Can I help you with your order?"
#         }

#         ### Rules:
#         - If unsure, default to "not allowed."
#         - No extra spaces, line breaks, or explanations outside the JSON structure.
#         """
        
#         input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

#         chatbot_output = get_chatbot_response(self.model_name, input_messages)
#         chatbot_output = double_check_json_output(self.model_name, chatbot_output)

#         output = self.postprocess(chatbot_output)

#         return output

#     def postprocess(self, output):
#         # Ensure output is in proper JSON format
#         output = json.loads(output)

#         # Structured response with decision and message
#         dict_output = {
#             "role": "assistant",
#             "content": output["message"],
#             "memory": {
#                 "agent": "gaurd_agent",
#                 "gaurd_decision": output["decision"]
#             }
#         }

#         return dict_output












# # import os
# # import ollama
# # import json
# # from copy import deepcopy
# # from .utils import get_chatbot_response , double_check_json_output
# # import dotenv
# # dotenv.load_dotenv()


# # class GaurdAgent():
# #     def __init__(self):
# #         self.model_name = os.getenv("MODEL_NAME")

# #     def get_response(self , messages):
# #         messages = deepcopy(messages)

# #         # system_prompt = """
# #         #     You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
# #         #     Your task is to determine whether the user is asking something relevant to the coffee shop or not.
# #         #     The user is allowed to:
# #         #     1. Ask questions about the coffee shop, like location, working hours, menue items and coffee shop related questions.
# #         #     2. Ask questions about menue items, they can ask for ingredients in an item and more details about the item.
# #         #     3. Make an order.
# #         #     4. ASk about recommendations of what to buy.

# #         #     The user is NOT allowed to:
# #         #     1. Ask questions about anything else other than our coffee shop.
# #         #     2. Ask questions about the staff or how to make a certain menue item.

# #         #     Your output should be in a structured json format like so. each key is a string and each value is a string. Make sure to follow the format exactly:
# #         #     {
# #         #     "chain of thought": "go over each of the points above and make see if the message lies under this point or not. Then you write some your thoughts about what point is this input relevant to.",
# #         #     "decision": "allowed" or "not allowed". Pick one of those. and only write the word. , 
# #         #     "message": leave the message empty "" if it's allowed, otherwise write "Sorry, I can't help with that. Can I help you with your order?"
# #         #     }
# #         # """
# #         system_prompt = """
# #         You are an AI assistant for a coffee shop application that serves drinks and pastries. 
# #         Your task is to analyze user inputs, classify them based on relevance to the coffee shop, and respond in a strictly structured JSON format.

# #         ### Allowed User Requests:
# #         1. Inquiries about the coffee shop (location, working hours, menu items, and related questions).
# #         2. Questions about menu items (ingredients, nutritional details, and descriptions).
# #         3. Placing an order for drinks, pastries, or other items on the menu.
# #         4. Requests for recommendations on what to buy.

# #         ### Not Allowed User Requests:
# #         1. Questions unrelated to the coffee shop (e.g., personal advice, unrelated topics).
# #         2. Questions about the staff or how to prepare specific menu items.

# #         ### Instructions:
# #         - Strictly output valid JSON without any extra text or explanations.
# #         - Ensure all keys and values are strings, with consistent formatting.
# #         - Carefully analyze the user input to ensure accurate classification.

# #         ### JSON Response Format:
# #         {
# #             "chain_of_thought": "Step-by-step reasoning based on the allowed/not allowed categories to justify the decision.",
# #             "decision": "allowed" or "not allowed",
# #             "message": "" if "decision" is "allowed", otherwise "Sorry, I can't help with that. Can I help you with your order?"
# #         }

# #         ### Example 1:
# #         User: "What time do you close today?"
# #         Response:
# #         {
# #             "chain_of_thought": "The question is about the coffee shop's working hours, which is allowed.",
# #             "decision": "allowed",
# #             "message": ""
# #         }

# #         ### Example 2:
# #         User: "How do I make a cappuccino at home?"
# #         Response:
# #         {
# #             "chain_of_thought": "The user is asking how to prepare a menu item, which is not allowed.",
# #             "decision": "not allowed",
# #             "message": "Sorry, I can't help with that. Can I help you with your order?"
# #         }

# #         ### Rules:
# #         - If unsure, default to "not allowed."
# #         - No extra spaces, line breaks, or explanations outside the JSON structure.
# #         """
        
# #         input_messages = [{"role":"system" , "content":system_prompt}] + messages[-3:]

# #         chatbot_output = get_chatbot_response(self.model_name , input_messages)
# #         chatbot_output = double_check_json_output(self.model_name , chatbot_output)


# #         output = self.postprocess(chatbot_output)

# #         return output
    
# #     def postprocess(self , output):
        
# #         output = json.loads(output)

# #         dict_output = {
# #             "role": "assistant" , 
# #             "content":output["message"] , 
# #             "memory":{
# #                 "agent":"gaurd_agent" , 
# #                 "gaurd_decision":output["decision"]
# #             }
# #         }

# #         return dict_output
    


