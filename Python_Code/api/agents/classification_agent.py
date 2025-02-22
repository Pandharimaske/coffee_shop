import os
import json
import dotenv
from copy import deepcopy
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

dotenv.load_dotenv()

class ClassificationAgent:
    def __init__(self):
        # Ensure the model name is set
        self.model_name = os.getenv("MODEL_NAME", "llama3.2")

        # Initialize the LangChain model
        self.llm = ChatOllama(model=self.model_name)

        # Define response schema for structured output parsing
        self.response_schemas = [
            ResponseSchema(name="chain_of_thought", description="Step-by-step reasoning for decision"),
            ResponseSchema(name="decision", description="Classification result: 'details_agent', 'order_taking_agent', or 'recommendation_agent'"),
            ResponseSchema(name="message", description="Message content (empty by default)")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

    def get_response(self, messages):
        messages = deepcopy(messages)

        system_prompt = """
        You are an AI assistant for a coffee shop application. Your task is to analyze user inputs and assign them to the correct agent.

        ### Agents:
        1. **details_agent**: Handles coffee shop details (location, hours, menu).
        2. **order_taking_agent**: Manages orders.
        3. **recommendation_agent**: Provides recommendations.

        ### Examples:
        - **User:** "What time do you open?" → **Decision:** "details_agent"
        - **User:** "Can you suggest a good pastry?" → **Decision:** "recommendation_agent"
        - **User:** "I want to order a cappuccino." → **Decision:** "order_taking_agent"

        Always return JSON in this format:
        {{"chain_of_thought": "...", "decision": "...", "message": ""}}
        """

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            *[(msg["role"], msg["content"]) for msg in messages[-3:]]
        ])

        try:
            chatbot_output = self.llm.invoke(prompt_template)
            output = self.output_parser.parse(chatbot_output.content)

        except json.JSONDecodeError:
            # Fallback mechanism
            output = {
                "chain_of_thought": "Unable to classify input due to parsing error.",
                "decision": "details_agent",
                "message": ""
            }

        return self.postprocess(output)

    def postprocess(self, output):
        return {
            "role": "assistant",
            "content": output.get("message", ""),
            "memory": {
                "agent": "classification_agent",
                "classification_decision": output.get("decision", "details_agent")
            }
        }












# import os
# import ollama
# import json
# from copy import deepcopy
# from .utils import get_chatbot_response, double_check_json_output
# import dotenv
# dotenv.load_dotenv()


# class ClassificationAgent():
#     def __init__(self):
#         self.model_name = os.getenv("MODEL_NAME")

#     def get_response(self, messages):
#         messages = deepcopy(messages)

#         system_prompt = """
#         You are an AI assistant for a coffee shop application. Your task is to analyze user inputs and assign them to the appropriate agent for handling.

#         ### Agents and Responsibilities:
#         1. **details_agent**: Handles:
#            - Coffee shop info: location, delivery areas, working hours.
#            - Menu details: ingredients, descriptions, listing items, or queries like "What do you have?"

#         2. **order_taking_agent**: Manages:
#            - Taking and finalizing orders for drinks, pastries, or other menu items.

#         3. **recommendation_agent**: Provides:
#            - Personalized suggestions based on user preferences.

#         ### 📝 Few-Shot Examples:
        
#         **Example 1:**
#         **User Input:** "What time do you open tomorrow?"
#         **Response:**
#         {{
#             "chain_of_thought": "The user is asking about shop hours, which falls under details_agent.",
#             "decision": "details_agent",
#             "message": ""
#         }}

#         **Example 2:**
#         **User Input:** "Can you recommend a good pastry with coffee?"
#         **Response:**
#         {{
#             "chain_of_thought": "The user is asking for a recommendation, so the recommendation_agent should handle it.",
#             "decision": "recommendation_agent",
#             "message": ""
#         }}

#         **Example 3:**
#         **User Input:** "I’d like to order a cappuccino and a croissant."
#         **Response:**
#         {{
#             "chain_of_thought": "The user is placing an order, which belongs to the order_taking_agent.",
#             "decision": "order_taking_agent",
#             "message": ""
#         }}

#         ### ⬇️ Now analyze the user's input and respond in **valid JSON** format:
#         """

#         # Pass the last 3 messages for context
#         input_messages = [{"role": "system", "content": system_prompt}]
#         input_messages += messages[-3:]

#         chatbot_output = get_chatbot_response(self.model_name, input_messages)
#         chatbot_output = double_check_json_output(self.model_name, chatbot_output)

#         output = self.postprocess(chatbot_output)
#         return output

#     def postprocess(self, output):
#         output = json.loads(output)

#         dict_output = {
#             "role": "assistant",
#             "content": output["message"],
#             "memory": {
#                 "agent": "classification_agent",
#                 "classification_decision": output["decision"]
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


# # class ClassificationAgent():
# #     def __init__(self):
# #         self.model_name = os.getenv("MODEL_NAME")

# #     def get_response(self , messages):
# #         messages = deepcopy(messages)

# #         # system_prompt = """
# #         #     You are a helpful AI assistant for a coffee shop application.
# #         #     Your task is to determine what agent should handle the user input. You have 3 agents to choose from:
# #         #     1. details_agent: This agent is responsible for answering questions about the coffee shop, like location, delivery places, working hours, details about menue items. Or listing items in the menu items. Or by asking what we have.
# #         #     2. order_taking_agent: This agent is responsible for taking orders from the user. It's responsible to have a conversation with the user about the order untill it's complete.
# #         #     3. recommendation_agent: This agent is responsible for giving recommendations to the user about what to buy. If the user asks for a recommendation, this agent should be used.

# #         #     Your output should be in a structured json format like so. each key is a string and each value is a string. Make sure to follow the format exactly:
# #         #     {
# #         #     "chain of thought": "go over each of the agents above and write some your thoughts about what agent is this input relevant to.",
# #         #     "decision": "details_agent" or "order_taking_agent" or "recommendation_agent". Pick one of those. and only write the word.,
# #         #     "message": leave the message empty.
# #         #     }
# #         # """

# #         system_prompt = """
# #         You are an AI assistant for a coffee shop application. Your task is to analyze user inputs and assign them to the appropriate agent for handling. 

# #         ### Agents and Responsibilities:
# #         1. **details_agent**: Handles questions related to:
# #         - Coffee shop information: location, delivery areas, working hours.
# #         - Menu details: ingredients, descriptions, listing items, or queries like "What do you have?"

# #         2. **order_taking_agent**: Manages the process of:
# #         - Taking orders for drinks, pastries, or other menu items.
# #         - Engaging in conversations about orders until they are fully completed.

# #         3. **recommendation_agent**: Provides:
# #         - Personalized suggestions for what to buy based on user preferences or requests for recommendations.

# #         ---

# #         ### 🗒️ **Decision-Making Process:**
# #         - Carefully analyze the user input.
# #         - Compare it with each agent's responsibilities.
# #         - Select the agent that best matches the user's intent.

# #         ---

# #         ### ✅ **Strict JSON Output Format:**

# #         Ensure your output follows **this exact JSON structure** with no additional text or formatting changes:

# #         {
# #             "chain_of_thought": "Step-by-step reasoning explaining why the chosen agent is the most suitable based on the user's input.",
# #             "decision": "details_agent" or "order_taking_agent" or "recommendation_agent",
# #             "message": ""
# #         }

# #         ---

# #         ### 📊 **Examples for Clarity:**

# #         1. **User Input:** "What time do you open tomorrow?"
# #         **Response:**
# #         {
# #             "chain_of_thought": "The user is asking about the coffee shop's working hours, which falls under the responsibilities of the details_agent.",
# #             "decision": "details_agent",
# #             "message": ""
# #         }

# #         2. **User Input:** "Can you recommend a good pastry with coffee?"
# #         **Response:**
# #         {
# #             "chain_of_thought": "The user is asking for a recommendation about what to buy, which should be handled by the recommendation_agent.",
# #             "decision": "recommendation_agent",
# #             "message": ""
# #         }

# #         3. **User Input:** "I’d like to order a cappuccino and a croissant."
# #         **Response:**
# #         {
# #             "chain_of_thought": "The user is clearly placing an order, which should be managed by the order_taking_agent.",
# #             "decision": "order_taking_agent",
# #             "message": ""
# #         }

# #         ---

# #         ### ⚠️ **Important Rules:**
# #         - Always output valid JSON—no additional text or explanations outside the JSON structure.
# #         - All keys and values must be strings.
# #         - If the intent is unclear, choose the agent that best matches based on reasoning.

# #         """

# #         input_messages = [
# #             {"role":"system" , "content":system_prompt}
# #         ]

# #         input_messages += messages[-3:]

# #         chatbot_output = get_chatbot_response(self.model_name , input_messages)
# #         chatbot_output = double_check_json_output(self.model_name , chatbot_output)
        
# #         output = self.postprocess(chatbot_output)

# #         return output
    
# #     def postprocess(self , output):
# #         output = json.loads(output)

# #         dict_output = {
# #             "role" : "assistant" , 
# #             "content": output["message"] , 
# #             "memory": {
# #                 "agent": "classification_agent" , 
# #                 "classification_decision": output["decision"]
# #             }
# #         }

# #         return dict_output


        
        