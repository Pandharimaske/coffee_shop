from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from typing import Dict
import pathlib

# Import your agents
from agents.gaurd_agent import GaurdAgent
from agents.classification_agent import ClassificationAgent
from agents.details_agent import DetailsAgent
from agents.recommendation_agent import RecommendationAgent
from agents.order_taking_agent import OrderTakingAgent
from agents.agent_protocol import AgentProtocol

# Get absolute path for dependencies
folder_path = pathlib.Path(__file__).parent.resolve()

class AgentController:
    def __init__(self):
        self.gaurd_agent = GaurdAgent()
        self.classification_agent = ClassificationAgent()

        # LangChain-based chatbot
        self.chat_model = ChatOllama(model="your-ollama-model", temperature=0.7)

        # LangChain Memory (keeps track of conversation history)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Recommendation agent with LangChain integration
        self.recommendation_agent = RecommendationAgent(
            f"{folder_path}/recommendation_objects/apriori_recommendations.json",
            f"{folder_path}/recommendation_objects/popularity_recommendation.csv"
        )

        # Dictionary to map agent types
        self.agent_dict: Dict[str, AgentProtocol] = {
            "details_agent": DetailsAgent(),
            "recommendation_agent": self.recommendation_agent,
            "order_taking_agent": OrderTakingAgent(recommendation_agent=self.recommendation_agent),
        }

    def get_response(self, input):
        # Extract user input
        job_input = input["input"]
        messages = job_input["messages"]

        # Convert messages into LangChain format
        lc_messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in messages
        ]

        # Step 1: GuardAgent Decision
        gaurd_agent_response = self.gaurd_agent.get_response(messages)
        if gaurd_agent_response["memory"]["gaurd_decision"] == "not allowed":
            return gaurd_agent_response

        # Step 2: ClassificationAgent Decision
        classification_agent_response = self.classification_agent.get_response(messages)
        chosen_agent = classification_agent_response["memory"]["classification_decision"]

        # Step 3: Get Response from the Chosen Agent
        if chosen_agent in self.agent_dict:
            agent = self.agent_dict[chosen_agent]
            response = agent.get_response(messages)

            # Store response in LangChain memory
            self.memory.chat_memory.add_message(AIMessage(content=response["message"]))

            return response
        else:
            return {"error": "Invalid agent decision"}
        










# from agents.gaurd_agent import GaurdAgent
# from agents.classification_agent import ClassificationAgent
# from agents.details_agent import DetailsAgent
# from agents.agent_protocol import AgentProtocol
# from agents.recommendation_agent import RecommendationAgent
# from agents.order_taking_agent import OrderTakingAgent
# import os
# from typing import Dict
# import sys
# import pathlib

# folder_path = pathlib.Path(__file__).parent.resolve()

# class AgentController():

#     def __init__(self):
#         self.gaurd_agent = GaurdAgent()
#         self.classification_agent = ClassificationAgent()

#         self.recommendation_agent = RecommendationAgent(
#                 "/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/apriori_recommendations.json" ,
#                 "/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/popularity_recommendation.csv"
#             )

#         self.agent_dict: Dict[str , AgentProtocol] = {
#             "details_agent" : DetailsAgent() , 
#             "recommendation_agent": self.recommendation_agent,
#             "order_taking_agent": OrderTakingAgent(recommendation_agent=self.recommendation_agent)
#         }

#     def get_response(self,input):
#         # Extract User Input
#         job_input = input["input"]
#         messages = job_input["messages"]

#         # Get GuardAgent's response
#         gaurd_agent_response = self.gaurd_agent.get_response(messages)
#         if gaurd_agent_response["memory"]["gaurd_decision"] == "not allowed":
#             return gaurd_agent_response
        
#         # Get ClassificationAgent's response
#         classification_agent_response = self.classification_agent.get_response(messages)
#         chosen_agent=classification_agent_response["memory"]["classification_decision"]

#         # Get the chosen agent's response
#         agent = self.agent_dict[chosen_agent]
#         response = agent.get_response(messages)

#         return response
    


    