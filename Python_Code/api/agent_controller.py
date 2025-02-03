from agents.gaurd_agent import GaurdAgent
from agents.classification_agent import ClassificationAgent
from agents.details_agent import DetailsAgent
from agents.agent_protocol import AgentProtocol
from agents.recommendation_agent import RecommendationAgent
from agents.order_taking_agent import OrderTakingAgent
import os
from typing import Dict
import sys
import pathlib

folder_path = pathlib.Path(__file__).parent.resolve()

class AgentController():

    def __init__(self):
        self.gaurd_agent = GaurdAgent()
        self.classification_agent = ClassificationAgent()

        self.recommendation_agent = RecommendationAgent(
                "/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/apriori_recommendations.json" ,
                "/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/popularity_recommendation.csv"
            )

        self.agent_dict: Dict[str , AgentProtocol] = {
            "details_agent" : DetailsAgent() , 
            "recommendation_agent": self.recommendation_agent,
            "order_taking_agent": OrderTakingAgent(recommendation_agent=self.recommendation_agent)
        }

    def get_response(self,input):
        # Extract User Input
        job_input = input["input"]
        messages = job_input["messages"]

        # Get GuardAgent's response
        gaurd_agent_response = self.gaurd_agent.get_response(messages)
        if gaurd_agent_response["memory"]["gaurd_decision"] == "not allowed":
            return gaurd_agent_response
        
        # Get ClassificationAgent's response
        classification_agent_response = self.classification_agent.get_response(messages)
        chosen_agent=classification_agent_response["memory"]["classification_decision"]

        # Get the chosen agent's response
        agent = self.agent_dict[chosen_agent]
        response = agent.get_response(messages)

        return response
    


    