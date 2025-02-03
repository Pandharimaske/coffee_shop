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


def main():
    
    gaurd_agent = GaurdAgent()
    classification_agent = ClassificationAgent()

    recommendation_agent = RecommendationAgent(
            "/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/apriori_recommendations.json" ,
            "/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/popularity_recommendation.csv"
        )

    agent_dict: Dict[str , AgentProtocol] = {
        "details_agent" : DetailsAgent() , 
        "recommendation_agent": recommendation_agent,
        "order_taking_agent": OrderTakingAgent(recommendation_agent=recommendation_agent)
    }

    
    messages = []
    while True:
        # Flush Output
        os.system("cls" if os.name == 'nt' else 'clear')

        print("\n\n Print Messages ..............")
        for message in messages:
            print(f"{message['role']} : {message['content']}")
                  
        # Get user input
        prompt = input("User: ")
        messages.append({"role":"user" , "content":prompt})


        # Get gaurd agent response
        gaurd_agent_response = gaurd_agent.get_response(messages)
        # print("GAURD AGENT OUTPUT:" , gaurd_agent_response)
        if gaurd_agent_response["memory"]["gaurd_decision"] == "not allowed":
            messages.append(gaurd_agent_response)
            continue

        # Get Classification Agent's response
        classification_agent_response = classification_agent.get_response(messages)
        chossen_agent = classification_agent_response["memory"]["classification_decision"]
        print("Chosen Agent: " , chossen_agent)


        # Get Chosen Agent's response
        agent = agent_dict[chossen_agent]
        response = agent.get_response(messages)
        print("Agent output: " , response)
        
        messages.append(response)

if __name__ == "__main__":
    main()
    
