from agents.gaurd_agent import GuardAgent
from agents.classification_agent import ClassificationAgent
from agents.details_agent import DetailsAgent
from agents.recommendation_agent import RecommendationAgent
from agents.order_taking_agent import OrderTakingAgent
from agents.agent_protocol import AgentProtocol
import os
from typing import Dict

def main():
    gaurd_agent = GuardAgent()
    classification_agent = ClassificationAgent()
    recommendation_agent = RecommendationAgent('/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/apriori_recommendations.json',
                                                '/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/popularity_recommendation.csv'
                                                )
    agent_dict : Dict[str ,AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "recommendation_agent": recommendation_agent ,  
        "order_taking_agent": OrderTakingAgent(recommendation_agent)
    }


    messages = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Get user input
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response
        guard_agent_response = gaurd_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            messages.append(guard_agent_response)
            continue
        
        # Get ClassificationAgent's response
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent=classification_agent_response["memory"]["classification_decision"]

        # Get the chosen agent's response
        agent = agent_dict[chosen_agent]
        response = agent.get_response(messages)
        
        messages.append(response)

if __name__ == "__main__":
    main()
