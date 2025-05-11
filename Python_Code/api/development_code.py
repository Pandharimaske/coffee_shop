from agents.gaurd_agent import GuardAgent
from agents.classification_agent import ClassificationAgent
from agents.details_agent import DetailsAgent
from agents.recommendation_agent import RecommendationAgent
from agents.agent_protocol import AgentProtocol
import os
from typing import Dict

def main():
    pass


if __name__ == "__main__":
    gaurd_agent = GuardAgent()
    classification_agent = ClassificationAgent()
    recommendation_agent = RecommendationAgent('/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/apriori_recommendations.json',
                                                    '/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/popularity_recommendation.csv'
                                                    )


    agent_dict : Dict[str ,AgentProtocol] = {
        "details_agent": DetailsAgent(),
        #"classification_agent": ClassificationAgent(),
        #"guard_agent": GuardAgent()
    }


    messages = []
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("\n\n Print Messages ...........")
        for message in messages:
            print(f"{message['role']}: {message['content']}")

        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})


        # Get Gaurd Agent Response
        gaurd_agent_response = gaurd_agent.get_response(messages)

        if gaurd_agent_response['memory']['guard_decision'] == "not allowed":
            messages.append(gaurd_agent_response)
            continue
        
        # Get Classifier Agent Response
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent = classification_agent_response["memory"]["classification_decision"]
        print("Chosen Agent: ", chosen_agent)

        # Get the chosen agent's response
        agent = agent_dict[chosen_agent]
        response = agent.get_response(messages)

        messages.append(response)
        print("Agent Response: ", response["content"])



