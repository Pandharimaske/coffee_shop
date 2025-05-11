from agents.gaurd_agent import GuardAgent
from agents.classification_agent import ClassificationAgent
import os

def main():
    pass


if __name__ == "__main__":
    gaurd_agent = GuardAgent()
    classification_agent = ClassificationAgent()


    messages = []
    while True:
        #os.system('cls' if os.name == 'nt' else 'clear')

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
