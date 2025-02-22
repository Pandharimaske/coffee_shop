from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama  # Example, replace if needed
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

from agents.gaurd_agent import GuardAgent
from agents.classification_agent import ClassificationAgent
from agents.details_agent import DetailsAgent
from agents.agent_protocol import AgentProtocol
from agents.recommendation_agent import RecommendationAgent
from agents.order_taking_agent import OrderTakingAgent

import os
from typing import Dict
import pathlib

# Get absolute path of the current directory
folder_path = pathlib.Path(__file__).parent.resolve()

def main():
    """Main function for running the chatbot system with LangChain integration."""
    
    # Initialize Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize AI agents
    gaurd_agent = GaurdAgent()
    classification_agent = ClassificationAgent()

    # Initialize recommendation agent with dataset paths
    recommendation_agent = RecommendationAgent(
        f"{folder_path}/recommendation_objects/apriori_recommendations.json",
        f"{folder_path}/recommendation_objects/popularity_recommendation.csv"
    )

    # Map agent names to their corresponding classes
    agent_dict: Dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "recommendation_agent": recommendation_agent,
        "order_taking_agent": OrderTakingAgent(recommendation_agent=recommendation_agent)
    }

    # LangChain Tools (agents wrapped as LangChain tools)
    tools = [
        Tool(
            name="DetailsAgent",
            func=lambda x: agent_dict["details_agent"].get_response(x),
            description="Handles inquiries about coffee shop details."
        ),
        Tool(
            name="RecommendationAgent",
            func=lambda x: agent_dict["recommendation_agent"].get_response(x),
            description="Provides coffee recommendations."
        ),
        Tool(
            name="OrderTakingAgent",
            func=lambda x: agent_dict["order_taking_agent"].get_response(x),
            description="Handles coffee orders."
        ),
    ]

    # LangChain Agent (LLM as a meta-controller)
    llm = ChatOllama(model="llama3.2" , temperature = 0)  # Change model if needed
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    # Conversation history
    messages = []
    
    while True:
        print("\n--- Conversation History ---")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")

        # Get user input
        prompt = input("\nUser: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break  # Exit loop
        
        messages.append({"role": "user", "content": prompt})

        # Step 1: GuardAgent Decision
        gaurd_agent_response = gaurd_agent.get_response(messages)
        if gaurd_agent_response["memory"].get("gaurd_decision") == "not allowed":
            messages.append({"role": "assistant", "content": "Access Denied."})
            continue  # Skip further processing

        # Step 2: ClassificationAgent Decision
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent = classification_agent_response["memory"].get("classification_decision", None)

        if chosen_agent not in agent_dict:
            print("⚠️ Error: Invalid agent classification. Defaulting to details_agent.")
            chosen_agent = "details_agent"

        print(f"🤖 Chosen Agent: {chosen_agent}")

        # Step 3: Use LangChain's Agent Executor for response
        try:
            response = agent_executor.run(prompt)
        except Exception as e:
            print(f"⚠️ LangChain Error: {str(e)}")
            response = "Sorry, I couldn't process that."

        print(f"🔹 Agent Response: {response}\n")
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()








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


# def main():
    
#     gaurd_agent = GaurdAgent()
#     classification_agent = ClassificationAgent()

#     recommendation_agent = RecommendationAgent(
#             "/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/apriori_recommendations.json" ,
#             "/Users/pandhari/coffee_shop/Python_Code/api/recommendation_objects/popularity_recommendation.csv"
#         )

#     agent_dict: Dict[str , AgentProtocol] = {
#         "details_agent" : DetailsAgent() , 
#         "recommendation_agent": recommendation_agent,
#         "order_taking_agent": OrderTakingAgent(recommendation_agent=recommendation_agent)
#     }

    
#     messages = []
#     while True:
#         # Flush Output
#         os.system("cls" if os.name == 'nt' else 'clear')

#         print("\n\n Print Messages ..............")
#         for message in messages:
#             print(f"{message['role']} : {message['content']}")
                  
#         # Get user input
#         prompt = input("User: ")
#         messages.append({"role":"user" , "content":prompt})


#         # Get gaurd agent response
#         gaurd_agent_response = gaurd_agent.get_response(messages)
#         # print("GAURD AGENT OUTPUT:" , gaurd_agent_response)
#         if gaurd_agent_response["memory"]["gaurd_decision"] == "not allowed":
#             messages.append(gaurd_agent_response)
#             continue

#         # Get Classification Agent's response
#         classification_agent_response = classification_agent.get_response(messages)
#         chossen_agent = classification_agent_response["memory"]["classification_decision"]
#         print("Chosen Agent: " , chossen_agent)


#         # Get Chosen Agent's response
#         agent = agent_dict[chossen_agent]
#         response = agent.get_response(messages)
#         print("Agent output: " , response)
        
#         messages.append(response)

# if __name__ == "__main__":
#     main()
    
