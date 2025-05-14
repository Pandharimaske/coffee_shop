from agents.gaurd_agent import GuardAgent
from agents.classification_agent import ClassificationAgent
from agents.details_agent import DetailsAgent
from agents.recommendation_agent import RecommendationAgent
from agents.order_taking_agent import OrderTakingAgent
from agents.agent_protocol import AgentProtocol, AgentResponse
import os
import logging
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CoffeeShopChatbot:
    def __init__(self):
        """Initialize the coffee shop chatbot with all necessary agents"""
        try:
            # Get the base path for recommendation objects
            base_path = Path(__file__).parent
            
            # Initialize agents
            self.guard_agent = GuardAgent()
            self.classification_agent = ClassificationAgent()
            self.recommendation_agent = RecommendationAgent(
                str(base_path / 'recommendation_objects/apriori_recommendations.json'),
                str(base_path / 'recommendation_objects/popularity_recommendation.csv')
            )
            
            # Initialize agent dictionary
            self.agent_dict: Dict[str, AgentProtocol] = {
                "details_agent": DetailsAgent(),
                "recommendation_agent": self.recommendation_agent,
                "order_taking_agent": OrderTakingAgent(self.recommendation_agent)
            }
            
            logger.info("CoffeeShopChatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing CoffeeShopChatbot: {str(e)}")
            raise

    def clear_screen(self):
        """Clear the terminal screen based on OS"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_chat_history(self, messages: List[Dict[str, Any]]):
        """Display the chat history in a formatted way"""
        print("\n\nChat History:")
        print("-" * 50)
        for message in messages:
            role = message['role'].capitalize()
            content = message['content']
            print(f"{role}: {content}")
        print("-" * 50)

    def process_user_input(self, messages: List[Dict[str, Any]], user_input: str) -> List[Dict[str, Any]]:
        """
        Process user input and get appropriate response
        
        Args:
            messages: List of conversation messages
            user_input: User's input message
            
        Returns:
            Updated list of messages
        """
        try:
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            logger.info(f"Processing user input: {user_input[:50]}...")

            # Get GuardAgent's response
            guard_agent_response = self.guard_agent.get_response(messages)
            if guard_agent_response["memory"]["guard_decision"] == "not allowed":
                logger.info("Guard agent rejected the query")
                messages.append(guard_agent_response)
                return messages

            # Get ClassificationAgent's response
            classification_agent_response = self.classification_agent.get_response(messages)
            chosen_agent = classification_agent_response["memory"]["classification_decision"]
            logger.info(f"Chosen agent: {chosen_agent}")

            # Get the chosen agent's response
            if chosen_agent in self.agent_dict:
                agent = self.agent_dict[chosen_agent]
                response = agent.get_response(messages)
                messages.append(response)
            else:
                logger.error(f"Invalid agent decision: {chosen_agent}")
                error_response = {
                    "role": "assistant",
                    "content": "Sorry, I couldn't process that request. Can I help you with something else?",
                    "memory": {"agent": "error"}
                }
                messages.append(error_response)

            return messages

        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            error_response = {
                "role": "assistant",
                "content": f"Sorry, something went wrong: {str(e)}",
                "memory": {"agent": "error"}
            }
            messages.append(error_response)
            return messages

def main():
    """Main function to run the coffee shop chatbot"""
    try:
        # Initialize chatbot
        chatbot = CoffeeShopChatbot()
        messages = []
        
        print("Welcome to the Coffee Shop Chatbot!")
        print("Type 'quit' or 'exit' to end the conversation.")
        
        while True:
            # Clear screen and display chat history
            chatbot.clear_screen()
            chatbot.display_chat_history(messages)
            
            # Get user input
            user_input = input("\nUser: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for chatting with us! Goodbye!")
                break
            
            # Process user input
            messages = chatbot.process_user_input(messages, user_input)
            
    except KeyboardInterrupt:
        print("\n\nChatbot terminated by user. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print("\nAn unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
