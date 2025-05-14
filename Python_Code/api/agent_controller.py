from agents.gaurd_agent import GaurdAgent
from agents.classification_agent import ClassificationAgent
from agents.details_agent import DetailsAgent
from agents.agent_protocol import AgentProtocol, AgentResponse, AgentMemory
from agents.recommendation_agent import RecommendationAgent
from agents.order_taking_agent import OrderTakingAgent
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import os
from typing import Dict, List, Any, Optional
import pathlib
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

folder_path = pathlib.Path(__file__).parent.resolve()

class AgentController:
    def __init__(self):
        try:
            # Initialize LangChain components with Groq
            self.chat_model = ChatGroq(
                model_name=os.getenv("GROQ_MODEL_NAME"),
                temperature=0.7,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            # Create system prompt template
            self.system_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a coffee shop assistant with multiple specialized capabilities:
                1. Guard Agent: Ensures appropriate conversation
                2. Classification Agent: Routes queries to appropriate specialists
                3. Details Agent: Provides product information
                4. Recommendation Agent: Suggests products based on preferences
                5. Order Taking Agent: Handles order processing
                
                Maintain context and provide coherent, helpful responses.
                If the response seems incomplete or unclear, enhance it while maintaining the original intent."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            
            # Create main chain
            self.chain = LLMChain(
                llm=self.chat_model,
                prompt=self.system_prompt,
                memory=self.memory,
                verbose=True
            )

            # Initialize agents
            self.gaurd_agent = GaurdAgent()
            self.classification_agent = ClassificationAgent()

            self.recommendation_agent = RecommendationAgent(
                f"{folder_path}/recommendation_objects/apriori_recommendations.json",
                f"{folder_path}/recommendation_objects/popularity_recommendation.csv"
            )

            self.agent_dict: Dict[str, AgentProtocol] = {
                "details_agent": DetailsAgent(),
                "recommendation_agent": self.recommendation_agent,
                "order_taking_agent": OrderTakingAgent(recommendation_agent=self.recommendation_agent)
            }
            
            logger.info("AgentController initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AgentController: {str(e)}")
            raise

    def get_response(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input and get response from appropriate agent.
        
        Args:
            input: Dictionary containing user input with 'input' key containing 'messages'
        
        Returns:
            Dictionary containing response message, memory state, and agent info
        """
        try:
            # Extract User Input
            job_input = input["input"]
            messages = job_input["messages"]
            
            logger.info(f"Processing message: {messages[-1]['content'][:50]}...")

            # Get GuardAgent's response
            gaurd_agent_response = self.gaurd_agent.get_response(messages)
            if gaurd_agent_response["memory"]["guard_decision"] == "not allowed":
                logger.info("Guard agent rejected the query")
                return self._format_response(gaurd_agent_response)
            
            # Get ClassificationAgent's response
            classification_agent_response = self.classification_agent.get_response(messages)
            chosen_agent = classification_agent_response["memory"]["classification_decision"]
            
            logger.info(f"Chosen agent: {chosen_agent}")

            # Get the chosen agent's response
            if chosen_agent in self.agent_dict:
                agent = self.agent_dict[chosen_agent]
                response = agent.get_response(messages)
                
                # Enhance response using LangChain
                enhanced_response = self.chain.predict(
                    input=response["content"],
                    chat_history=messages
                )
                
                # Update memory
                self.memory.chat_memory.add_message(
                    AIMessage(content=enhanced_response)
                )
                
                return self._format_response({
                    "content": enhanced_response,
                    "memory": response.get("memory", {}),
                    "agent": chosen_agent
                })
            else:
                logger.error(f"Invalid agent decision: {chosen_agent}")
                return self._format_error_response("Invalid agent decision")
                
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return self._format_error_response(str(e))

    def _format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format the response according to the AgentResponse schema"""
        return {
            "message": response["content"],
            "memory": response.get("memory", {}),
            "agent": response.get("agent", "unknown")
        }

    def _format_error_response(self, error_message: str) -> Dict[str, Any]:
        """Format error response"""
        return {
            "message": f"Sorry, something went wrong: {error_message}",
            "memory": {"agent": "error"},
            "agent": "error"
        }
    


    