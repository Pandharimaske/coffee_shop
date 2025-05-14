from dotenv import load_dotenv
import os
import json
import logging
from copy import deepcopy
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from .utils import double_check_json_output
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GuardDecision(BaseModel):
    """Schema for guard agent decision"""
    chain_of_thought: str = Field(description="Reasoning about whether the query is allowed")
    decision: str = Field(description="Decision: 'allowed' or 'not allowed'")
    message: str = Field(description="Response message if not allowed, empty if allowed")

class GuardAgent:
    def __init__(self):
        # Initialize Groq client
        self.client = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME"),
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Create the system prompt template
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
            Your task is to determine whether the user is asking something relevant to the coffee shop or not.

            The user is allowed to:
            1. Ask questions about the coffee shop, like location, working hours, menu items and coffee shop related questions.
            2. Ask questions about menu items, they can ask for ingredients in an item and more details about the item.
            3. Make an order.
            4. Ask about recommendations of what to buy.

            The user is NOT allowed to:
            1. Ask questions about anything else other than our coffee shop.
            2. Ask questions about the staff or how to make a certain menu item.

            Your response should be in JSON format with these fields:
            {
                "chain_of_thought": "your reasoning about the query",
                "decision": "allowed" or "not allowed",
                "message": "" if allowed, or "Sorry, I can't help with that. Can I help you with your order?" if not allowed
            }"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the chain
        self.chain = LLMChain(
            llm=self.client,
            prompt=self.system_prompt,
            memory=self.memory,
            verbose=True
        )
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=GuardDecision)
        
        logger.info("GuardAgent initialized successfully")
    
    def get_response(self, messages):
        """Get guard agent response"""
        messages = deepcopy(messages)
        chain_response = None
        
        try:
            logger.info(f"Processing message: {messages[-1]['content']}")
            
            # Get response from the chain
            chain_response = self.chain.predict(input=messages[-1]['content'])
            logger.info(f"Chain response: {chain_response}")
            
            # Parse the response
            parsed_response = self.output_parser.parse(chain_response)
            logger.info(f"Parsed response: {parsed_response}")
            
            output = self.postprocess(parsed_response)
            logger.info(f"Final output: {output}")
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            # Fallback to JSON parsing if Pydantic parsing fails
            if chain_response is not None:
                try:
                    json_response = double_check_json_output(
                        self.client,
                        os.getenv("GROQ_MODEL_NAME"),
                        chain_response
                    )
                    output = self.postprocess(json.loads(json_response))
                except Exception as json_error:
                    logger.error(f"Error in JSON parsing: {str(json_error)}")
                    output = {
                        "role": "assistant",
                        "content": "Sorry, I couldn't process that request. Can I help you with something else?",
                        "memory": {
                            "agent": "guard_agent",
                            "guard_decision": "not allowed"
                        }
                    }
            else:
                output = {
                    "role": "assistant",
                    "content": "Sorry, I couldn't process that request. Can I help you with something else?",
                    "memory": {
                        "agent": "guard_agent",
                        "guard_decision": "not allowed"
                    }
                }
        
        return output

    def postprocess(self, output):
        """Process the output and format the response"""
        try:
            if isinstance(output, GuardDecision):
                decision = output.decision
                message = output.message if output.message else "I understand. How can I help you with your order?"
            else:
                decision = output['decision']
                message = output['message'] if output['message'] else "I understand. How can I help you with your order?"
            
            return {
                "role": "assistant",
                "content": message,
                "memory": {
                    "agent": "guard_agent",
                    "guard_decision": decision
                }
            }
        except Exception as e:
            logger.error(f"Error in postprocess: {str(e)}")
            return {
                "role": "assistant",
                "content": "Sorry, I couldn't process that request. Can I help you with something else?",
                "memory": {
                    "agent": "guard_agent",
                    "guard_decision": "not allowed"
                }
            }



   