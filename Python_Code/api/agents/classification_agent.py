from dotenv import load_dotenv
import os
import json
from copy import deepcopy
from .utils import get_chatbot_response, double_check_json_output
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

class AgentDecision(BaseModel):
    """Schema for agent classification decision"""
    chain_of_thought: str = Field(description="Reasoning about which agent should handle the input")
    decision: Literal["details_agent", "order_taking_agent", "recommendation_agent"] = Field(
        description="The chosen agent to handle the input"
    )
    message: str = Field(description="Response message to the user", default="")

class ClassificationAgent:
    def __init__(self):
        self.client = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME"),
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Create the system prompt template
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant for a coffee shop application.
            Your task is to determine what agent should handle the user input. You have 3 agents to choose from:
            
            1. details_agent: Handles questions about:
               - Coffee shop location and working hours
               - Delivery information
               - Menu items and their details
               - General inquiries about the shop
            
            2. order_taking_agent: Handles:
               - Taking and processing orders
               - Order modifications
               - Order completion and confirmation
            
            3. recommendation_agent: Handles:
               - Product recommendations
               - Personalized suggestions
               - Menu exploration
            
            Analyze the user's input and determine the most appropriate agent to handle their request."""),
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
        self.output_parser = PydanticOutputParser(pydantic_object=AgentDecision)
    
    def get_response(self, messages):
        messages = deepcopy(messages)
        
        # Get the last 3 messages for context
        recent_messages = messages[-3:]
        
        # Convert messages to string format for the chain
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
        
        # Get response from the chain
        chain_response = self.chain.predict(input=input_text)
        
        # Parse the response into structured format
        try:
            parsed_response = self.output_parser.parse(chain_response)
            output = self.postprocess(parsed_response)
        except Exception as e:
            # Fallback to JSON parsing if Pydantic parsing fails
            json_response = double_check_json_output(self.client, os.getenv("GROQ_MODEL_NAME"), chain_response)
            output = self.postprocess(json.loads(json_response))
        
        return output

    def postprocess(self, output):
        if isinstance(output, AgentDecision):
            decision = output.decision
            message = output.message
        else:
            decision = output['decision']
            message = output.get('message', '')
        
        return {
            "role": "assistant",
            "content": message,
            "memory": {
                "agent": "classification_agent",
                "classification_decision": decision
            }
        }

    