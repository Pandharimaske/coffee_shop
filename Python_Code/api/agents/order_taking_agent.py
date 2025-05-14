import os
import json
from .utils import double_check_json_output
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()

class OrderItem(BaseModel):
    """Schema for order items"""
    item: str = Field(description="Name of the menu item")
    quantity: int = Field(description="Quantity of the item ordered")
    price: float = Field(description="Total price for this item")

class OrderResponse(BaseModel):
    """Schema for order taking response"""
    chain_of_thought: str = Field(description="Reasoning about the current step and user input")
    step_number: int = Field(description="Current step in the order process")
    order: List[OrderItem] = Field(description="List of ordered items")
    response: str = Field(description="Response to the user")

class OrderTakingAgent:
    def __init__(self, recommendation_agent):
        # Initialize Groq client
        self.client = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME"),
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.recommendation_agent = recommendation_agent
        
        # Create the system prompt template
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a customer support Bot for a coffee shop called "Merry's way".

            Menu:
            - Cappuccino - $4.50
            - Jumbo Savory Scone - $3.25
            - Latte - $4.75
            - Chocolate Chip Biscotti - $2.50
            - Espresso shot - $2.00
            - Hazelnut Biscotti - $2.75
            - Chocolate Croissant - $3.75
            - Dark chocolate (Drinking Chocolate) - $5.00
            - Cranberry Scone - $3.50
            - Croissant - $3.25
            - Almond Croissant - $4.00
            - Ginger Biscotti - $2.50
            - Oatmeal Scone - $3.25
            - Ginger Scone - $3.50
            - Chocolate syrup - $1.50
            - Hazelnut syrup - $1.50
            - Carmel syrup - $1.50
            - Sugar Free Vanilla syrup - $1.50
            - Dark chocolate (Packaged Chocolate) - $3.00

            Important Rules:
            * DO NOT ask about payment methods (cash/card)
            * DO NOT direct users to the counter
            * DO NOT tell users where to collect their order

            Your Process:
            1. Take the User's Order
            2. Validate all items against the menu
            3. For invalid items, inform the user and confirm remaining valid items
            4. Ask if they need anything else
            5. If yes, repeat from step 3
            6. If no, complete the order by:
                - Listing all items and prices
                - Calculating the total
                - Thanking the user and closing the conversation

            Use the memory section to track:
            - Current order
            - Step number
            - Previous interactions"""),
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
        self.output_parser = PydanticOutputParser(pydantic_object=OrderResponse)
    
    def get_response(self, messages):
        messages = deepcopy(messages)
        
        # Get previous order status
        last_order_taking_status = ""
        asked_recommendation_before = False
        
        for message_index in range(len(messages)-1, 0, -1):
            message = messages[message_index]
            agent_name = message.get("memory", {}).get("agent", "")
            
            if message["role"] == "assistant" and agent_name == "order_taking_agent":
                step_number = message["memory"]["step number"]
                order = message["memory"]["order"]
                asked_recommendation_before = message["memory"]["asked_recommendation_before"]
                last_order_taking_status = f"""
                Current Status:
                Step Number: {step_number}
                Current Order: {order}
                """
                break
        
        # Combine status with user message
        combined_input = f"{last_order_taking_status}\nUser Message: {messages[-1]['content']}"
        
        try:
            # Get response from the chain
            chain_response = self.chain.predict(input=combined_input)
            
            # Parse the response
            parsed_response = self.output_parser.parse(chain_response)
            output = self.postprocess(parsed_response, messages, asked_recommendation_before)
            
        except Exception as e:
            # Fallback to JSON parsing if Pydantic parsing fails
            json_response = double_check_json_output(self.client, os.getenv("GROQ_MODEL_NAME"), chain_response)
            output = self.postprocess(json.loads(json_response), messages, asked_recommendation_before)
        
        return output

    def postprocess(self, output, messages, asked_recommendation_before):
        """
        Process the output and add recommendations if needed.
        
        Args:
            output: Either OrderResponse object or dict
            messages: List of conversation messages
            asked_recommendation_before: Boolean indicating if recommendations were given
            
        Returns:
            dict: Formatted response with role and memory
        """
        if isinstance(output, OrderResponse):
            order = [item.dict() for item in output.order]
            step_number = output.step_number
            response = output.response
        else:
            order = output["order"]
            step_number = output["step number"]
            response = output["response"]
        
        # Add recommendations if needed
        if not asked_recommendation_before and len(order) > 0:
            recommendation_output = self.recommendation_agent.get_recommendations_from_order(
                messages, order
            )
            response = recommendation_output['content']
            asked_recommendation_before = True
        
        return {
            "role": "assistant",
            "content": response,
            "memory": {
                "agent": "order_taking_agent",
                "step number": step_number,
                "order": order,
                "asked_recommendation_before": asked_recommendation_before
            }
        }

    