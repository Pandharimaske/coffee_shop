from typing import Protocol, List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentMemory(BaseModel):
    """Base schema for agent memory"""
    agent: str = Field(description="Name of the agent")
    step_number: Optional[int] = Field(description="Current step in the process", default=None)
    order: Optional[List[Dict[str, Any]]] = Field(description="Current order items", default=None)
    guard_decision: Optional[str] = Field(description="Guard agent's decision", default=None)
    asked_recommendation_before: Optional[bool] = Field(description="Whether recommendations were given", default=None)

class AgentResponse(BaseModel):
    """Base schema for agent response"""
    role: str = Field(description="Role of the response (assistant)")
    content: str = Field(description="Content of the response")
    memory: AgentMemory = Field(description="Agent's memory state")

class AgentProtocol(Protocol):
    """Protocol defining the interface for all agents in the coffee shop chatbot"""
    
    def get_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a response from the agent based on the conversation history.
        
        Args:
            messages: List of conversation messages, where each message is a dictionary
                     containing 'role' and 'content' keys.
        
        Returns:
            Dictionary containing:
            - role: str - The role of the response (assistant)
            - content: str - The response content
            - memory: Dict[str, Any] - Agent's memory state including:
                - agent: str - Name of the agent
                - step_number: Optional[int] - Current step in the process
                - order: Optional[List[Dict]] - Current order items
                - guard_decision: Optional[str] - Guard agent's decision
                - asked_recommendation_before: Optional[bool] - Whether recommendations were given
        """
        ...