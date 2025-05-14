import json
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from copy import deepcopy
from dotenv import load_dotenv
from .utils import get_chatbot_response, double_check_json_output
load_dotenv()

class RecommendationType(BaseModel):
    """Schema for recommendation classification"""
    chain_of_thought: str = Field(description="Reasoning about the recommendation type")
    recommendation_type: str = Field(description="Type of recommendation: apriori, popular, or popular by category")
    parameters: List[str] = Field(description="List of items or categories for recommendations")

class RecommendationAgent():
    def __init__(self,apriori_recommendation_path,popular_recommendation_path):
        self.client = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME"),
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.model_name = os.getenv("MODEL_NAME")

        with open(apriori_recommendation_path, 'r') as file:
            self.apriori_recommendations = json.load(file)

        self.popular_recommendations = pd.read_csv(popular_recommendation_path)
        self.products = self.popular_recommendations['product'].tolist()
        self.product_categories = self.popular_recommendations['product_category'].tolist()
    
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant for a coffee shop application which serves drinks and pastries. We have 3 types of recommendations:

            1. Apriori Recommendations: Based on user's order history, recommending items frequently bought together
            2. Popular Recommendations: Based on overall popularity of items in the coffee shop
            3. Popular Recommendations by Category: Based on popularity within specific categories
            
            Available Products: {products}
            Available Categories: {categories}
            
            Determine the recommendation type based on the user's message.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant for a coffee shop application.
            Recommend items to the user based on their input message.
            Respond in a friendly but concise way using an unordered list with brief descriptions.
            
            Items to recommend: {recommendations}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.classification_chain = LLMChain(
            llm=self.client,
            prompt=self.classification_prompt,
            memory=self.memory,
            verbose=True
        )
        
        self.response_chain = LLMChain(
            llm=self.client,
            prompt=self.response_prompt,
            memory=self.memory,
            verbose=True
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=RecommendationType)

    def get_apriori_recommendation(self,products,top_k=5):
        recommendation_list = []
        for product in products:
            if product in self.apriori_recommendations:
                recommendation_list += self.apriori_recommendations[product]
        
        # Sort recommendation list by "confidence"
        recommendation_list = sorted(recommendation_list,key=lambda x: x['confidence'],reverse=True)

        recommendations = []
        recommendations_per_category = {}
        for recommendation in recommendation_list:
            # If Duplicated recommendations then skip
            if recommendation in recommendations:
                continue 

            # Limit 2 recommendations per category
            product_catory = recommendation['product_category']
            if product_catory not in recommendations_per_category:
                recommendations_per_category[product_catory] = 0
            
            if recommendations_per_category[product_catory] >= 2:
                continue

            recommendations_per_category[product_catory]+=1

            # Add recommendation
            recommendations.append(recommendation['product'])

            if len(recommendations) >= top_k:
                break

        return recommendations 

    def get_popular_recommendation(self,product_categories=None,top_k=5):
        recommendations_df = self.popular_recommendations
        
        if type(product_categories) == str:
            product_categories = [product_categories]

        if product_categories is not None:
            recommendations_df = self.popular_recommendations[self.popular_recommendations['product_category'].isin(product_categories)]
        recommendations_df = recommendations_df.sort_values(by='number_of_transactions',ascending=False)
        
        if recommendations_df.shape[0] == 0:
            return []

        recommendations = recommendations_df['product'].tolist()[:top_k]
        return recommendations

    def recommendation_classification(self, messages):
        """Classify the type of recommendation needed"""
        try:
            # Get response from classification chain
            chain_response = self.classification_chain.predict(
                input=messages[-1]['content'],
                products=", ".join(self.products),
                categories=", ".join(self.product_categories)
            )
            
            # Parse the response
            parsed_response = self.output_parser.parse(chain_response)
            return {
                "recommendation_type": parsed_response.recommendation_type,
                "parameters": parsed_response.parameters
            }
            
        except Exception as e:
            # Fallback to JSON parsing if Pydantic parsing fails
            json_response = double_check_json_output(
                self.client,
                os.getenv("GROQ_MODEL_NAME"),
                chain_response
            )
            return json.loads(json_response)

    def get_response(self, messages):
        """Get recommendation response"""
        messages = deepcopy(messages)
        
        # Get recommendation classification
        classification = self.recommendation_classification(messages)
        recommendation_type = classification['recommendation_type']
        
        # Get recommendations based on type
        recommendations = []
        if recommendation_type == "apriori":
            recommendations = self.get_apriori_recommendation(classification['parameters'])
        elif recommendation_type == "popular":
            recommendations = self.get_popular_recommendation()
        elif recommendation_type == "popular by category":
            recommendations = self.get_popular_recommendation(classification['parameters'])
        
        if not recommendations:
            return {
                "role": "assistant",
                "content": "Sorry, I can't help with that. Can I help you with your order?",
                "memory": {"agent": "recommendation_agent"}
            }
        
        # Generate response using response chain
        response = self.response_chain.predict(
            input=messages[-1]['content'],
            recommendations=", ".join(recommendations)
        )
        
        return {
            "role": "assistant",
            "content": response,
            "memory": {"agent": "recommendation_agent"}
        }

    def postprocess_classfication(self,output):
        output = json.loads(output)

        dict_output = {
            "recommendation_type": output['recommendation_type'],
            "parameters": output['parameters'],
        }
        return dict_output

    def get_recommendations_from_order(self,messages,order):
        products = []
        for product in order:
            products.append(product['item'])

        recommendations = self.get_apriori_recommendation(products)
        recommendations_str = ", ".join(recommendations)

        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        your task is to recommend items to the user based on their order.

        I will provide what items you should recommend to the user based on their order in the user message. 
        """

        prompt = f"""
        {messages[-1]['content']}

        Please recommend me those items exactly: {recommendations_str}
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
        output = self.postprocess(chatbot_output)

        return output
    
    def postprocess(self,output):
        output = {
            "role": "assistant",
            "content": output,
            "memory": {"agent":"recommendation_agent"
                      }
        }
        return output

