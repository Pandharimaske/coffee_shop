import os
import ollama
import json
from copy import deepcopy
from .utils import get_chatbot_response , get_embedding , double_check_json_output
import dotenv
dotenv.load_dotenv()
import pandas as pd


class RecommendationAgent():

    def __init__(self , apriori_recommendation_path , popular_recommendation_path):
        
        self.model_name = os.getenv("MODEL_NAME")

        with open(apriori_recommendation_path , "r") as file:
            self.apriori_recommendations = json.load(file)
        
        self.popular_recommendations = pd.read_csv(popular_recommendation_path)
        self.products = self.popular_recommendations["product"].tolist()
        self.product_categories = list(set(self.popular_recommendations["product_category"].tolist()))

    def get_apriori_recommendation(self , products , top_k = 5):
        recommendation_list = []

        for product in products:
            if product in self.apriori_recommendations:
                recommendation_list += self.apriori_recommendations[product]
        
        # Sort recommendation list by "confidence"
        recommendation_list = sorted(recommendation_list , key=lambda x: x["confidence"] , reverse=True)

        recommendations = []
        recommendations_per_category = {}
        for recommendation in recommendation_list:
            if recommendation in recommendations:
                continue

            # Limits 2 recommendations per category
            product_category = recommendation["product_category"]
            if product_category not in recommendations_per_category:
                recommendations_per_category[product_category] = 0
            
            if recommendations_per_category[product_category] >= 2:
                continue

            recommendations_per_category[product_category] += 1

            # Add recommendations
            recommendations.append(recommendation["product"])

            if len(recommendations) >= top_k:
                break
        
        return recommendations

    
    def get_popular_recommendation(self , product_categories = None , top_k = 5):
        recommendation_df = self.popular_recommendations

        if type(product_categories) == str:
            product_categories = [product_categories]
        
        if product_categories is not None:
            recommendation_df = self.popular_recommendations[self.popular_recommendations["product_category"].isin(product_categories)]
        recommendation_df = recommendation_df.sort_values("number_of_transactions" , ascending=False)

        if recommendation_df.shape[0] == 0:
            return []
        
        recommendations = recommendation_df["product"].tolist()[:top_k]
        return recommendations
    

    def recommendation_classification(self , message):
        # system_prompt = """ You are a helpful AI assistant for a coffee shop application which serves drinks and pastries. We have 3 types of recommendations:

        # 1. Apriori Recommendations: These are recommendations based on the user's order history. We recommend items that are frequently bought together with the items in the user's order.
        # 2. Popular Recommendations: These are recommendations based on the popularity of items in the coffee shop. We recommend items that are popular among customers.
        # 3. Popular Recommendations by Category: Here the user asks to recommend them product in a category. Like what coffee do you recommend me to get?. We recommend items that are popular in the category of the user's requested category.
        
        # Here is the list of items in the coffee shop:
        # """+ ",".join(self.products) + """
        # Here is the list of Categories we have in the coffee shop:
        # """ + ",".join(self.product_categories) + """

        # Your task is to determine which type of recommendation to provide based on the user's message.

        # Your output should be in a structured json format like so. Each key is a string and each value is a string. Make sure to follow the format exactly:
        # {
        # "chain of thought": "Write down your critical thinking about what type of recommendation is this input relevant to." , 
        # "recommendation_type": "apriori" or "popular" or "popular by category". Pick one of those and only write the word.
        # "parameters": This is a  python list. It's either a list of of items for apriori recommendations or a list of categories for popular by category recommendations. Leave it empty for popular recommendations. Make sure to use the exact strings from the list of items and categories above.
        # }
        # """

        system_prompt = """
        You are a helpful AI assistant for a coffee shop application that serves drinks and pastries.

        ### 📋 **Types of Recommendations:**  
        1. **Apriori Recommendations:**  
        - Based on the user's order history.  
        - Recommend items that are frequently bought together with items from the user's current order.  

        2. **Popular Recommendations:**  
        - Based on overall item popularity in the coffee shop.  
        - Recommend items that are most popular among all customers.  

        3. **Popular Recommendations by Category:**  
        - Based on category-specific popularity.  
        - When a user asks, "What coffee do you recommend?" suggest the most popular items in that category.  

        ---

        ### ☕ **Menu Items:**  
        """ + ",".join(self.products) + """

        ### 📂 **Product Categories:**  
        """ + ",".join(self.product_categories) + """

        ---

        ### 🚀 **Your Task:**  
        Determine which type of recommendation to provide based on the user's message.

        ---

        ### 🗂️ **Output Format:**  
        Respond strictly in the following JSON format—no extra text, only this structure:

        ```json
        {
        "chain of thought": "Explain your reasoning for determining the recommendation type based on the user's message.",
        "recommendation_type": "apriori" or "popular" or "popular by category",
        "parameters": "A Python list. For 'apriori', include the relevant items from the user's order. For 'popular by category', include the relevant categories. Leave it empty for 'popular' recommendations."
        }
        """

        input_messages = [{"role":"system" , "content":system_prompt}] + message[-3:]

        chatbot_output = get_chatbot_response(self.model_name , input_messages)
        chatbot_output = double_check_json_output(self.model_name , chatbot_output)

        output = self.postprocess_classification(chatbot_output)


        return output
    

    def get_response(self,messages):
        messages = deepcopy(messages)

        recommendation_classification = self.recommendation_classification(messages)
        recommendation_type = recommendation_classification['recommendation_type']
        recommendations = []
        if recommendation_type == "apriori":
            recommendations = self.get_apriori_recommendation(recommendation_classification['parameters'])
        elif recommendation_type == "popular":
            recommendations = self.get_popular_recommendation()
        elif recommendation_type == "popular by category":
            recommendations = self.get_popular_recommendation(recommendation_classification['parameters'])
        
        if recommendations == []:
            return {"role": "assistant", "content":"Sorry, I can't help with that. Can I help you with your order?"}
        
        # Respond to User
        recommendations_str = ", ".join(recommendations)
        
        # system_prompt = f"""
        # You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        # your task is to recommend items to the user based on their input message. And respond in a friendly but concise way. And put it an unordered list with a very small description.

        # I will provide what items you should recommend to the user based on their order in the user message. 
        # """

        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application that serves drinks and pastries.

        🎯 **Your Task:**  
        - Recommend items to the user based on their input message.  
        - Respond in a friendly, engaging, yet concise manner.  
        - Present recommendations as an unordered list (•) with a brief, appealing description for each item.

        💡 **Note:**  
        I will provide the specific items you should recommend based on the user's order in the input message.
        """

        prompt = f"""
        {messages[-1]['content']}

        Please recommend me those items exactly: {recommendations_str}
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.model_name,input_messages)
        output = self.postprocess(chatbot_output)

        return output
    
    
    def postprocess_classification(self , output):
        output = json.loads(output)
        dict_output = {
            "recommendation_type": output['recommendation_type'],
            "parameters": output['parameters']
        }
        return dict_output
    
    def get_recommendations_from_order(self , messages , order):
        messages = deepcopy(messages)
        products = []
        for product in order:
            products.append(product["item"])
        
        recommendations = self.get_apriori_recommendation(products)
        recommendations_str = ",".join(recommendations)

        # system_prompt = f"""
        # You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        # your task is to recommend items to the user based on their order.

        # I will provide what items you should recommend to the user based on their order in the user message. 
        # """

        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application that serves drinks and pastries.

        🎯 **Your Task:**  
        - Recommend items to the user based on their order.  
        - Keep your responses friendly, concise, and relevant.

        💡 **Note:**  
        The specific items to recommend will be provided in the user's message.
        """

        prompt = f"""
        {messages[-1]['content']}

        Please recommend me those items exactly: {recommendations_str}
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.model_name,input_messages)
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
    
    



