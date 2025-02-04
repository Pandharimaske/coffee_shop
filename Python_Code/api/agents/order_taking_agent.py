import os
import ollama
import json
from copy import deepcopy
from .utils import get_chatbot_response , get_embedding , double_check_json_output
import dotenv
dotenv.load_dotenv()
import pandas as pd



class OrderTakingAgent():

    def __init__(self , recommendation_agent):
        
        self.model_name = os.getenv("MODEL_NAME")
        self.recommendation_agent = recommendation_agent

    def get_response(self , messages):
        messages = deepcopy(messages)

        # system_prompt = """
        #     You are a customer support Bot for a coffee shop called "Merry's way"

        #     here is the menu for this coffee shop.

        #     Cappuccino - $4.50
        #     Jumbo Savory Scone - $3.25
        #     Latte - $4.75
        #     Chocolate Chip Biscotti - $2.50
        #     Espresso shot - $2.00
        #     Hazelnut Biscotti - $2.75
        #     Chocolate Croissant - $3.75
        #     Dark chocolate (Drinking Chocolate) - $5.00
        #     Cranberry Scone - $3.50
        #     Croissant - $3.25
        #     Almond Croissant - $4.00
        #     Ginger Biscotti - $2.50
        #     Oatmeal Scone - $3.25
        #     Ginger Scone - $3.50
        #     Chocolate syrup - $1.50
        #     Hazelnut syrup - $1.50
        #     Carmel syrup - $1.50
        #     Sugar Free Vanilla syrup - $1.50
        #     Dark chocolate (Packaged Chocolate) - $3.00

        #     Things to NOT DO:
        #     * DON't ask how to pay by cash or Card.
        #     * Don't tell the user to go to the counter
        #     * Don't tell the user to go to place to get the order


        #     You're task is as follows:
        #     1. Take the User's Order
        #     2. Validate that all their items are in the menu
        #     3. if an item is not in the menu let the user and repeat back the remaining valid order
        #     4. Ask them if they need anything else.
        #     5. If they do then repeat starting from step 3
        #     6. If they don't want anything else. Using the "order" object that is in the output. Make sure to hit all three points
        #         1. list down all the items and their prices
        #         2. calculate the total. 
        #         3. Thank the user for the order and close the conversation with no more questions

        #     The user message will contain a section called memory. This section will contain the following:
        #     "order"
        #     "step number"
        #     please utilize this information to determine the next step in the process.
            
        #     produce the following output without any additions, not a single letter outside of the structure bellow.
        #     Your output should be in a structured json format like so. each key is a string and each value is a string. Make sure to follow the format exactly:
        #     {
        #     "chain of thought": Write down your critical thinking about what is the maximum task number the user is on write now. Then write down your critical thinking about the user input and it's relation to the coffee shop process. Then write down your thinking about how you should respond in the response parameter taking into consideration the Things to NOT DO section. and Focus on the things that you should not do. 
        #     "step number": Determine which task you are on based on the conversation.
        #     "order": this is going to be a list of jsons like so. [{"item":put the item name, "quanitity": put the number that the user wants from this item, "price":put the total price of the item }]
        #     "response": write the a response to the user
        #     }
        # """

        system_prompt = """
        You are a customer support bot for a coffee shop called "Merry's Way."

        Here is the menu for this coffee shop:

        - Cappuccino - $4.50  
        - Jumbo Savory Scone - $3.25  
        - Latte - $4.75  
        - Chocolate Chip Biscotti - $2.50  
        - Espresso Shot - $2.00  
        - Hazelnut Biscotti - $2.75  
        - Chocolate Croissant - $3.75  
        - Dark Chocolate (Drinking Chocolate) - $5.00  
        - Cranberry Scone - $3.50  
        - Croissant - $3.25  
        - Almond Croissant - $4.00  
        - Ginger Biscotti - $2.50  
        - Oatmeal Scone - $3.25  
        - Ginger Scone - $3.50  
        - Chocolate Syrup - $1.50  
        - Hazelnut Syrup - $1.50  
        - Caramel Syrup - $1.50  
        - Sugar-Free Vanilla Syrup - $1.50  
        - Dark Chocolate (Packaged Chocolate) - $3.00  

        ### 🚫 **Things You MUST NOT Do:**  
        - Don’t ask how to pay by cash or card.  
        - Don’t tell the user to go to the counter.  
        - Don’t tell the user where to pick up their order.  

        ---

        ### ✅ **Your Task:**  
        1. Take the user’s order.  
        2. Validate that all items are on the menu.  
        3. If an item is not on the menu, inform the user and repeat the valid remaining order.  
        4. Ask if they need anything else.  
        5. If they do, repeat from Step 3.  
        6. If they don’t want anything else:  
        - List all items and their prices.  
        - Calculate the total cost.  
        - Thank the user and close the conversation without asking any further questions.  

        ---

        ### 🧠 **Important Notes:**  
        - The user's message will include a **"memory"** section with:  
        - `"order"`: A list of current items in the order.  
        - `"step number"`: The current step in the process.  
        - Use this information to determine the next step in the conversation.  

        ---

        ### 🗂️ **Output Format:**  
        Respond strictly in the following JSON format—nothing extra, not even a single character outside this structure:

        ```json
        {
        "chain of thought": "Analyze the conversation to determine the current step based on the user's input. Explain how the input relates to the coffee shop process. Reflect on how to respond appropriately while strictly avoiding the actions listed in the 'Things You MUST NOT Do' section.",
        "step number": "Determine which step you are on based on the conversation.",
        "order": "[{\"item\": \"Item Name\", \"quantity\": \"Quantity\", \"price\": \"Total Price\"}]",
        "response": "Write a polite, concise response to the user."
        }
        """


        last_order_taking_status = ""
        asked_recommendation_before = False
        for message_index in range(len(messages)-1,0,-1):
            message = messages[message_index]
            
            agent_name = message.get("memory",{}).get("agent","")
            if message["role"] == "assistant" and agent_name == "order_taking_agent":
                step_number = message["memory"]["step number"]
                order = message["memory"]["order"]
                asked_recommendation_before = message["memory"]["asked_recommendation_before"]
                last_order_taking_status = f"""
                step number: {step_number}
                order: {order}
                """
                break
            
        messages[-1]["content"] = last_order_taking_status + "\n" + messages[-1]["content"]
        input_messages = [{"role":"system" , "content": system_prompt}] + messages

        chatbot_output = get_chatbot_response(self.model_name , input_messages)


        chatbot_output = double_check_json_output(self.model_name , chatbot_output)

        output = self.postprocess(chatbot_output , input_messages)

        return output
    
    def postprocess(self,output,messages,asked_recommendation_before):
        output = json.loads(output)

        if type(output["order"]) == str:
            output["order"] = json.loads(output["order"])

        response = output['response']
        if not asked_recommendation_before and len(output["order"])>0:
            recommendation_output = self.recommendation_agent.get_recommendations_from_order(messages,output['order'])
            response = recommendation_output['content']
            asked_recommendation_before = True

        dict_output = {
            "role": "assistant",
            "content": response ,
            "memory": {"agent":"order_taking_agent",
                       "step number": output.get("step number" , 1),
                       "order": output["order"],
                       "asked_recommendation_before": asked_recommendation_before
                      }
        }

        return dict_output
    

    





