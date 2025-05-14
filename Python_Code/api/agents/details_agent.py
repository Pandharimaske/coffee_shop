from dotenv import load_dotenv
import os
from .utils import get_embedding
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from copy import deepcopy
from pinecone import Pinecone as PineconeClient
load_dotenv()

class DetailsAgent:
    def __init__(self):
        # Initialize Groq client
        self.client = ChatGroq(
            model_name=os.getenv("GROQ_MODEL_NAME"),
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize Pinecone client
        self.pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("HF_EMBEDDING_MODEL"),
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create the system prompt template
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly and knowledgeable customer support agent for Merry's Way coffee shop.
            Your role is to:
            1. Provide accurate information about the coffee shop
            2. Answer questions about menu items, prices, and availability
            3. Share details about location, hours, and services
            4. Maintain a warm, welcoming tone
            5. Use the provided context to give precise answers
            
            Always be helpful and professional while maintaining the personality of a friendly waiter."""),
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
    
    def get_closest_results(self, index_name, input_embeddings, top_k=2):
        """
        Retrieve the most relevant documents from Pinecone.
        
        Args:
            index_name (str): Name of the Pinecone index
            input_embeddings (list): Query embeddings
            top_k (int): Number of results to return
            
        Returns:
            list: List of relevant documents with metadata
        """
        index = self.pc.Index(index_name)
        
        results = index.query(
            namespace="ns1",
            vector=input_embeddings,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        return results

    def get_response(self, messages):
        messages = deepcopy(messages)
        user_message = messages[-1]['content']
        
        # Get embeddings for the query
        embedding = self.embedding_model.embed_query(user_message)
        
        # Get relevant documents
        results = self.get_closest_results(self.index_name, embedding)
        
        # Format the context from retrieved documents
        source_knowledge = "\n".join([
            f"Context {i+1}: {match['metadata']['text'].strip()}"
            for i, match in enumerate(results['matches'])
        ])
        
        # Create the prompt with context
        prompt = f"""Using the following contexts, answer the user's query accurately and concisely.
        If the contexts don't contain enough information, say so politely.

        Contexts:
        {source_knowledge}

        User Query: {user_message}
        """
        
        # Get response from the chain
        response = self.chain.predict(input=prompt)
        
        return self.postprocess(response)

    def postprocess(self, output):
        """
        Format the response into the expected structure.
        
        Args:
            output (str): The model's response
            
        Returns:
            dict: Formatted response with role and memory
        """
        return {
            "role": "assistant",
            "content": output,
            "memory": {
                "agent": "details_agent"
            }
        }

    