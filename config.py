import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# Load environment variables from .env file
load_dotenv()

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            # deployment_name = os.getenv("google_genai_deployment_name"),  # Replace with your Google GenAI deployment name
            model = os.getenv("google_genai_model_name"),  # Replace with your Google GenAI model name
            temperature = 0.1,  # Deterministic,
            google_api_key = os.getenv("google_genai_api_key")  # Replace with your Google GenAI API key
        )
       

class ConversationConfig:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model = os.getenv("google_genai_model_name"),  # Replace with your Google GenAI model name
            temperature = 0.7,  # Deterministic,
            google_api_key = os.getenv("google_genai_api_key")  # Replace with your Google GenAI API key
        )
        

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            # deployment_name = os.getenv("google_genai_deployment_name"),  # Replace with your Google GenAI deployment name
            model = os.getenv("google_genai_model_name"),  # Replace with your Google GenAI model name
            temperature = 0.3,  # Deterministic,
            google_api_key = os.getenv("google_genai_api_key")  # Replace with your Google GenAI API key
        )
        
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 384  # Add the embedding dimension here
        self.distance_metric = "Cosine"  # Add this with a default value
        self.use_local = True  # Add this with a default value
        self.vector_local_path = "./data/qdrant_db"  # Add this with a default value
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"  # Ensure a valid name
        self.chunk_size = 512  # Modify based on documents and performance
        self.chunk_overlap = 50  # Modify based on documents and performance
        
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.llm = ChatGoogleGenerativeAI(
            # deployment_name = os.getenv("google_genai_deployment_name"),  # Replace with your Google GenAI deployment name
            model = os.getenv("google_genai_model_name"),  # Replace with your Google GenAI model name
            temperature = 0.3,  # Slightly creative but factual
            google_api_key = os.getenv("google_genai_api_key")  # Replace with your Google GenAI API key
        )
        
        self.summarizer_model = ChatGoogleGenerativeAI(
            # deployment_name = os.getenv("google_genai_deployment_name"),  # Replace with your Google GenAI deployment name
            model = os.getenv("google_genai_model_name"),  # Replace with your Google GenAI model name
            temperature = 0.5,  # Slightly creative but factual
            google_api_key = os.getenv("google_genai_api_key")  # Replace with your Google GenAI API key
        )
       
        self.chunker_model = ChatGoogleGenerativeAI(
            # deployment_name = os.getenv("google_genai_deployment_name"),  # Replace with your Google GenAI deployment name
            model = os.getenv("google_genai_model_name"),  # Replace with your Google GenAI model name
            temperature = 0.0,  # factual
            api_key = os.getenv("google_genai_api_key")  # Replace with your Google GenAI API key
        )
       
        self.response_generator_model = ChatGoogleGenerativeAI(
            # deployment_name = os.getenv("google_genai_deployment_name"),  # Replace with your Google GenAI deployment name
            model = os.getenv("google_genai_model_name"),  # Replace with your Google GenAI model name
            temperature = 0.3,  # Slightly creative but factual
            google_api_key = os.getenv("google_genai_api_key")  # Replace with your Google GenAI API key
        )
    
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        # self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history


class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "PLANT_DISEASE_AGENT": True,  # Critical health info needs validation
            "PEST_AGENT": True,         # Pest ID needs validation
            "GENERAL_AGENT": False        # General description is lower risk
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        # self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20
        self.plant_disease_model_name = "wambugu71/crop_leaf_diseases_vit"
        self.pest_model_name = "dima806/farm_insects_image_detection"
        self.general_vqa_model_name = "gemini-3-flash-preview"
        self.google_api_key = os.getenv("google_genai_api_key")

# # Example usage
config = Config()