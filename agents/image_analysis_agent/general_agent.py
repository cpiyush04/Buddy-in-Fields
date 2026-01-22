import base64
from mimetypes import guess_type
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import config

class GeneralImageAgent:
    """
    Uses Google's Gemini Flash to analyze general farming images 
    (equipment, soil, unknown objects) and answer user questions.
    """
    
    def __init__(self):
        # Ensure config.py has: self.general_vqa_model_name = "gemini-1.5-flash"
        self.model_name = config.general_vqa_model_name
        self.api_key = config.google_api_key
        self.model = None

    def _load_model(self):
        """Lazy load the Gemini Flash model."""
        if self.model is None:
            print(f"⚡ Loading General Agent ({self.model_name})...")
            self.model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.3
            )

    def _local_image_to_data_url(self, image_path: str) -> str:
        """Helper: Convert local image file to Base64 string."""
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"

    def analyze(self, image_path: str, user_query: str = "Describe this image in detail."):
        """
        Analyzes image using Gemini Flash.
        """
        try:
            self._load_model()
            
            # 1. Convert Image
            image_data_url = self._local_image_to_data_url(image_path)
            
            # 2. Prepare Message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": user_query},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    }
                ]
            )
            
            # 3. Call Gemini
            response = self.model.invoke([message])
            
            return {
                "answer": response.content,
                "context": f"Visual analysis by {self.model_name}"
            }

        except Exception as e:
            return {"error": f"Gemini Vision Error: {str(e)}"}