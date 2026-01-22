from transformers import pipeline
from config import config

class PestAgent:
    def __init__(self):
        self.model_name = config.pest_model_name
        self.pipeline = None

    def _load_model(self):
        if self.pipeline is None:
            print(f"⚡ Loading Pest Model ({self.model_name})...")
            self.pipeline = pipeline("image-classification", model=self.model_name)

    def analyze(self, image_path: str):
        try:
            self._load_model()
            results = self.pipeline(image_path)
            top_result = results[0]
            
            return {
                "pest_type": top_result['label'],
                "confidence": round(top_result['score'], 4),
                "details": f"Identified pest: {top_result['label']}."
            }
        except Exception as e:
            return {"error": f"Pest Agent Error: {str(e)}"}