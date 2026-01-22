from transformers import pipeline
from config import config

class PlantDiseaseAgent:
    def __init__(self):
        # LIGHTWEIGHT: Just store the name, don't load weights yet
        self.model_name = config.plant_disease_model_name
        self.pipeline = None 

    def _load_model(self):
        """Loads the model only when absolutely necessary."""
        if self.pipeline is None:
            print(f"⚡ Loading Plant Disease Model ({self.model_name})...")
            self.pipeline = pipeline("image-classification", model=self.model_name)

    def analyze(self, image_path: str):
        try:
            # 1. Lazy Load (First time only)
            self._load_model()
            
            # 2. Run Inference
            results = self.pipeline(image_path)
            top_result = results[0]
            
            return {
                "diagnosis": top_result['label'],
                "confidence": round(top_result['score'], 4),
                "details": f"Detected {top_result['label']} ({round(top_result['score']*100, 1)}% confidence)."
            }
        except Exception as e:
            return {"error": f"Plant Agent Error: {str(e)}"}