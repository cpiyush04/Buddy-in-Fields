import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings



class ContentProcessor:
    """
    Processes the parsed content - summarizes images, creates llm based semantic chunks
    """
    def __init__(self, config):
        """
        Initialize the response generator.
        
        Args:
            llm: Large language model for image summarization
        """
        self.logger = logging.getLogger(__name__)
        # self.summarizer_model = config.rag.summarizer_model
        
        # --- NEW: Load BLIP Model Locally (Run once) ---
        print("Loading BLIP image captioning model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        # -----------------------------------------------
        # self.chunker_model = config.rag.chunker_model     # temperature 0.0

        # --- ADD THIS (Load once, runs locally) ---
        print("Loading local embedding model for chunking...")
        self.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize the splitter with a percentile threshold (adaptive splitting)
        self.semantic_splitter = SemanticChunker(
            self.embed_model, 
            breakpoint_threshold_type="percentile"
        )

    def summarize_images(self, images: List[str]) -> List[str]:
        """
        Summarize images using local BLIP model.
        Fast, free, and efficient.
        """
        results = []
        
        for image_path in images:
            try:
                # 1. Open the image (handles local paths)
                # If your 'images' are URLs, you might need requests.get(image_path, stream=True).raw
                raw_image = Image.open(image_path).convert('RGB')

                # 2. Preprocess
                inputs = self.blip_processor(raw_image, return_tensors="pt").to(self.device)

                # 3. Generate Caption
                # max_new_tokens=50 is enough for a concise description
                out = self.blip_model.generate(**inputs, max_new_tokens=50)

                # 4. Decode
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                
                # Append formatted result
                results.append(f"Image content: {caption}")

            except Exception as e:
                # Log error
                print(f"Error processing image {image_path}: {str(e)}")
                
                # Fallback return string (matches your previous 'non-informative' logic)
                results.append("no image summary")

        return results
    
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        Format the parsed document by replacing image placeholders with image summaries.
        
        Args:
            parsed_document: Parsed document from doc_parser
            image_summaries: List of image summaries
            
        Returns:
            Formatted document text with image summaries
        """
        IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
        PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
        
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )
        
        formatted_document = self._replace_occurrences(
            formatted_parsed_document, 
            IMAGE_PLACEHOLDER, 
            image_summaries
        )
        
        return formatted_document
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        Replace occurrences of a target placeholder with corresponding replacements.
        
        Args:
            text: Text containing placeholders
            target: Placeholder to replace
            replacements: List of replacements for each occurrence
            
        Returns:
            Text with replacements
        """
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(
                        target, 
                        f'picture_counter_{counter}' + ' ' + replacement, 
                        1
                    )
                else:
                    result = result.replace(target, '', 1)
            else:
                # Instead of raising an error, just break the loop when no more occurrences are found
                break
        
        return result
    
    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        Split the document into semantic chunks using local embeddings.
        Fast, Free, and No LLM required.
        """
        try:
            # 1. The semantic splitter handles the logic internally
            # It calculates embeddings for sentences and splits where topics change
            docs = self.semantic_splitter.create_documents([formatted_document])
            
            # 2. Extract the text content from the document objects
            chunks = [doc.page_content for doc in docs]
            
            # 3. (Optional but recommended) Verification
            # If a chunk is too small (noise), you might want to filter it, 
            # but usually, the splitter is good enough.
            return chunks

        except Exception as e:
            print(f"Error in semantic chunking: {e}")
            # Fallback: If the fancy splitter fails, just split by double newlines
            return formatted_document.split("\n\n")