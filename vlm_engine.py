import os
import time
import random
import json
import re
from PIL import Image
from google import genai
from google.genai import types

class VLMEngine:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initializes the Gemini VLM Engine with dual models for hybrid processing using the new SDK.
        """
        self.api_key = api_key
        # Initialize the client
        self.client = genai.Client(api_key=api_key)
        
        # Model names
        self.lite_model_name = "gemini-2.5-flash-lite"
        if "lite" in model_name:
             self.lite_model_name = model_name
        self.pro_model_name = "models/gemini-2.5-pro"
        
        self.default_model_name = model_name
        
        # Context from reference books
        self.reference_cache_name = None
        
        # Try to load cache info
        self.load_cache_info()

    def load_cache_info(self):
        """Loads cache name from .cache_info.json."""
        cache_file = ".cache_info.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    cache_name = data.get("cache_name")
                    if cache_name:
                        self.reference_cache_name = cache_name
                        print(f"Using Cached Content: {cache_name}")
                        return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        self.reference_cache_name = None
        return False

    def load_reference_books(self, directory_path: str):
        """
        Loads and returns a list of PDF filenames from the directory.
        Note: The actual content is handled by Context Caching, this is mainly for UI display.
        """
        if not os.path.exists(directory_path):
            return []
        
        loaded_files = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                loaded_files.append(filename)
        return loaded_files

    def extract_math_problems(self, image: Image.Image) -> str:
        """
        [Stage 1: Lite] Extracts structured problem data (no solutions).
        """
        extract_prompt = (
            "You are a math problem extractor. Extract problems from the image in XML format:\n"
            "1. Each problem in `<problem>` tags.\n"
            "2. Full problem content (text + LaTeX) in `<content>`. \n"
            "   - **IMPORTANT**: Euclidean geometry, calculus, and algebra expressions MUST be in LaTeX format.\n"
            "   - Use `$` for inline math (e.g., $x^2$, $f(x)$) and `$$` for block math.\n"
            "   - Use standard LaTeX commands: `\\frac` for fractions, `\\lim_{x \\to 0}` for limits, `\\sqrt` for roots.\n"
            "3. Separate tags for `<number>`, `<options>`, `<score>`, and `<image_description>`.\n"
            "\nExample:\n"
            "<problem>\n"
            "  <number>1</number>\n"
            "  <content>Determine the value of $\\lim_{h \\to 0} \\frac{f(4+h)-f(4)}{h}$.</content>\n"
            "  <options><option>A</option></options>\n"
            "  <image_description>... or None</image_description>\n"
            "</problem>"
        )
        try:
            # New SDK call
            response = self.client.models.generate_content(
                model=self.lite_model_name,
                contents=[extract_prompt, image]
            )
            return response.text
        except Exception as e:
            return f"Error during extraction: {e}"

    def solve_problem(self, problem_text: str, stream: bool = False, image: Image.Image = None, use_cache: bool = True):
        """
        [Stage 2: Pro] Generates a clear and step-by-step solution.
        """
        solve_prompt = (
            f"Please solve the following math problem:\n\n{problem_text}\n\n"
            "Instructions:\n"
            "1. Use LaTeX ($$) for all mathematical expressions.\n"
            "2. Provide a clear, concise step-by-step logical explanation.\n"
            "3. Append a difficulty analysis in the following XML format at the end:\n"
            "<analysis>\n"
            "  <level>Integer 1-6</level>\n"
            "  <conceptual_reason>Explanation</conceptual_reason>\n"
            "  <logical_reason>Explanation</logical_reason>\n"
            "  <computational_reason>Explanation</computational_reason>\n"
            "  <summary>Overall comment</summary>\n"
            "</analysis>\n"
            "4. Provide the final numeric or algebraic answer inside `<final_answer>value</final_answer>` tags.\n"
        )
        
        if image:
             solve_prompt = (
                 "Please use the attached image to verify the problem statement and numbers before solving. "
                 "The image content is the primary source of truth.\n\n"
                 + solve_prompt
             )
        
        contents = [solve_prompt]
        if image:
            contents.append(image)

        try:
            if self.reference_cache_name and use_cache:
                try:
                    # Check .cache_info.json for model name
                    cached_model = "models/gemini-2.5-pro" # Default fallback
                    if os.path.exists(".cache_info.json"):
                        with open(".cache_info.json", "r") as f:
                            data = json.load(f)
                            cached_model = data.get("model_name", cached_model)

                    print(f"Solving with cached content: {self.reference_cache_name}")
                    
                    if stream:
                        response = self.client.models.generate_content_stream(
                            model=cached_model,
                            contents=contents,
                            config=types.GenerateContentConfig(cached_content=self.reference_cache_name)
                        )
                    else:
                        response = self.client.models.generate_content(
                            model=cached_model,
                            contents=contents,
                            config=types.GenerateContentConfig(cached_content=self.reference_cache_name)
                        )
                except Exception as e:
                    # 403 error or CachedContent not found
                    if "403" in str(e) or "CachedContent not found" in str(e):
                        print(f"[RECOVERY] Cache expired or invalid. Falling back to normal Pro model. Error: {e}")
                        self.reference_cache_name = None
                        if os.path.exists(".cache_info.json"):
                            try:
                                os.remove(".cache_info.json")
                            except:
                                pass
                        # Retry without cache
                        return self.solve_problem(problem_text, stream, image)
                    else:
                        raise e
            else:
                # No cache, use the Pro model
                if stream:
                    response = self.client.models.generate_content_stream(model=self.pro_model_name, contents=contents)
                else:
                    response = self.client.models.generate_content(model=self.pro_model_name, contents=contents)
            
            if stream:
                return response
            return response.text
        except Exception as e:
            return f"Error during solving: {e}"

    def detect_graph_coordinates(self, image: Image.Image, description: str):
        """
        Uses object detection/spatial understanding to find coordinates of a graph/image.
        Returns [ymin, xmin, ymax, xmax] in 0-1000 scale.
        """
        if not image:
            return None
            
        detect_prompt = (
            f"In this image, find the bounding box for the following element: {description}. "
            "Respond ONLY with the coordinates in this format: [ymin, xmin, ymax, xmax]. "
            "The values should be integers between 0 and 1000."
        )
        try:
            response = self.client.models.generate_content(
                model=self.lite_model_name,
                contents=[detect_prompt, image]
            )
            text = response.text.strip()
            # Extract numbers from [ymin, xmin, ymax, xmax]
            import re
            nums = re.findall(r'\d+', text)
            if len(nums) >= 4:
                return [int(n) for n in nums[:4]]
        except Exception as e:
            print(f"Detection error: {e}")
        return None




