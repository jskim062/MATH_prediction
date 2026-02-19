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
            "너는 전문 수학 문제 추출기야. 이미지에서 각 수학 문제를 추출해서 다음 XML 형식으로 응답해 줘.\n\n"
            "**작성 규칙:**\n"
            "1. 모든 수식은 LaTeX 형식으로 변환할 것.\n"
            "2. JSON이 아니므로 백슬래시(\\)나 따옴표를 이스케이프할 필요 없음. 있는 그대로 작성할 것.\n"
            "3. 각 문제는 `<problem>` 태그로 감싸고, 그 안에 세부 태그를 작성할 것.\n"
            "4. 문제의 **전문(지문과 수식 포함)**을 한 글자도 빠짐없이 `<content>` 태그에 LaTeX 형식으로 작성할 것.\n"
            "   - 모든 수식과 기호는 $ (inline) 또는 $$ (block)로 감싸서 LaTeX로 변환할 것.\n\n"
            "**XML 구조 예시:**\n"
            "<root>\n"
            "  <problem>\n"
            "    <number>1</number>\n"
            "    <content>여기에 지문과 $수식$을 섞어서 문제 전문을 작성...</content>\n"
            "    <options>\n"
            "      <option>보기1</option>\n"
            "      <option>보기2</option>\n"
            "    </options>\n"
            "    <score>10</score>\n"
            "    <image_description>그래프 설명</image_description>\n"
            "  </problem>\n"
            "</root>\n\n"
            "위 형식을 정확히 지켜서 응답해 줘."
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

    def solve_problem(self, problem_text: str, stream: bool = False, image: Image.Image = None):
        """
        [Stage 2: Pro] Generates a concise but clear solution using reference context if available.
        If image is provided, it is included in the prompt for visual verification (Re-scan).
        """
        solve_prompt = (
            f"다음 수학 문제에 대한 상세 풀이 과정을 작성해 줘. \n\n{problem_text}\n\n"
            "지침:\n"
            "1. 수식은 LaTeX($$)를 활용할 것.\n"
            "2. 핵심 흐름 위주로 간결하게 설명할 것.\n"
            "3. 마지막에 반드시 다음 형식으로 난이도 분석을 XML 블록으로 추가할 것:\n"
            "<analysis>\n"
            "  <level>난이도 레벨 (1~6 정수)</level>\n"
            "  <conceptual_reason>개념적 복잡도 설명</conceptual_reason>\n"
            "  <logical_reason>논리적 추론 단계 설명</logical_reason>\n"
            "  <computational_reason>계산 복잡도 설명</computational_reason>\n"
            "  <summary>종합적인 난이도 평 및 조언</summary>\n"
            "</analysis>\n"
            "4. 최종 정답(숫자 또는 최종 표현식)은 반드시 `<final_answer>값</final_answer>` 태그로 감싸서 풀이 과정 끝에 포함할 것.\n"
        )
        
        if image:
             solve_prompt = (
                 "**[Visual Re-scan Requested]**\n"
                 "The text extraction might be incorrect. Please use the provided image to verify the problem text and numbers, "
                 "then solve the problem correctly based on the image content.\n\n"
                 + solve_prompt
             )
        
        contents = [solve_prompt]
        if image:
            contents.append(image)

        try:
            if self.reference_cache_name:
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




