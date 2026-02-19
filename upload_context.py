
print("Starting upload_context.py...")
import os
import json
import time
import datetime
from google import genai
from google.genai import types

# Load API Key
print("Checking API Key...")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit(1)

print("Initializing GenAI Client...")
client = genai.Client(api_key=api_key)

# Debug: List available models
print("Available models:")
for m in client.models.list():
    print(f" - {m.name}")
    # print(f"   {m.supported_actions}") # If this exists



CACHE_FILE = ".cache_info.json"
REF_DIR = "reference_books"

def upload_and_cache():
    if not os.path.exists(REF_DIR):
        print(f"Directory {REF_DIR} not found.")
        return

    files_to_upload = []
    print(f"Scanning {REF_DIR}...")
    
    # 1. Upload Files
    for filename in os.listdir(REF_DIR):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(REF_DIR, filename)
            print(f"Uploading {filename}...")
            try:
                # Upload using the new SDK
                # Note: The new SDK manages uploads via client.files.upload
                with open(path, "rb") as f:
                    uploaded_file = client.files.upload(
                        file=f,
                        config={'display_name': filename, 'mime_type': 'application/pdf'}
                    )
                files_to_upload.append(uploaded_file)
                print(f"  - Uploaded: {uploaded_file.name}")
            except Exception as e:
                print(f"  - Failed to upload {filename}: {e}")

    if not files_to_upload:
        print("No PDF files found to cache.")
        return

    print("Waiting for files to be processed...")
    # 2. Wait for processing
    for f in files_to_upload:
        while True:
            # Refresh file state
            current_file = client.files.get(name=f.name)
            if current_file.state == "ACTIVE":
                print(f"  - {current_file.display_name}: Ready")
                break
            elif current_file.state == "FAILED":
                print(f"  - {current_file.display_name}: Failed processing")
                return
            else:
                print('.', end='', flush=True)
                time.sleep(2)

    print("\nCreating cache...")
    
    # 3. Create Cache
    # The new SDK uses client.caches.create
    try:
        # According to error, 'model' is required keyword-only argument for create()
        # It seems it's not enough to be in config, or the signature is create(model=..., config=...)
        
        cache_config = {
            'display_name': 'math_reference_books',
            'system_instruction': 'You are a helpful math reasoning assistant. Use the provided context to answer questions.',
            'contents': [
                types.Content(
                    role='user',
                    parts=[types.Part(file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type)) for f in files_to_upload]
                )
            ],
            'ttl': '7200s' # 2 hours
        }
        
        # Explicitly passing model
        # User requested Pro model for solving
        # Using gemini-1.5-pro-002 as the stable Pro model that supports context caching
        cache = client.caches.create(
            model='models/gemini-2.5-pro',
            config=cache_config
        )

        print(f"Cache created: {cache.name}")
        
        # Save cache info
        cache_info = {
            "cache_name": cache.name,
            "model_name": cache.model,
            "expire_time": cache.expire_time, # Check format
            "files": [f.uri for f in files_to_upload]
        }
        
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_info, f, indent=2, default=str)
        
        print(f"Cache info saved to {CACHE_FILE}")
        
    except Exception as e:
        print(f"Failed to create cache: {e}")

if __name__ == "__main__":
    upload_and_cache()
