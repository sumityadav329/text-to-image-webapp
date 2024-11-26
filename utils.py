import os
import requests
import time
from typing import Optional

def load_environment():
    """
    Attempt to load environment variables with error handling.
    
    Returns:
        Optional[str]: Hugging Face Token or None
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed. Ensure HF_TOKEN is set in environment.")
    
    return os.getenv("HF_TOKEN")

def query_hf_api(
    prompt: str, 
    model_url: str = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
    max_retries: int = 3
) -> Optional[bytes]:
    """
    Query the Hugging Face Inference API with robust error handling and retry mechanism.
    
    Args:
        prompt (str): Text prompt for image generation
        model_url (str): URL of the Hugging Face model
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        Optional[bytes]: Generated image bytes or None
    """
    # Validate inputs
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    # Load token
    HF_TOKEN = load_environment()
    if not HF_TOKEN:
        raise ValueError("Hugging Face token not found. Set HF_TOKEN in .env or environment variables.")
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Payload with additional configuration
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "low quality, bad anatomy, blurry",
            "num_inference_steps": 50,
        }
    }
    
    # Retry mechanism
    for attempt in range(max_retries):
        try:
            response = requests.post(
                model_url, 
                headers=headers, 
                json=payload,
                timeout=120  # 2-minute timeout
            )
            
            # Check for specific error conditions
            if response.status_code == 503:
                # Model might be loading, wait and retry
                print(f"Service unavailable, retrying in {5 * (attempt + 1)} seconds...")
                time.sleep(5 * (attempt + 1))
                continue
            
            response.raise_for_status()  # Raise exception for bad status codes
            
            return response.content
        
        except requests.exceptions.RequestException as e:
            print(f"Request error (Attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate image after {max_retries} attempts: {e}")
            
            # Wait before retrying
            time.sleep(5 * (attempt + 1))
    
    raise RuntimeError("Unexpected error in image generation")
