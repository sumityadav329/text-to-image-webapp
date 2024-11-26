import os
import gradio as gr
from PIL import Image
import io
import requests
from typing import Optional, Tuple

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

def craft_realistic_prompt(base_prompt: str) -> str:
    """
    Enhance prompts for more photorealistic results
    
    Args:
        base_prompt (str): Original user prompt
    
    Returns:
        str: Enhanced, detailed prompt
    """
    realistic_modifiers = [
        "photorealistic",
        "high resolution",
        "sharp focus",
        "professional photography",
        "natural lighting",
        "detailed textures"
    ]
    
    # Combine base prompt with realistic modifiers
    enhanced_prompt = f"{' '.join(realistic_modifiers)}, {base_prompt}, shot on professional camera, 8k resolution"
    
    return enhanced_prompt

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
    
    # Payload with enhanced configuration for realism
    payload = {
        "inputs": craft_realistic_prompt(prompt),
        "parameters": {
            "negative_prompt": "cartoon, anime, low quality, bad anatomy, blurry, unrealistic, painting, drawing, sketch",
            "num_inference_steps": 75,  # Increased steps
            "guidance_scale": 8.5,      # Higher guidance
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
            
            response.raise_for_status()  # Raise exception for bad status codes
            
            return response.content
        
        except requests.exceptions.RequestException as e:
            print(f"Request error (Attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate image after {max_retries} attempts: {e}")

    raise RuntimeError("Unexpected error in image generation")

def generate_image(prompt: str) -> Tuple[Optional[Image.Image], str]:
    """
    Generate an image from a text prompt.
    
    Args:
        prompt (str): Text description for image generation
    
    Returns:
        Tuple[Optional[Image.Image], str]: 
        Generated PIL Image and status message
    """
    try:
        # Validate prompt
        if not prompt or not prompt.strip():
            return None, "Error: Prompt cannot be empty"
        
        # Generate image bytes
        image_bytes = query_hf_api(prompt)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        return image, "Image generated successfully!"
    
    except Exception as e:
        print(f"Image generation error: {e}")
        return None, f"Error: {str(e)}"

def create_gradio_interface():
    """
    Create and configure the Gradio interface.
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        title="🎨 AI Image Generator"
    ) as demo:
        # Title and Description
        gr.Markdown("# 🎨 AI Image Generator ")
        gr.Markdown("This webapp is created with ❤ by Sumit Yadav as GenAI project. Credits to HuggingFace Spaces and StabilityAI")
        gr.Markdown("Generate stunning images from your text prompts using AI!")
        
        # Input and Output Components
        with gr.Row():
            with gr.Column(scale=3):
                # Prompt Input
                text_input = gr.Textbox(
                    label="Enter your image prompt", 
                    placeholder="e.g., 'Photorealistic portrait of a woman in natural light'",
                    lines=3
                )
                
                # Generate Button
                generate_button = gr.Button("✨ Generate Image", variant="primary")
            
            # Output Image Display
            with gr.Column(scale=4):
                output_image = gr.Image(
                    label="Generated Image", 
                    type="pil", 
                    interactive=False
                )
        
        # Status Output
        status_output = gr.Textbox(label="Status")
        
        # Event Handlers
        generate_button.click(
            fn=generate_image,
            inputs=[text_input],
            outputs=[output_image, status_output]
        )
    
    return demo

def main():
    """
    Main entry point for the Gradio application.
    """
    try:
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",  # Listen on all network interfaces
            server_port=7860,  # Default Gradio port
            share=False  # Set to True for a public link
        )
    except Exception as e:
        print(f"Error launching Gradio app: {e}")

if __name__ == "__main__":
    main()