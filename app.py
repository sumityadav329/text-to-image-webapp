import os
import gradio as gr
from PIL import Image
import io
from utils import query_hf_api

def generate_image(prompt: str) -> Image.Image:
    """
    Generate an image from a text prompt.
    
    Args:
        prompt (str): Text description for image generation
    
    Returns:
        Image.Image: Generated PIL Image
    """
    try:
        # Generate image bytes
        image_bytes = query_hf_api(prompt)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        return image
    
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

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
        gr.Markdown("# 🎨 AI Image Generator")
        gr.Markdown("Generate stunning images from your text prompts using AI!")
        
        # Input and Output Components
        with gr.Row():
            with gr.Column(scale=3):
                # Prompt Input
                text_input = gr.Textbox(
                    label="Enter your image prompt", 
                    placeholder="e.g., 'Astronaut riding a bike on Mars at sunset'",
                    lines=3
                )
                
                # Advanced Options
                with gr.Accordion("Advanced Options", open=False):
                    steps_slider = gr.Slider(
                        minimum=10, 
                        maximum=100, 
                        value=50, 
                        step=1, 
                        label="Inference Steps"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1, 
                        maximum=20, 
                        value=7.5, 
                        step=0.5, 
                        label="Guidance Scale"
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
        
        # Error Handling Output
        error_output = gr.Textbox(label="Status", visible=False)
        
        # Event Handlers
        generate_button.click(
            fn=generate_image,
            inputs=[text_input],
            outputs=[output_image, error_output],
            api_name="generate"
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
            share=True  # Set to True if you want a public link
        )
    except Exception as e:
        print(f"Error launching Gradio app: {e}")

if __name__ == "__main__":
    main()