---
title: Img Genwebapp
emoji: 🖼
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference




# 🎨 AI Image Generator Web App

Welcome to the **AI Image Generator** web application! This app allows you to generate stunning photorealistic images from text prompts using cutting-edge AI technology.

### 🌐 Live Demo
Check out the live version of the app hosted on Hugging Face Spaces:  
[AI Image Generator](https://huggingface.co/spaces/sumityadav329/text-to-image-webapp)

---

## 🚀 Features

- **Text-to-Image Generation**: Generate high-quality, photorealistic images from detailed prompts.
- **Enhanced Realism**: Automatically enriches prompts for better results.
- **Error Handling**: Robust handling for empty prompts or API issues.
- **Streamlined UI**: Simple and elegant interface powered by Gradio.
- **Status Feedback**: Displays generation status to keep you informed.

---

## 🛠️ How It Works

1. **Input Prompt**: Enter a detailed description of the image you want to generate. For example:
   - *"Photorealistic portrait of a woman in natural light"*
   - *"Astronaut riding a bike on Mars during sunset"*
   
2. **Generate Image**: Click on the **✨ Generate Image** button.

3. **View Result**: The generated image will appear alongside a status message.

---

## 💻 Technologies Used

- **Gradio**: For building the interactive web application.
- **Hugging Face Inference API**: For accessing the Stable Diffusion model.
- **PIL (Python Imaging Library)**: For handling and displaying images.

---

## 🔗 Repository

This project is version-controlled and hosted on GitHub. It is also integrated with Hugging Face Spaces for seamless updates.  
You can clone the repository to explore or contribute to the codebase:

```bash
git clone https://github.com/sumityadav329/text-to-image-webapp.git
```

---

## 📜 Requirements

- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Hugging Face Token: Set your **HF_TOKEN** in a `.env` file.

---

## ⚡ Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/sumityadav329/text-to-image-webapp.git
   cd text-to-image-webapp
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:
   ```bash
   gradio app.py
   ```

4. Access the app at [http://localhost:7860](http://localhost:7860).

---

## ❤️ Credits

- **Developer**: Sumit Yadav  
- **Hosting**: Hugging Face Spaces  
- **Model**: StabilityAI's Stable Diffusion XL  

Feel free to fork, modify, or contribute to this project! 😊





