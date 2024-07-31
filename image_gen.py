import streamlit as st
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image

# Load the model
pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
)

# Streamlit app
st.title("Text-to-Image Generation with Stable Diffusion")
st.write("Enter a text prompt and generate an image.")

# Input text box
prompt = st.text_input("Enter your prompt:")

# Button to generate the image
if st.button("Generate"):
    with st.spinner("Generating image..."):
        generator = torch.Generator("cpu").manual_seed(31)
        image = pipeline(prompt, generator=generator).images[0]
        
        # Convert the image to a format that can be displayed by Streamlit
        image_pil = Image.fromarray(image.cpu().numpy())
        st.image(image_pil, caption="Generated Image", use_column_width=True)
