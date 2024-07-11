import os
from typing import Union

import torch
import uvicorn
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

load_dotenv()
app = FastAPI(title="fastapi-classification-demo")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
templates = Jinja2Templates(directory="templates")


def gen_caption(image: Union[Image.Image, str]) -> str:
    """Generate a caption for an image.

    Args:
        image (Union[Image.Image, str]): Image to generate a caption for.

    Returns:
        str: Generated caption.
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Get the model and processor
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    inputs = feature_extractor(images=image, return_tensors="pt").pixel_values
    inputs = inputs.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_ids = model.generate(inputs, **gen_kwargs)

    preds = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return preds.strip()


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "name": "Class of Cinnamon AI Bootcamp 2023"}
    )


@app.post("/predict/")
async def predict(file: UploadFile):
    """Predict caption for an uploaded image.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        dict: Generated caption.
    """
    file_obj = file.file
    image = Image.open(file_obj)
    caption = gen_caption(image)
    return {"caption": caption}


def main():
    # Run web server with uvicorn
    uvicorn.run(
        "fastapi_backend:app",
        host=os.getenv("FASTAPI_HOST", "127.0.0.1"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        reload=True,  # Uncomment this for debug
    )


if __name__ == "__main__":
    main()
