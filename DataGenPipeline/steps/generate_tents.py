import os
import io
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from steps.utils import image_step
from dotenv import load_dotenv
import random

load_dotenv()

client = genai.Client(api_key=os.getenv("API_KEY"))
colors = ["red", "green", "blue", "tan", "orange", "grey", "white"]
positions = ["top left", "top right", "center", "bottom left", "bottom right"]


@image_step
def run(image):
    color = random.choice(colors)
    position = random.choice(positions)
    # prompt = f"Add a realistic {color}, open, pop-up, camping tent to this image. Keep the lighting and style consistent."
    prompt = f"Here is a top down, nadir view of a landscape from roughly 100 feet away. Add a very very small, realistic, {color}, open, pop-up, camping tent in the {position} of this image. Keep the lighting and style consistent."
    try:
        print(f"gen tent {color}, {position}")
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[
                prompt,
                image
            ],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(
                    image_size="4K"
                )
            )
        )

        new_image = image
        for part in response.parts:
            new_image = part.as_image()
        return new_image

    except Exception as e:
        print(f"API Call Failed: {e}")
        return image
