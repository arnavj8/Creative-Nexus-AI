import os
import requests

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

from google import genai

from Backend.logger import logging
from Backend.db_utils import ensure_api_keys


# Load .env locally.
# On Render, environment variables come from Render settings.
load_dotenv()


# ---------------------------------------------------------
# API KEYS
# ---------------------------------------------------------

try:
    ensure_api_keys()
except Exception as e:
    logging.error(
        f"Error ensuring API keys: {e}"
    )


GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GEN_API_KEY")
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")


logging.info(
    "API keys loaded - "
    f"HF: {'Set' if HF_API_TOKEN else 'Not set'}, "
    f"Gemini: {'Set' if GEMINI_API_KEY else 'Not set'}"
)


# ---------------------------------------------------------
# GEMINI CLIENT
# ---------------------------------------------------------

def get_gemini_client():
    """
    Create a Gemini client using the configured API key.
    """

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )

    if not api_key:
        raise ValueError(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY in the environment."
        )

    return genai.Client(api_key=api_key)


# ---------------------------------------------------------
# GENERATE MEME CONTENT
# ---------------------------------------------------------

def generate_meme_content(
    prompt,
    emotion,
    language="english"
):
    """
    Generate:
    - meme caption
    - image description
    - text color
    - text position
    """

    logging.info(
        "Generating meme content using Gemini API."
    )

    try:
        client = get_gemini_client()

        lang_instruction = (
            "in English"
            if language.lower() == "english"
            else "in Hindi"
        )

        gemini_prompt = (
            f"Generate a short, funny meme caption under 10 words "
            f"{lang_instruction} and an image description for: {prompt}. "
            f"The image should be meme-style, expressive, and based "
            f"on the emotion: {emotion}. "
            f"Also suggest a meme-matching text color "
            f"(red, green, yellow, white, black, etc.) "
            f"and whether the text should be placed at "
            f"'top' or 'bottom'. "
            f"Return ONLY this format:\n"
            f"Caption: <caption> | "
            f"Image: <image description> | "
            f"Color: <color> | "
            f"Position: <top/bottom>"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_prompt
        )

        response_text = (
            response.text.strip()
            if response and response.text
            else ""
        )

        logging.info(
            "Gemini response received."
        )

        if "| Image:" in response_text:
            parts = response_text.split("|")

            if len(parts) >= 4:
                caption = (
                    parts[0]
                    .replace("Caption:", "")
                    .strip()
                )

                image_description = (
                    parts[1]
                    .replace("Image:", "")
                    .strip()
                )

                text_color = (
                    parts[2]
                    .replace("Color:", "")
                    .strip()
                    .lower()
                )

                text_position = (
                    parts[3]
                    .replace("Position:", "")
                    .strip()
                    .lower()
                )

                # Safety checks for Gemini output
                allowed_colors = {
                    "red",
                    "green",
                    "yellow",
                    "white",
                    "black",
                    "blue",
                    "orange",
                    "purple"
                }

                if text_color not in allowed_colors:
                    text_color = "white"

                if text_position not in {
                    "top",
                    "bottom"
                }:
                    text_position = "top"

                logging.info(
                    "Meme content generated successfully."
                )

                return (
                    caption,
                    image_description,
                    text_color,
                    text_position,
                    language
                )

        logging.warning(
            "Gemini response format was invalid."
        )

    except Exception as e:
        logging.error(
            f"Gemini meme generation failed: {e}"
        )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    logging.warning(
        "Using default meme content."
    )

    if language.lower() == "english":
        caption = "AI is too funny!"
    else:
        caption = "एआई बहुत मज़ेदार है!"

    image_description = (
        f"A funny, exaggerated meme-style image "
        f"about {prompt}. "
        f"Cartoonish, expressive and meme-worthy."
    )

    text_color = "white"
    text_position = "top"

    return (
        caption,
        image_description,
        text_color,
        text_position,
        language
    )


# ---------------------------------------------------------
# GENERATE MEME IMAGE
# ---------------------------------------------------------

def generate_meme_image(image_prompt):
    """
    Generate a meme image using Hugging Face.
    """

    if not HF_API_TOKEN:
        raise ValueError(
            "HF_API_TOKEN is not configured."
        )

    logging.info(
        "Sending image prompt to Hugging Face API."
    )

    payload = {
        "inputs": image_prompt
    }

    api_url = (
        "https://api-inference.huggingface.co/models/"
        "stabilityai/stable-diffusion-xl-base-1.0"
    )

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            logging.info(
                "Image generated successfully."
            )

            return Image.open(
                BytesIO(response.content)
            )

        logging.error(
            f"Hugging Face API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

        raise Exception(
            f"Error generating image: {response.text}"
        )

    except requests.RequestException as e:
        logging.error(
            f"Hugging Face request failed: {e}"
        )

        raise


# ---------------------------------------------------------
# ADD MEME TEXT
# ---------------------------------------------------------

def add_meme_text(
    image,
    text,
    text_color,
    position,
    language
):
    """
    Overlay meme text dynamically.
    """

    logging.info(
        "Adding text to meme image."
    )

    draw = ImageDraw.Draw(image)

    # Project root:
    # content-generator/
    #
    # Font location:
    # Backend/static/arial.ttf
    # Backend/static/arial_hindi.ttf

    backend_static = (
        Path(__file__).resolve().parent / "static"
    )

    font_paths = {
        "english": backend_static / "arial.ttf",
        "hindi": backend_static / "arial_hindi.ttf"
    }

    selected_font = font_paths.get(
        language.lower(),
        font_paths["english"]
    )

    try:
        font_size = max(
            20,
            int(image.height * 0.07)
        )

        font = ImageFont.truetype(
            str(selected_font),
            size=font_size
        )

    except Exception as e:
        logging.warning(
            f"Failed to load custom font: {e}"
        )

        font = ImageFont.load_default()

    width, height = image.size

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) / 2

    if position.lower() == "bottom":
        y = height - text_height - 30
    else:
        y = 10

    # Text outline
    for offset in [
        (-3, -3),
        (3, -3),
        (-3, 3),
        (3, 3)
    ]:
        draw.text(
            (
                x + offset[0],
                y + offset[1]
            ),
            text,
            font=font,
            fill="black"
        )

    draw.text(
        (x, y),
        text,
        fill=text_color,
        font=font,
        stroke_width=3,
        stroke_fill="black"
    )

    logging.info(
        "Text added to meme successfully."
    )

    return image