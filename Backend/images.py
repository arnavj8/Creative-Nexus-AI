import os
import requests
from PIL import Image
from io import BytesIO
import base64

from dotenv import load_dotenv
from google import genai

from Backend.logger import logging
from Backend.db_utils import ensure_api_keys


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# API KEYS
# =========================================================

try:
    ensure_api_keys()

    # Preferred Gemini variable
    # GEN_API_KEY remains as a backward-compatible fallback.
    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )

    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

    logging.info(
        "Image generator API configuration - "
        f"Gemini: {'Set' if GEMINI_API_KEY else 'Not set'}, "
        f"Hugging Face: {'Set' if HF_API_TOKEN else 'Not set'}"
    )

except Exception as e:
    logging.error(
        f"Error ensuring API keys: {str(e)}"
    )

    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )

    HF_API_TOKEN = os.getenv(
        "HF_API_TOKEN"
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():
    """
    Create and return a Gemini client.
    """
    ensure_api_keys()

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
        or GEMINI_API_KEY
    )

    if not api_key:
        raise ValueError(
            "Gemini API key is not configured. "
            "Set GEMINI_API_KEY in the environment or save it on the home page."
        )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# GENERATE IMAGE PROMPT WITH GEMINI
# =========================================================

def generate_prompt_with_gemini(
    topic: str,
    style: str
) -> str:
    """
    Use Gemini API to generate a rich,
    creative image prompt.
    """

    prompt_input = (
        "Generate a highly descriptive, visually rich prompt "
        "for an AI image generation model based on the topic "
        f"'{topic}' in the '{style}' style. "
        "Avoid camera settings, but include artistic elements."
    )

    try:

        logging.info(
            f"Generating prompt with Gemini for topic: "
            f"'{topic}' and style: '{style}'"
        )

        # Create Gemini client
        client = get_gemini_client()

        # Generate content
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_input
        )

        if not response or not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        prompt_text = response.text.strip()

        logging.info(
            f"Successfully generated prompt for topic: "
            f"'{topic}'"
        )

        return prompt_text

    except Exception as e:

        logging.error(
            f"Gemini prompt generation failed for topic "
            f"'{topic}' with error: {e}"
        )

        raise RuntimeError(
            f"Gemini prompt generation failed: {e}"
        )


# =========================================================
# GENERATE IMAGE FROM TOPIC AND STYLE
# =========================================================

def generate_image_from_topic_and_style(
    topic: str,
    style: str
):
    """
    Generate image using Hugging Face Stable Diffusion
    based on a Gemini-enhanced prompt.

    Returns:
        Tuple[str, str]:
        (description, base64 image string)
    """

    try:

        logging.info(
            f"Starting image generation for topic: "
            f"'{topic}' and style: '{style}'"
        )

        # -------------------------------------------------
        # Validate Hugging Face token
        # -------------------------------------------------

        if not HF_API_TOKEN:

            raise ValueError(
                "HF_API_TOKEN is not configured."
            )

        # -------------------------------------------------
        # Step 1: Generate enhanced prompt
        # -------------------------------------------------

        enhanced_prompt = (
            generate_prompt_with_gemini(
                topic,
                style
            )
        )

        logging.info(
            f"Enhanced prompt generated for topic: "
            f"'{topic}'"
        )

        # -------------------------------------------------
        # Step 2: Hugging Face API
        # -------------------------------------------------

        api_url = (
            "https://api-inference.huggingface.co/"
            "models/stabilityai/"
            "stable-diffusion-xl-base-1.0"
        )

        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}"
        }

        payload = {
            "inputs": enhanced_prompt
        }

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        logging.info(
            f"Image successfully generated from "
            f"Hugging Face for topic: '{topic}'"
        )

        # -------------------------------------------------
        # Step 3: Process image
        # -------------------------------------------------

        image = Image.open(
            BytesIO(response.content)
        )

        buffer = BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        image_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        description = (
            f"Generated image for topic '{topic}' "
            f"in '{style}' style using enhanced prompt."
        )

        logging.info(
            f"Image processing completed for topic: "
            f"'{topic}'"
        )

        return description, image_base64

    except Exception as e:

        logging.error(
            f"Image generation failed for topic "
            f"'{topic}' with error: {e}"
        )

        raise RuntimeError(
            f"Image generation failed: {e}"
        )