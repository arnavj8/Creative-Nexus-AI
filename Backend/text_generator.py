from google import genai
from dotenv import load_dotenv
import os
import re
import json

from Backend.logger import logging
from Backend.db_utils import ensure_api_keys


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# API KEY SETUP
# =========================================================

try:
    # Ensure API keys are available
    ensure_api_keys()

    # Prefer GEMINI_API_KEY, but keep GEN_API_KEY
    # for backward compatibility.
    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )

    if not GEMINI_API_KEY:
        logging.error(
            "Gemini API key not found in environment variables."
        )
    else:
        logging.info(
            "Gemini API key loaded successfully."
        )

except Exception as e:
    logging.error(
        f"Error while loading API keys: {str(e)}"
    )

    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():
    """
    Create and return a Gemini API client.
    """

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
        or GEMINI_API_KEY
    )

    if not api_key:
        raise ValueError(
            "Gemini API key is not configured."
        )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# GENERATE VIDEO SCRIPT
# =========================================================

def generate_video_script(topic, style):

    logging.info(
        f"Generating video script for topic: "
        f"'{topic}' with style: '{style}'"
    )

    prompt = f"""
You are a professional AI video script generator.
Given a video topic and style, generate a detailed
script and structured breakdown for an AI-generated video.

## Instructions:

1. Generate a structured script with clear narration
   for voiceover.

2. Break the script into multiple scenes
   (each 10 seconds long).

3. For each scene, provide:

   - Timestamp: (Start - End)
   - Voiceover: (The spoken dialogue for this scene)
   - Scene Description: (What should be visually shown)
   - Character & Object Details:
     (To ensure consistency in generated images)
   - Shot Type & Camera Angle:
     (e.g., Wide shot, Close-up, Aerial view)
   - Mood & Emotion:
     (To determine music and voiceover tone)
   - Suggested Transition Effect:
     (choose from this list only:
     quick cuts, fade-in, zoom out and crossfade
     and use a single effect at a time)

4. Provide a final summary of the entire video mood
   (e.g., inspirational, dramatic, educational).

5. Generate a background music prompt based on
   the overall video theme and mood.

6. Ensure clarity, consistency, and coherence
   in the storytelling.

## Example Input:

Topic: "{topic}"
Style: "{style}"

## Expected Output:

Return ONLY valid JSON in this format:

{{
  "video_title": "{topic}",
  "scenes": [
    {{
      "timestamp": "00:00 - 00:10",
      "voiceover": "Example narration for first scene.",
      "scene_description":
        "Description of the first scene.",
      "character_object_details":
        "Characters, objects, and their details.",
      "shot_type_camera_angle":
        "Wide shot, aerial view, etc.",
      "mood_emotion":
        "Mood setting for this scene.",
      "suggested_transition_effect":
        "Fade-in"
    }}
  ],
  "overall_video_mood":
    "Final video mood description.",
  "background_music_prompt":
    "A cinematic orchestral soundtrack with a dramatic "
    "build-up, matching the video's emotional tone."
}}

Important:
- Return valid JSON only.
- Do not add Markdown code fences.
- Do not add explanations before or after the JSON.
"""

    try:

        logging.info(
            "Sending prompt to Gemini model..."
        )

        # Create Gemini client
        client = get_gemini_client()

        # Generate response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if not response or not response.text:
            logging.error(
                "Gemini returned an empty response."
            )
            return None

        logging.info(
            "Successfully received response "
            "from Gemini model."
        )

        return response.text

    except Exception as e:

        logging.error(
            f"Error during video script generation: {e}"
        )

        return None


# =========================================================
# EXTRACT JSON
# =========================================================

def extract_json(text):
    """
    Extract valid JSON from model output.
    Supports both plain JSON and Markdown
    code-block responses.
    """

    try:

        logging.info(
            "Extracting JSON from AI response..."
        )

        if not text:
            logging.warning(
                "Empty AI response received."
            )

            return {
                "error": "Empty AI response"
            }

        text = text.strip()

        # -------------------------------------------------
        # Remove Markdown JSON code fences
        # -------------------------------------------------

        if text.startswith("```json"):
            text = text[7:]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # -------------------------------------------------
        # Try parsing the entire response
        # -------------------------------------------------

        try:

            data = json.loads(text)

            logging.info(
                "JSON extracted successfully."
            )

            return data

        except json.JSONDecodeError:
            pass

        # -------------------------------------------------
        # Find JSON object inside response
        # -------------------------------------------------

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:

            json_str = match.group(0)

            data = json.loads(json_str)

            logging.info(
                "JSON extracted successfully "
                "from model response."
            )

            return data

        logging.warning(
            "No valid JSON found in AI response."
        )

        return {
            "error": "No valid JSON found"
        }

    except json.JSONDecodeError as e:

        logging.error(
            f"Invalid JSON format: {e}"
        )

        return {
            "error": f"Invalid JSON format: {e}"
        }

    except Exception as e:

        logging.error(
            f"Error extracting JSON: {e}"
        )

        return {
            "error": f"JSON extraction error: {e}"
        }


# =========================================================
# SAVE JSON
# =========================================================

def save_json(video_data, output_dir):
    """
    Save the video data to a JSON file
    in the given directory.
    """

    try:

        # Create directory if it doesn't exist
        os.makedirs(
            output_dir,
            exist_ok=True
        )

        json_file_path = os.path.join(
            output_dir,
            "video_script.json"
        )

        logging.info(
            f"Saving video script to "
            f"{json_file_path}"
        )

        with open(
            json_file_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                video_data,
                file,
                indent=4,
                ensure_ascii=False
            )

        logging.info(
            f"Video script saved successfully "
            f"to {json_file_path}"
        )

        return json_file_path

    except Exception as e:

        logging.error(
            f"Error saving video script to file: {e}"
        )

        return None