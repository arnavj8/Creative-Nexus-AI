import os
import re
import json
import arxiv
import requests
import markdown
from PIL import Image
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv

import wikipedia as wiki_wiki
import wikipedia.exceptions as wiki_exceptions
from newsapi import NewsApiClient
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
    # Ensure API keys are available
    ensure_api_keys()

    # Gemini API key
    # GEMINI_API_KEY is preferred.
    # GEN_API_KEY is kept as backward compatibility.
    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )

    # Hugging Face API token
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

    # News API key
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")

    logging.info(
        "API keys loaded - "
        f"Gemini: {'Set' if GEMINI_API_KEY else 'Not set'}, "
        f"HF: {'Set' if HF_API_TOKEN else 'Not set'}, "
        f"NewsAPI: {'Set' if NEWS_API_KEY else 'Not set'}"
    )

except Exception as e:
    logging.error(
        f"Error loading API keys: {str(e)}"
    )

    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )

    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():
    """
    Create and return a Gemini client.

    Uses GEMINI_API_KEY as the primary environment variable
    and GEN_API_KEY as a backward-compatible fallback.
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
            "Set GEMINI_API_KEY in the environment or save it via home page."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# WIKIPEDIA
# =========================================================

def fetch_wikipedia(topic: str) -> str:
    """Fetch summary content from Wikipedia."""

    try:
        page = wiki_wiki.page(topic)

        logging.info(
            f"Fetching Wikipedia page for topic: {topic}"
        )

        return (
            page.summary
            if page.content
            else "No content found for this topic."
        )

    except wiki_exceptions.DisambiguationError as e:
        logging.error(
            f"Disambiguation error for topic "
            f"'{topic}': {e.options}"
        )

        return (
            "Disambiguation error: The topic is ambiguous. "
            f"Suggestions: {e.options}"
        )

    except wiki_exceptions.HTTPTimeoutError as e:
        logging.error(
            f"HTTP timeout error while fetching "
            f"topic '{topic}': {e}"
        )

        return f"HTTP timeout error: {e}"

    except wiki_exceptions.RedirectError as e:
        logging.error(
            f"Redirect error for topic '{topic}': "
            f"{e.args[0]}"
        )

        return (
            f"Redirect error: The page redirects to "
            f"{e.args[0]}"
        )

    except wiki_exceptions.PageError as e:
        logging.error(
            f"Page error for topic '{topic}': {e}"
        )

        return (
            f"Page error: The page '{topic}' "
            f"doesn't exist."
        )

    except Exception as e:
        logging.error(
            f"Unexpected error while fetching "
            f"Wikipedia content for '{topic}': {e}"
        )

        return (
            f"Error fetching Wikipedia content: {e}"
        )


# =========================================================
# ARXIV
# =========================================================

def fetch_arxiv(query):
    try:
        client = arxiv.Client()

        search = arxiv.Search(
            query=query,
            max_results=1,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = [
            result.summary
            for result in client.results(search)
        ]

        logging.info(
            f"Fetched Arxiv results for query: {query}"
        )

        return (
            results[0]
            if results
            else "No Arxiv content found."
        )

    except Exception as e:
        logging.error(
            f"Error fetching Arxiv content "
            f"for query '{query}': {str(e)}"
        )

        return f"Arxiv fetch error: {str(e)}"


# =========================================================
# DUCKDUCKGO
# =========================================================

def duckduckgo_instant_answer(query: str):
    """Fetch relevant content from DuckDuckGo Instant Answer API."""

    try:
        url = (
            "https://api.duckduckgo.com/"
            f"?q={requests.utils.quote(query)}"
            "&format=json"
        )

        response = requests.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if (
            "AbstractText" in data
            and data["AbstractText"]
        ):
            logging.info(
                f"Fetched DuckDuckGo result "
                f"for query: {query}"
            )

            return data["AbstractText"]

        elif (
            "RelatedTopics" in data
            and len(data["RelatedTopics"]) > 0
            and isinstance(data["RelatedTopics"][0], dict)
            and "Text" in data["RelatedTopics"][0]
        ):
            logging.info(
                f"Fetched related DuckDuckGo result "
                f"for query: {query}"
            )

            return data["RelatedTopics"][0]["Text"]

        else:
            logging.warning(
                f"No relevant content found for "
                f"DuckDuckGo query: {query}"
            )

            return "No relevant content found."

    except Exception as e:
        logging.error(
            f"Error fetching DuckDuckGo content "
            f"for query '{query}': {e}"
        )

        return (
            f"Error fetching DuckDuckGo content: {e}"
        )


# =========================================================
# NEWS API
# =========================================================

def fetch_news_newsapi(query):
    try:
        if not NEWS_API_KEY:
            return "NewsAPI key is not configured."

        newsapi = NewsApiClient(
            api_key=NEWS_API_KEY
        )

        articles = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt"
        )

        logging.info(
            f"Fetched news articles for query: {query}"
        )

        return (
            articles["articles"][0]["description"]
            if articles["articles"]
            else "No news articles found."
        )

    except Exception as e:
        logging.error(
            f"Error fetching NewsAPI content "
            f"for query '{query}': {str(e)}"
        )

        return (
            f"News fetch error (NewsAPI): {str(e)}"
        )


# =========================================================
# GENERATE BLOG USING GEMINI
# =========================================================

def generate_blog_with_gemini(
    context,
    topic,
    style,
    length
):
    """
    Generate a structured blog using Gemini.
    """

    try:
        client = get_gemini_client()

        logging.info(
            f"Generating blog with Gemini for "
            f"topic: {topic}, style: {style}, "
            f"length: {length}"
        )

        # -------------------------------------------------
        # JSON STRUCTURE
        # -------------------------------------------------

        json_structure = json.dumps(
            {
                "title": "Blog Title",

                "tags": [
                    "Keyword1",
                    "Keyword2",
                    "Keyword3",
                    "Keyword4",
                    "Keyword5"
                ],

                "content": (
                    "Full blog content with proper formatting, "
                    "headings, and paragraphs. "
                    "Include max 2 image links using the "
                    "markdown format and name should be same "
                    "as given below:\n\n"
                    "![Description](blogs/image1.png)\n\n"
                    "![Another relevant description]"
                    "(blogs/image2.png)\n\n"
                    "This ensures images are embedded within "
                    "the blog."
                ),

                "image_prompts": [
                    (
                        "A detailed description of an image "
                        "related to the blog topic."
                    ),
                    (
                        "Another descriptive prompt for an "
                        "image relevant to the content."
                    ),
                    (
                        "A third image prompt that complements "
                        "the blog visually."
                    )
                ]
            },
            indent=2
        )

        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------

        prompt = f"""
Write a well-structured, engaging, and professional blog
on the topic: "{topic}" with style "{style}" and length
"{length}".

Use the provided context to ensure factual accuracy.
If relevant context is missing, generate a factually
correct and well-researched blog based on your knowledge.

The blog must be suitable for publishing and provide
valuable insights to the reader.

### Requirements:

- Maintain a natural, professional, and engaging tone.
- Structure the content with clear headings,
  subheadings, and well-formatted paragraphs.
- Ensure the blog is informative, easy to read,
  and logically structured.
- Avoid unnecessary repetition.
- Ensure a smooth flow of ideas.
- Include relevant images within the content using
  Markdown format:
  ![Alt Text](image_url)

### Provided Context:

{context}

### Recheck Before Returning the Answer:

- Verify the title is relevant, engaging, and reflects
  the blog content.
- Ensure the tags are diverse and relevant.
- Return exactly five tags.
- Ensure the content is well-structured and logically
  organized.
- Avoid factual errors.
- Include image references where appropriate.
- Provide image prompts that complement the blog.
- Return ONLY valid JSON.
- Do not include Markdown code fences.
- Do not include additional explanations.

### Output Format:

Return the blog in exactly this JSON structure:

{json_structure}
"""

        # -------------------------------------------------
        # GEMINI REQUEST
        # -------------------------------------------------

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        if response and response.text:
            logging.info(
                f"Blog generated successfully "
                f"for topic: {topic}"
            )

            return response.text

        logging.error(
            f"Empty response from Gemini model "
            f"for topic: {topic}"
        )

        return {
            "error": "Failed to generate blog content from Gemini."
        }

    except Exception as e:
        logging.error(
            f"Gemini blog generation error: {e}"
        )

        return {
            "error": f"Gemini error: {str(e)}"
        }


# =========================================================
# GENERATE IMAGE FROM PROMPT
# =========================================================

def generate_image_from_prompt(
    image_prompt,
    output_path
):
    """
    Generate an image using Hugging Face Stable Diffusion.
    """

    try:
        hf_token = os.getenv("HF_API_TOKEN") or HF_API_TOKEN

        if not hf_token:
            logging.warning(
                "HF_API_TOKEN is not configured. "
                "Skipping image generation."
            )

            return None

        api_url = (
            "https://api-inference.huggingface.co/"
            "models/stabilityai/"
            "stable-diffusion-xl-base-1.0"
        )

        headers = {
            "Authorization": f"Bearer {hf_token}"
        }

        payload = {
            "inputs": image_prompt
        }

        logging.info(
            f"Generating image for prompt: "
            f"{image_prompt}"
        )

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        )

        # Ensure output directory exists
        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image.save(output_path)

        logging.info(
            f"Image saved to: {output_path}"
        )

        return str(output_path)

    except Exception as e:
        logging.error(
            f"Image generation error for prompt "
            f"'{image_prompt}': {e}"
        )

        return None


# =========================================================
# GENERATE BLOG IMAGES
# =========================================================

def generate_images_from_blog_json(
    blog_json,
    output_dir="static/blogs"
):
    """
    Generate images from image prompts contained
    in the generated blog JSON.
    """

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    image_prompts = blog_json.get(
        "image_prompts",
        []
    )

    image_paths = []

    for idx, prompt in enumerate(
        image_prompts
    ):

        filename = (
            Path(output_dir)
            / f"image{idx + 1}.png"
        )

        image_path = generate_image_from_prompt(
            prompt,
            filename
        )

        if image_path:
            image_paths.append(
                str(image_path)
            )

    logging.info(
        f"Generated {len(image_paths)} "
        f"images from blog JSON"
    )

    return image_paths


# =========================================================
# CLEAN JSON
# =========================================================

def clean_json_string(json_string):
    """
    Remove Markdown code fences and invalid
    control characters from Gemini JSON output.
    """

    cleaned_string = re.sub(
        r"```json|```",
        "",
        json_string
    ).strip()

    cleaned_string = re.sub(
        r"[\x00-\x1F\x7F]",
        "",
        cleaned_string
    )

    return cleaned_string


# =========================================================
# SAVE JSON
# =========================================================

def save_json_to_file(
    json_string,
    filename="corrected_blog.json"
):
    """
    Parse Gemini JSON output and save it to a file.
    """

    corrected_json_str = clean_json_string(
        json_string
    )

    try:
        data = json.loads(
            corrected_json_str
        )

        parent_dir = Path(filename).parent

        # Only create directory if one exists
        if str(parent_dir) != ".":
            parent_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as corrected_file:

            json.dump(
                data,
                corrected_file,
                indent=4,
                ensure_ascii=False
            )

        logging.info(
            f"Corrected JSON saved as "
            f"'{filename}'"
        )

        return data

    except json.JSONDecodeError as e:

        logging.error(
            f"Error in JSON format: {e}"
        )

        return None


# =========================================================
# SAVE MARKDOWN FILE
# =========================================================

def save_markdown_file(
    blog_data,
    image_paths,
    output_dir="static/blogs"
):
    """
    Save generated blog content as Markdown.
    """

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    title = blog_data.get(
        "title",
        "Untitled Blog"
    )

    content = blog_data.get(
        "content",
        ""
    )

    markdown_lines = [
        f"# {title}\n",
        content.strip() + "\n"
    ]

    md_path = (
        Path(output_dir)
        / "blogs.md"
    )

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(markdown_lines)
        )

    logging.info(
        f"Markdown file saved to: {md_path}"
    )

    return str(md_path)


# =========================================================
# READ MARKDOWN
# =========================================================

def read_markdown_content(
    markdown_path: str
) -> str:
    """
    Read Markdown file content.
    """

    try:

        with open(
            markdown_path,
            "r",
            encoding="utf-8"
        ) as f:

            logging.info(
                f"Successfully read Markdown "
                f"content from: {markdown_path}"
            )

            return f.read()

    except Exception as e:

        logging.error(
            f"Failed to read Markdown file "
            f"'{markdown_path}': {e}"
        )

        raise RuntimeError(
            f"Failed to read Markdown file: {str(e)}"
        )