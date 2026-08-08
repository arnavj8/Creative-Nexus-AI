import os
import tempfile
import shutil
from pathlib import Path
import requests
import zipfile
import io
import numpy as np
import faiss
import threading

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai

from dotenv import load_dotenv

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

    # Preferred variable
    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )

    GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/arnavj8/Creative-Nexus-AI")

    logging.info(
        "Chatbot API configuration - "
        f"Gemini: {'Set' if GEMINI_API_KEY else 'Not set'}, "
        f"GitHub URL: {GITHUB_REPO_URL}"
    )

except Exception as e:
    logging.error(
        f"Error loading API keys: {str(e)}"
    )

    GEMINI_API_KEY = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
    )

    GITHUB_REPO_URL = os.getenv(
        "GITHUB_REPO_URL",
        "https://github.com/arnavj8/Creative-Nexus-AI"
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():
    """
    Create a Gemini client using the configured API key.
    """
    ensure_api_keys()

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GEN_API_KEY")
        or GEMINI_API_KEY
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured."
        )

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# KNOWLEDGE BASE
# =========================================================

class KnowledgeBase:

    def __init__(self):
        self.initialized = False
        self.repo_path = None
        self.embeddings = None
        self.index = None
        self.metadata = []

        self._initialization_lock = threading.Lock()
        self._initialization_in_progress = False

    # =====================================================
    # INITIALIZE
    # =====================================================

    def initialize(self):
        """
        Initialize the knowledge base.
        """

        temp_dir = None

        with self._initialization_lock:

            if self.initialized:
                logging.info(
                    "Knowledge base already initialized"
                )
                return True

            if self._initialization_in_progress:
                logging.info(
                    "Initialization already in progress"
                )
                return False

            self._initialization_in_progress = True

        try:

            logging.info(
                "Starting knowledge base initialization..."
            )

            # -------------------------------------------------
            # Validate environment variables
            # -------------------------------------------------

            ensure_api_keys()
            gemini_key = (
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GEN_API_KEY")
                or GEMINI_API_KEY
            )
            repo_url = (
                os.getenv("GITHUB_REPO_URL")
                or GITHUB_REPO_URL
                or "https://github.com/arnavj8/Creative-Nexus-AI"
            )

            if not gemini_key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured."
                )

            # -------------------------------------------------
            # Create Gemini client
            # -------------------------------------------------

            client = get_gemini_client()

            logging.info(
                "Gemini client initialized successfully"
            )

            # -------------------------------------------------
            # Setup embeddings
            # -------------------------------------------------

            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=gemini_key
            )

            logging.info(
                "Gemini embeddings initialized successfully"
            )

            # -------------------------------------------------
            # Create temporary directory
            # -------------------------------------------------

            temp_dir = tempfile.mkdtemp()

            self.repo_path = os.path.join(
                temp_dir,
                "repo"
            )

            os.makedirs(
                self.repo_path,
                exist_ok=True
            )

            # -------------------------------------------------
            # Download repository
            # -------------------------------------------------

            self._download_repo()

            # -------------------------------------------------
            # Process files
            # -------------------------------------------------

            self._process_files()

            # -------------------------------------------------
            # Mark initialized
            # -------------------------------------------------

            with self._initialization_lock:

                self.initialized = True
                self._initialization_in_progress = False

            logging.info(
                "Knowledge base initialization complete!"
            )

            return True

        except Exception as e:

            logging.error(
                f"Knowledge base initialization failed: "
                f"{str(e)}"
            )

            # -------------------------------------------------
            # Cleanup temporary directory
            # -------------------------------------------------

            if (
                temp_dir
                and os.path.exists(temp_dir)
            ):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    logging.error(
                        f"Cleanup error: "
                        f"{cleanup_error}"
                    )

            # -------------------------------------------------
            # Reset initialization state
            # -------------------------------------------------

            with self._initialization_lock:

                self.initialized = False
                self._initialization_in_progress = False

            return False

    # =====================================================
    # DOWNLOAD REPOSITORY
    # =====================================================

    def _download_repo(self):
        """
        Download the GitHub repository.
        """

        repo_parts = (
            GITHUB_REPO_URL
            .rstrip("/")
            .split("/")
        )

        if len(repo_parts) < 2:
            raise ValueError(
                "Invalid GITHUB_REPO_URL"
            )

        repo_name = repo_parts[-1]
        repo_owner = repo_parts[-2]

        # Remove .git if present
        repo_name = repo_name.replace(
            ".git",
            ""
        )

        # Try main branch first, then master
        for branch in ["main", "master"]:

            zip_url = (
                f"https://github.com/"
                f"{repo_owner}/"
                f"{repo_name}/"
                f"archive/refs/heads/"
                f"{branch}.zip"
            )

            logging.info(
                f"Trying to download repository "
                f"from {branch} branch"
            )

            try:

                response = requests.get(
                    zip_url,
                    timeout=60
                )

                if response.status_code == 200:

                    with zipfile.ZipFile(
                        io.BytesIO(
                            response.content
                        )
                    ) as z:

                        z.extractall(
                            self.repo_path
                        )

                    logging.info(
                        f"Downloaded and extracted "
                        f"{branch} branch"
                    )

                    # Find extracted directory
                    extracted_dirs = [
                        d
                        for d in os.listdir(
                            self.repo_path
                        )
                        if os.path.isdir(
                            os.path.join(
                                self.repo_path,
                                d
                            )
                        )
                    ]

                    if extracted_dirs:

                        self.repo_path = os.path.join(
                            self.repo_path,
                            extracted_dirs[0]
                        )

                    return

                logging.warning(
                    f"Branch {branch} unavailable. "
                    f"Status: {response.status_code}"
                )

            except requests.RequestException as e:

                logging.warning(
                    f"Error downloading {branch} "
                    f"branch: {e}"
                )

        raise Exception(
            "Failed to download repository"
        )

    # =====================================================
    # PROCESS FILES
    # =====================================================

    def _process_files(self):
        """
        Process repository files and create
        FAISS embeddings.
        """

        text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200
            )
        )

        all_embeddings = []
        doc_id = 0

        # -------------------------------------------------
        # Process files
        # -------------------------------------------------

        for file_path in Path(
            self.repo_path
        ).glob("**/*"):

            if (
                file_path.is_file()
                and self._should_process_file(
                    file_path
                )
            ):

                try:

                    relative_path = (
                        file_path.relative_to(
                            self.repo_path
                        )
                    )

                    content = file_path.read_text(
                        encoding="utf-8",
                        errors="ignore"
                    )

                    chunks = (
                        text_splitter.split_text(
                            content
                        )
                    )

                    logging.info(
                        f"Processing "
                        f"{relative_path}: "
                        f"{len(chunks)} chunks"
                    )

                    for i, chunk in enumerate(
                        chunks
                    ):

                        try:

                            embedding = (
                                self.embeddings
                                .embed_documents(
                                    [chunk]
                                )[0]
                            )

                            if embedding:

                                all_embeddings.append(
                                    embedding
                                )

                                self.metadata.append(
                                    {
                                        "source": str(
                                            relative_path
                                        ),
                                        "chunk": i,
                                        "total_chunks": len(
                                            chunks
                                        ),
                                        "text": chunk
                                    }
                                )

                                doc_id += 1

                        except Exception as e:

                            logging.error(
                                f"Error embedding "
                                f"chunk {i} from "
                                f"{relative_path}: "
                                f"{str(e)}"
                            )

                            continue

                except Exception as e:

                    logging.error(
                        f"Error processing "
                        f"{file_path}: "
                        f"{str(e)}"
                    )

                    continue

        # -------------------------------------------------
        # Create FAISS index
        # -------------------------------------------------

        if all_embeddings:

            dimension = len(
                all_embeddings[0]
            )

            self.index = faiss.IndexFlatL2(
                dimension
            )

            self.index.add(
                np.array(
                    all_embeddings
                ).astype("float32")
            )

            logging.info(
                f"Processed {doc_id} documents "
                f"and created FAISS index"
            )

        else:

            raise Exception(
                "No embeddings were generated"
            )

    # =====================================================
    # FILE FILTER
    # =====================================================

    def _should_process_file(
        self,
        file_path
    ):
        """
        Determine whether a file should
        be processed.
        """

        # Ignore hidden directories/files
        if any(
            part.startswith(".")
            for part in file_path.parts
        ):
            return False

        ignored_extensions = {
            ".exe",
            ".bin",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".mp4",
            ".mp3",
            ".zip",
            ".tar",
            ".gz",
            ".pdf",
            ".pyc",
            ".ttf",
            ".html",
            ".js",
        }

        if (
            file_path.suffix.lower()
            in ignored_extensions
        ):
            return False

        # Skip files larger than 1 MB
        if file_path.stat().st_size > 1024 * 1024:
            return False

        return True

    # =====================================================
    # QUERY KNOWLEDGE BASE
    # =====================================================

    def query(
        self,
        user_query: str
    ) -> str:
        """
        Query the knowledge base and generate
        a response using Gemini.
        """

        if not self.initialized:
            return (
                "Knowledge base not initialized"
            )

        try:

            # -------------------------------------------------
            # Generate query embedding
            # -------------------------------------------------

            query_embedding = (
                self.embeddings.embed_query(
                    user_query
                )
            )

            query_embedding = (
                np.array(
                    query_embedding
                )
                .astype("float32")
                .reshape(1, -1)
            )

            # -------------------------------------------------
            # Search FAISS
            # -------------------------------------------------

            distances, indices = (
                self.index.search(
                    query_embedding,
                    5
                )
            )

            # -------------------------------------------------
            # Prepare context
            # -------------------------------------------------

            context_parts = []

            for idx in indices[0]:

                # Safety check
                if (
                    idx < 0
                    or idx >= len(self.metadata)
                ):
                    continue

                metadata = self.metadata[idx]

                context_parts.append(
                    f"[From "
                    f"{metadata['source']}, "
                    f"chunk "
                    f"{metadata['chunk'] + 1}/"
                    f"{metadata['total_chunks']}]\n"
                    f"{metadata['text']}"
                )

            context = "\n\n".join(
                context_parts
            )

            # -------------------------------------------------
            # Gemini client
            # -------------------------------------------------

            client = get_gemini_client()

            # -------------------------------------------------
            # Prompt
            # -------------------------------------------------

            prompt = f"""
You are a specialized AI assistant focused on
explaining this specific codebase and project.

Your primary role is to provide accurate,
technical, and helpful information about the
project's implementation, architecture, and
functionality.

Guidelines for responding:

1. Greeting Handling:
- If the user sends a greeting such as hello,
  hi, or hey, respond warmly and offer assistance.
- Mention that you're an AI assistant for this
  project's code repository only on greetings.

2. Question Answering:
- Use ONLY the provided context to formulate
  your responses.
- Be precise and directly address the user's query.
- If the information is not in the context,
  clearly state that you don't have enough
  information.

3. Technical Explanation:
- When explaining code, provide detailed
  technical breakdowns.
- Include relevant code snippets from the
  context if available.
- Explain implementation logic and design
  patterns.

4. Architecture Discussion:
- When discussing project structure, explain
  relationships between components.
- Highlight system design decisions and their
  implications.
- Focus on how different parts interact.

5. Error Handling:
- For debugging questions, analyze potential
  issues systematically.
- Suggest troubleshooting steps based on the
  codebase.
- Reference specific error-handling patterns
  in the code.

6. Response Style:
- Be friendly and professional.
- Provide clear and concise answers.
- If uncertain, admit the limitation honestly.

CONTEXT:

{context}

CURRENT QUERY:

{user_query}

Additional Instructions:

- First identify the most relevant guideline
  category for this query.
- Follow those guidelines while maintaining
  natural conversation flow.
- Prioritize accuracy over speculation.
- Try to answer in a maximum of 150 words.
- If asked about your creation, mention you're
  an AI assistant for the project.
- Focus on helping users understand the
  project's code and functionality.
"""

            # -------------------------------------------------
            # Generate Gemini response
            # -------------------------------------------------

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            if response and response.text:
                return response.text

            return (
                "Could not generate response"
            )

        except Exception as e:

            logging.error(
                f"Knowledge base query error: "
                f"{str(e)}"
            )

            return f"Error: {str(e)}"

    # =====================================================
    # CLEANUP
    # =====================================================

    def cleanup(self):
        """
        Cleanup temporary repository
        and FAISS resources.
        """

        try:

            if (
                self.repo_path
                and os.path.exists(
                    self.repo_path
                )
            ):

                parent_dir = os.path.dirname(
                    self.repo_path
                )

                if os.path.exists(
                    parent_dir
                ):
                    shutil.rmtree(
                        parent_dir
                    )

            self.initialized = False
            self.index = None
            self.metadata = []
            self.embeddings = None
            self.repo_path = None

            logging.info(
                "Knowledge base cleaned up"
            )

        except Exception as e:

            logging.error(
                f"Error during cleanup: "
                f"{str(e)}"
            )