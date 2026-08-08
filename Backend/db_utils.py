import os
from typing import Dict, Optional

from pymongo import MongoClient
from dotenv import load_dotenv

from Backend.logger import logging

load_dotenv()


class DatabaseManager:
    _instance = None
    _keys = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._keys = cls._get_environment_keys()
            logging.info("DatabaseManager initialized")

        return cls._instance

    @staticmethod
    def _get_environment_keys() -> Dict[str, Optional[str]]:
        """
        Get API keys from environment variables.

        GEMINI_API_KEY is the preferred Gemini variable.
        GEN_API_KEY is supported for backward compatibility.
        """

        gemini_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GEN_API_KEY")
        )

        huggingface_key = os.getenv("HF_API_TOKEN")

        return {
            "gemini_key": gemini_key,
            "huggingface_key": huggingface_key
        }

    def initialize(self, mongo_uri: str):
        """
        Initialize MongoDB connection.

        Environment variables always have priority.
        MongoDB keys are only used when an environment key
        is missing.
        """

        self._keys = self._get_environment_keys()

        try:
            logging.info("Attempting to initialize database connection")

            self.client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000
            )

            self.db = self.client.ai_generator

            # Test MongoDB connection
            self.client.server_info()

            logging.info("MongoDB connection successful")

            keys_doc = self.db.api_keys.find_one({"_id": "keys"})

            if keys_doc:
                # NEVER overwrite environment variables.
                # MongoDB is only a fallback.

                if not self._keys.get("gemini_key"):
                    self._keys["gemini_key"] = keys_doc.get("gemini_key")

                if not self._keys.get("huggingface_key"):
                    self._keys["huggingface_key"] = keys_doc.get(
                        "huggingface_key"
                    )

                logging.info(
                    "MongoDB API keys loaded only for missing environment keys"
                )
            else:
                logging.info(
                    "No API keys document found in MongoDB"
                )

            return True

        except Exception as e:
            logging.error(f"Database connection error: {e}")
            logging.info(
                "Continuing with environment API keys"
            )

            return False

    def save_keys(
        self,
        gemini_key: str,
        huggingface_key: str
    ):
        """
        Save API keys to MongoDB.

        Environment variables remain the preferred source.
        """

        try:
            if not hasattr(self, "db"):
                logging.error(
                    "Database is not initialized"
                )
                return False

            self.db.api_keys.update_one(
                {"_id": "keys"},
                {
                    "$set": {
                        "gemini_key": gemini_key,
                        "huggingface_key": huggingface_key
                    }
                },
                upsert=True
            )

            self._keys = {
                "gemini_key": gemini_key,
                "huggingface_key": huggingface_key
            }

            logging.info(
                "API keys saved to MongoDB"
            )

            return True

        except Exception as e:
            logging.error(
                f"Error saving API keys: {e}"
            )
            return False

    def get_keys(self) -> Dict[str, Optional[str]]:
        """
        Return currently configured API keys.

        Environment variables are checked again so that
        Render/local environment changes are respected.
        """

        environment_keys = self._get_environment_keys()

        # Environment variables always take priority.
        if environment_keys.get("gemini_key"):
            self._keys["gemini_key"] = environment_keys["gemini_key"]

        if environment_keys.get("huggingface_key"):
            self._keys["huggingface_key"] = (
                environment_keys["huggingface_key"]
            )

        if not self._keys:
            self._keys = environment_keys

        return self._keys

    def is_using_default_keys(self) -> bool:
        """
        Kept for backward compatibility.

        Returns True when at least one key is missing.
        """

        keys = self.get_keys()

        return not bool(
            keys.get("gemini_key")
            and keys.get("huggingface_key")
        )


def get_api_keys():
    """
    Get API keys and expose them through environment variables.
    """

    db = DatabaseManager.get_instance()
    keys = db.get_keys()

    gemini_key = keys.get("gemini_key")
    huggingface_key = keys.get("huggingface_key")

    if not gemini_key:
        logging.error("Gemini API key not configured")
        raise ValueError(
            "Gemini API key not configured"
        )

    if not huggingface_key:
        logging.warning(
            "Hugging Face API token not configured"
        )

    # Canonical Gemini variable
    os.environ["GEMINI_API_KEY"] = gemini_key

    # Backward compatibility with existing project code
    os.environ["GEN_API_KEY"] = gemini_key

    if huggingface_key:
        os.environ["HF_API_TOKEN"] = huggingface_key

    return keys


def ensure_api_keys():
    """
    Ensure API keys are available.

    Render environment variables are the primary source.
    MongoDB is only a fallback.
    """

    try:
        db = DatabaseManager.get_instance()
        keys = db.get_keys()

        gemini_key = keys.get("gemini_key")
        huggingface_key = keys.get("huggingface_key")

        if not gemini_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured"
            )

        # Set canonical Gemini variable
        os.environ["GEMINI_API_KEY"] = gemini_key

        # Backward compatibility
        os.environ["GEN_API_KEY"] = gemini_key

        if huggingface_key:
            os.environ["HF_API_TOKEN"] = huggingface_key

        logging.info(
            "Gemini API key loaded successfully"
        )

        logging.info(
            f"Hugging Face API token: "
            f"{'configured' if huggingface_key else 'not configured'}"
        )

        return True

    except Exception as e:
        logging.error(
            f"Error ensuring API keys: {e}"
        )
        return False