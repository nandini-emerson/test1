
# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    # --- API Keys and Endpoints ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CRM_CONTACT_API_URL = os.getenv("CRM_CONTACT_API_URL")
    CRM_ACCOUNT_API_URL = os.getenv("CRM_ACCOUNT_API_URL")
    CRM_QUOTE_API_URL = os.getenv("CRM_QUOTE_API_URL")
    PRICING_VALIDATION_API_URL = os.getenv("PRICING_VALIDATION_API_URL")
    CRM_QUOTEORDER_API_URL = os.getenv("CRM_QUOTEORDER_API_URL")
    CRM_API_AUTH_TOKEN = os.getenv("CRM_API_AUTH_TOKEN")  # OAuth2.0 token or API key

    # --- Redis/Cache ---
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # --- LLM Configuration ---
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    LLM_SYSTEM_PROMPT = (
        "You are a professional CRM Quote Agent. Your role is to validate inputs, resolve customer and account data, "
        "and create quotes and quote orders in the CRM system via API. Always enforce business rules, request only the minimum "
        "additional information needed, and provide deterministic, auditable results. Never guess or fabricate data. Communicate clearly and concisely."
    )
    LLM_USER_PROMPT_TEMPLATE = (
        "To proceed with your quote request, please provide the following missing information: {missing_fields}. If you have questions, let me know."
    )
    LLM_FEW_SHOT_EXAMPLES = [
        "customerEmailId is missing. => To continue, please provide the customer's email address.",
        "receivedDateTime is not in ISO-8601 format. => The received date/time must be in ISO-8601 format (e.g., 2024-06-01T12:00:00Z). Please provide a valid value.",
        "Line items are missing. => Please provide at least one product/SKU, quantity, and requested start/end dates to create the quote."
    ]

    # --- Domain/Business Rules ---
    DOMAIN = "general"
    AGENT_NAME = "CRM Quote Creation Agent edited"
    ENFORCE_IDEMPOTENCY = True
    AUDIT_LOG_RETENTION_DAYS = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "90"))
    MAX_INPUT_SIZE = int(os.getenv("MAX_INPUT_SIZE", "50000"))
    RESPONSE_TIME_MS = int(os.getenv("RESPONSE_TIME_MS", "1500"))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50"))

    # --- Validation/Error Handling Defaults ---
    EMAIL_REGEX = r"^[^@]+@[^@]+\.[^@]+$"
    ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
    ERROR_STRATEGY = "strict"
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
    BACKOFF_BASE = int(os.getenv("BACKOFF_BASE", "2"))
    FALLBACK_PROMPT = "Please provide the required information to proceed."

    # --- Security ---
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "dummy_key_for_dev_only")
    MASK_PII_IN_LOGS = True

    # --- API Rate Limits/Backoff ---
    API_RATE_LIMITS = {
        "CRM_Contact_API": {"backoff": True},
        "CRM_Account_API": {"backoff": True},
        "CRM_Quote_API": {"backoff": True},
        "Pricing_Validation_API": {"backoff": True},
        "CRM_QuoteOrder_API": {"backoff": True},
    }

    # --- Default Values/Fallbacks ---
    DEFAULT_QUOTE_STATUS = "PENDING"
    DEFAULT_ORDER_STATUS = "PENDING"

    @classmethod
    def validate(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.CRM_CONTACT_API_URL:
            missing.append("CRM_CONTACT_API_URL")
        if not cls.CRM_ACCOUNT_API_URL:
            missing.append("CRM_ACCOUNT_API_URL")
        if not cls.CRM_QUOTE_API_URL:
            missing.append("CRM_QUOTE_API_URL")
        if not cls.PRICING_VALIDATION_API_URL:
            missing.append("PRICING_VALIDATION_API_URL")
        if not cls.CRM_QUOTEORDER_API_URL:
            missing.append("CRM_QUOTEORDER_API_URL")
        if not cls.CRM_API_AUTH_TOKEN:
            missing.append("CRM_API_AUTH_TOKEN")
        if not cls.ENCRYPTION_KEY or cls.ENCRYPTION_KEY == "dummy_key_for_dev_only":
            missing.append("ENCRYPTION_KEY (should be set securely in production)")
        if missing:
            raise ConfigError(f"Missing required configuration: {', '.join(missing)}")

# Validate config at import time
try:
    Config.validate()
except ConfigError as e:
    # Comment out the next line if you want to allow import without raising
    # raise
    print(f"Configuration error: {e}")

# Usage example:
# from config import Config
# api_key = Config.OPENAI_API_KEY
