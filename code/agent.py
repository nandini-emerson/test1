
import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, validator, root_validator
from dotenv import load_dotenv
from loguru import logger
from email_validator import validate_email, EmailNotValidError
from dateutil.parser import parse as parse_datetime, ParserError
import openai
import redis
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cryptography.fernet import Fernet

# -------------------- Configuration Management --------------------

class Config:
    """Centralized configuration management."""
    load_dotenv()
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CRM_CONTACT_API_URL: str = os.getenv("CRM_CONTACT_API_URL", "")
    CRM_ACCOUNT_API_URL: str = os.getenv("CRM_ACCOUNT_API_URL", "")
    CRM_QUOTE_API_URL: str = os.getenv("CRM_QUOTE_API_URL", "")
    PRICING_VALIDATION_API_URL: str = os.getenv("PRICING_VALIDATION_API_URL", "")
    CRM_QUOTEORDER_API_URL: str = os.getenv("CRM_QUOTEORDER_API_URL", "")
    CRM_API_AUTH_TOKEN: str = os.getenv("CRM_API_AUTH_TOKEN", "")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    AUDIT_LOG_RETENTION_DAYS: int = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "90"))
    MAX_INPUT_SIZE: int = 50000
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000
    SYSTEM_PROMPT: str = (
        "You are a professional CRM Quote Agent. Your role is to validate inputs, resolve customer and account data, "
        "and create quotes and quote orders in the CRM system via API. Always enforce business rules, request only the minimum "
        "additional information needed, and provide deterministic, auditable results. Never guess or fabricate data. Communicate clearly and concisely."
    )
    USER_PROMPT_TEMPLATE: str = (
        "To proceed with your quote request, please provide the following missing information: {missing_fields}. If you have questions, let me know."
    )
    FEW_SHOT_EXAMPLES: List[str] = [
        "customerEmailId is missing. => To continue, please provide the customer's email address.",
        "receivedDateTime is not in ISO-8601 format. => The received date/time must be in ISO-8601 format (e.g., 2024-06-01T12:00:00Z). Please provide a valid value.",
        "Line items are missing. => Please provide at least one product/SKU, quantity, and requested start/end dates to create the quote."
    ]
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """Validate critical configuration."""
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.CRM_CONTACT_API_URL:
            missing.append("CRM_CONTACT_API_URL")
        if not cls.CRM_ACCOUNT_API_URL:
            missing.append("CRM_ACCOUNT_API_URL")
        if not cls.CRM_QUOTE_API_URL:
            missing.append("CRM_QUOTE_API_URL")
        if not cls.CRM_QUOTEORDER_API_URL:
            missing.append("CRM_QUOTEORDER_API_URL")
        if not cls.PRICING_VALIDATION_API_URL:
            missing.append("PRICING_VALIDATION_API_URL")
        if not cls.CRM_API_AUTH_TOKEN:
            missing.append("CRM_API_AUTH_TOKEN")
        if missing:
            raise RuntimeError(f"Missing required configuration: {', '.join(missing)}")

Config.validate()

# -------------------- Logging Configuration --------------------

logger.remove()
logger.add("crm_quote_agent.log", rotation="10 MB", retention=f"{Config.AUDIT_LOG_RETENTION_DAYS} days", level=Config.LOG_LEVEL)
logger.add(lambda msg: print(msg, end=""), level=Config.LOG_LEVEL)

# -------------------- Persistence Layer --------------------

class RedisCache:
    """Idempotency cache using Redis."""
    def __init__(self, url: str):
        self.redis = redis.Redis.from_url(url, decode_responses=True)

    def check_duplicate(self, key: str) -> bool:
        return self.redis.exists(key)

    def store_request(self, key: str, value: str, ttl: int = 3600):
        self.redis.set(key, value, ex=ttl)

class AuditLogger:
    """Audit logging with PII masking and encryption."""
    def __init__(self, encryption_key: str):
        self.fernet = Fernet(encryption_key.encode())

    def mask_email(self, email: str) -> str:
        """Mask email for logs."""
        if not email or "@" not in email:
            return "*****"
        local, domain = email.split("@")
        masked_local = local[0] + "***" + local[-1] if len(local) > 2 else "***"
        return f"{masked_local}@{domain}"

    def redact_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII fields."""
        redacted = data.copy()
        if "customerEmailId" in redacted:
            redacted["customerEmailId"] = self.mask_email(redacted["customerEmailId"])
        return redacted

    def encrypt_log(self, log_data: Dict[str, Any]) -> str:
        """Encrypt log data."""
        serialized = json.dumps(log_data)
        return self.fernet.encrypt(serialized.encode()).decode()

    def log_action(self, action: str, details: Dict[str, Any]):
        """Log action with PII masking and encryption."""
        redacted_details = self.redact_pii(details)
        encrypted_log = self.encrypt_log({"action": action, "details": redacted_details})
        logger.info(f"AUDIT_LOG: {encrypted_log}")

# -------------------- Presentation Layer --------------------

class QuoteRequestModel(BaseModel):
    customerEmailId: Optional[str] = Field(None, description="Customer email address")
    receivedDateTime: Optional[str] = Field(None, description="ISO-8601 datetime string")
    lineItems: Optional[List[Dict[str, Any]]] = Field(None, description="List of line items")
    additionalFields: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("customerEmailId")
    def validate_email(cls, v):
        if v is None or not v.strip():
            raise ValueError("customerEmailId is required and cannot be empty.")
        try:
            validate_email(v)
        except EmailNotValidError:
            raise ValueError("customerEmailId is not a valid email address.")
        return v.strip()

    @validator("receivedDateTime")
    def validate_datetime(cls, v):
        if v is None or not v.strip():
            raise ValueError("receivedDateTime is required and cannot be empty.")
        try:
            parse_datetime(v)
        except (ParserError, ValueError):
            raise ValueError("receivedDateTime must be in ISO-8601 format (e.g., 2024-06-01T12:00:00Z).")
        return v.strip()

    @validator("lineItems")
    def validate_line_items(cls, v):
        if v is None or not isinstance(v, list) or len(v) == 0:
            raise ValueError("At least one line item is required.")
        for item in v:
            if not isinstance(item, dict):
                raise ValueError("Each line item must be a dictionary.")
            if "productSKU" not in item or not item["productSKU"]:
                raise ValueError("Each line item must have a productSKU.")
            if "quantity" not in item or not isinstance(item["quantity"], int) or item["quantity"] <= 0:
                raise ValueError("Each line item must have a positive integer quantity.")
            if "startDate" not in item or not item["startDate"]:
                raise ValueError("Each line item must have a startDate.")
            if "endDate" not in item or not item["endDate"]:
                raise ValueError("Each line item must have an endDate.")
        return v

    @root_validator
    def check_input_size(cls, values):
        total_size = len(json.dumps(values))
        if total_size > Config.MAX_INPUT_SIZE:
            raise ValueError(f"Input payload exceeds maximum allowed size ({Config.MAX_INPUT_SIZE} characters).")
        return values

# -------------------- Domain Layer --------------------

class InputValidator:
    """Validates customerEmailId, receivedDateTime, and required fields according to business rules."""
    @staticmethod
    def validate_required_fields(payload: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        missing_fields = []
        cleaned_payload = {}
        # Validate email
        email = payload.get("customerEmailId")
        if not email:
            missing_fields.append("customerEmailId")
        else:
            try:
                validate_email(email)
                cleaned_payload["customerEmailId"] = email.strip()
            except EmailNotValidError:
                missing_fields.append("customerEmailId (invalid format)")
        # Validate datetime
        dt = payload.get("receivedDateTime")
        if not dt:
            missing_fields.append("receivedDateTime")
        else:
            try:
                parse_datetime(dt)
                cleaned_payload["receivedDateTime"] = dt.strip()
            except (ParserError, ValueError):
                missing_fields.append("receivedDateTime (invalid format)")
        # Validate line items
        line_items = payload.get("lineItems")
        if not line_items or not isinstance(line_items, list) or len(line_items) == 0:
            missing_fields.append("lineItems")
        else:
            valid_items = []
            for item in line_items:
                if not isinstance(item, dict):
                    missing_fields.append("lineItems (invalid structure)")
                    continue
                if not item.get("productSKU"):
                    missing_fields.append("lineItems.productSKU")
                if not item.get("quantity") or not isinstance(item["quantity"], int) or item["quantity"] <= 0:
                    missing_fields.append("lineItems.quantity")
                if not item.get("startDate"):
                    missing_fields.append("lineItems.startDate")
                if not item.get("endDate"):
                    missing_fields.append("lineItems.endDate")
                valid_items.append(item)
            cleaned_payload["lineItems"] = valid_items
        return missing_fields, cleaned_payload

# -------------------- Integration Layer --------------------

class CustomerResolver:
    """Resolves customer and account context using CRM_Contact_API and CRM_Account_API."""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(requests.RequestException))
    def resolve_customer(self, email: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {Config.CRM_API_AUTH_TOKEN}"}
        response = requests.get(f"{Config.CRM_CONTACT_API_URL}?email={email}", headers=headers, timeout=5)
        if response.status_code == 404:
            return {"error": "CUSTOMER_NOT_FOUND"}
        elif response.status_code != 200:
            raise requests.RequestException(f"CRM_Contact_API error: {response.status_code}")
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(requests.RequestException))
    def resolve_account(self, account_id: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {Config.CRM_API_AUTH_TOKEN}"}
        response = requests.get(f"{Config.CRM_ACCOUNT_API_URL}?accountId={account_id}", headers=headers, timeout=5)
        if response.status_code == 404:
            return {"error": "ACCOUNT_NOT_FOUND"}
        elif response.status_code != 200:
            raise requests.RequestException(f"CRM_Account_API error: {response.status_code}")
        return response.json()

class QuoteManager:
    """Builds and submits quote payloads, validates line items, and manages quote creation via CRM_Quote_API."""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(requests.RequestException))
    def create_quote(self, resolved_fields: Dict[str, Any], line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {Config.CRM_API_AUTH_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "customerEmailId": resolved_fields.get("customerEmailId"),
            "accountId": resolved_fields.get("accountId"),
            "receivedDateTime": resolved_fields.get("receivedDateTime"),
            "lineItems": line_items
        }
        response = requests.post(Config.CRM_QUOTE_API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code != 200:
            raise requests.RequestException(f"CRM_Quote_API error: {response.status_code}")
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(requests.RequestException))
    def validate_pricing(self, quote_id: str, line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {Config.CRM_API_AUTH_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "quoteId": quote_id,
            "lineItems": line_items
        }
        response = requests.post(Config.PRICING_VALIDATION_API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code != 200:
            raise requests.RequestException(f"Pricing_Validation_API error: {response.status_code}")
        return response.json()

class OrderManager:
    """Submits quote orders to CRM_QuoteOrder_API and manages order status."""
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(requests.RequestException))
    def create_quote_order(self, quote_id: str, pricing_status: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {Config.CRM_API_AUTH_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "quoteId": quote_id,
            "pricingStatus": pricing_status
        }
        response = requests.post(Config.CRM_QUOTEORDER_API_URL, headers=headers, json=payload, timeout=10)
        if response.status_code != 200:
            raise requests.RequestException(f"CRM_QuoteOrder_API error: {response.status_code}")
        return response.json()

# -------------------- Idempotency Layer --------------------

class IdempotencyManager:
    """Prevents duplicate quote/order creation by caching recent requests and checking for duplicates."""
    def __init__(self, cache: RedisCache):
        self.cache = cache

    def _make_key(self, customer_email: str, received_dt: str) -> str:
        return f"idempotency:{customer_email}:{received_dt}"

    def check_duplicate(self, customer_email: str, received_dt: str) -> bool:
        key = self._make_key(customer_email, received_dt)
        return self.cache.check_duplicate(key)

    def store_request(self, customer_email: str, received_dt: str, request_id: str):
        key = self._make_key(customer_email, received_dt)
        self.cache.store_request(key, request_id)

# -------------------- LLM Adapter --------------------

class LLMInterface:
    """Formats prompts, interacts with the LLM for communication, and applies response templates."""
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, system_prompt: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    async def generate_prompt(self, missing_fields: List[str]) -> str:
        """Generate user-facing prompt for missing/invalid fields using LLM."""
        user_prompt = Config.USER_PROMPT_TEMPLATE.format(missing_fields=", ".join(missing_fields))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM prompt generation failed: {str(e)}")
            # Fallback to template
            return user_prompt

    async def parse_response(self, response_text: str) -> str:
        """Parse LLM response (for future extensibility)."""
        return response_text

# -------------------- Error Handling Layer --------------------

class ErrorHandler(Exception):
    """Handles errors, retries, fallback, and escalation to human-in-the-loop."""
    def __init__(self, message: str, error_type: str = "GENERAL_ERROR", tips: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.tips = tips or ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "success": False,
            "error_type": self.error_type,
            "error_message": self.message,
            "tips": self.tips
        }

# -------------------- Application Layer --------------------

class CRMQuoteCreationAgent:
    """Main agent orchestrating the quote creation workflow."""
    def __init__(self):
        self.input_validator = InputValidator()
        self.customer_resolver = CustomerResolver()
        self.quote_manager = QuoteManager()
        self.order_manager = OrderManager()
        self.audit_logger = AuditLogger(Config.ENCRYPTION_KEY)
        self.redis_cache = RedisCache(Config.REDIS_URL)
        self.idempotency_manager = IdempotencyManager(self.redis_cache)
        self.llm_interface = LLMInterface(
            api_key=Config.OPENAI_API_KEY,
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            system_prompt=Config.SYSTEM_PROMPT
        )

    async def process_request(self, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for processing a quote creation request."""
        try:
            # Input validation
            missing_fields, cleaned_payload = self.input_validator.validate_required_fields(input_payload)
            if missing_fields:
                prompt = await self.llm_interface.generate_prompt(missing_fields)
                self.audit_logger.log_action("INPUT_VALIDATION_FAILED", {"missing_fields": missing_fields, "input": input_payload})
                return {
                    "success": False,
                    "error_type": "VALIDATION_ERROR",
                    "error_message": f"Missing or invalid fields: {', '.join(missing_fields)}",
                    "tips": prompt
                }

            customer_email = cleaned_payload["customerEmailId"]
            received_dt = cleaned_payload["receivedDateTime"]

            # Idempotency check
            if self.idempotency_manager.check_duplicate(customer_email, received_dt):
                self.audit_logger.log_action("DUPLICATE_REQUEST", {"customerEmailId": customer_email, "receivedDateTime": received_dt})
                return {
                    "success": False,
                    "error_type": "DUPLICATE_REQUEST",
                    "error_message": "Duplicate quote/order request detected.",
                    "tips": "This quote/order has already been processed. Please check your records or contact support."
                }

            # Resolve customer
            customer_info = self.customer_resolver.resolve_customer(customer_email)
            if "error" in customer_info:
                self.audit_logger.log_action("CUSTOMER_RESOLUTION_FAILED", {"customerEmailId": customer_email, "error": customer_info["error"]})
                return {
                    "success": False,
                    "error_type": customer_info["error"],
                    "error_message": "Customer not found or not eligible.",
                    "tips": "Please verify the customer's email address or eligibility."
                }
            account_id = customer_info.get("accountId")
            if not account_id:
                self.audit_logger.log_action("ACCOUNT_RESOLUTION_FAILED", {"customerEmailId": customer_email, "error": "ACCOUNT_ID_MISSING"})
                return {
                    "success": False,
                    "error_type": "ACCOUNT_ID_MISSING",
                    "error_message": "Account ID missing for customer.",
                    "tips": "Please contact support to resolve account linkage."
                }

            account_info = self.customer_resolver.resolve_account(account_id)
            if "error" in account_info:
                self.audit_logger.log_action("ACCOUNT_RESOLUTION_FAILED", {"accountId": account_id, "error": account_info["error"]})
                return {
                    "success": False,
                    "error_type": account_info["error"],
                    "error_message": "Account not found or not eligible.",
                    "tips": "Please verify the account status or contact support."
                }

            resolved_fields = {
                "customerEmailId": customer_email,
                "accountId": account_id,
                "receivedDateTime": received_dt
            }
            line_items = cleaned_payload["lineItems"]

            # Create quote
            quote_result = self.quote_manager.create_quote(resolved_fields, line_items)
            quote_id = quote_result.get("quoteId")
            quote_status = quote_result.get("quoteStatus")
            if not quote_id or not quote_status:
                self.audit_logger.log_action("QUOTE_CREATION_FAILED", {"resolved_fields": resolved_fields, "line_items": line_items, "error": "QUOTE_ID_OR_STATUS_MISSING"})
                return {
                    "success": False,
                    "error_type": "QUOTE_CREATION_ERROR",
                    "error_message": "Quote creation failed. Missing quoteId or quoteStatus.",
                    "tips": "Please retry or contact support."
                }

            # Validate pricing
            pricing_result = self.quote_manager.validate_pricing(quote_id, line_items)
            pricing_status = pricing_result.get("pricingStatus")
            if not pricing_status or pricing_status != "VALID":
                self.audit_logger.log_action("PRICING_VALIDATION_FAILED", {"quoteId": quote_id, "pricing_result": pricing_result})
                return {
                    "success": False,
                    "error_type": "PRICING_VALIDATION_ERROR",
                    "error_message": "Pricing validation failed.",
                    "tips": "Please review line items and discounts."
                }

            # Create quote order
            order_result = self.order_manager.create_quote_order(quote_id, pricing_result)
            quote_order_id = order_result.get("quoteOrderId")
            order_status = order_result.get("orderStatus")
            if not quote_order_id or not order_status:
                self.audit_logger.log_action("QUOTE_ORDER_CREATION_FAILED", {"quoteId": quote_id, "pricing_result": pricing_result, "error": "QUOTE_ORDER_ID_OR_STATUS_MISSING"})
                return {
                    "success": False,
                    "error_type": "QUOTE_ORDER_CREATION_ERROR",
                    "error_message": "Quote order creation failed. Missing quoteOrderId or orderStatus.",
                    "tips": "Please retry or contact support."
                }

            # Store idempotency
            request_id = quote_order_id
            self.idempotency_manager.store_request(customer_email, received_dt, request_id)

            # Audit log
            self.audit_logger.log_action("QUOTE_ORDER_CREATED", {
                "customerEmailId": customer_email,
                "accountId": account_id,
                "quoteId": quote_id,
                "quoteOrderId": quote_order_id,
                "status": order_status
            })

            return {
                "success": True,
                "quoteId": quote_id,
                "quoteStatus": quote_status,
                "quoteOrderId": quote_order_id,
                "orderStatus": order_status,
                "correlationId": request_id
            }

        except ErrorHandler as eh:
            self.audit_logger.log_action("ERROR_HANDLER", {"error_type": eh.error_type, "error_message": eh.message})
            return eh.to_json()
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}")
            self.audit_logger.log_action("UNHANDLED_EXCEPTION", {"exception": str(e)})
            return {
                "success": False,
                "error_type": "UNHANDLED_EXCEPTION",
                "error_message": str(e),
                "tips": "Please check your input, ensure JSON is properly formatted, and retry. If the issue persists, contact support."
            }

# -------------------- FastAPI Presentation Layer --------------------

app = FastAPI(title="CRM Quote Creation Agent", version="1.0.0")

agent = CRMQuoteCreationAgent()

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "INPUT_VALIDATION_ERROR",
            "error_message": "Input validation failed.",
            "tips": "Check required fields, email format, ISO-8601 datetime, and line item structure. Ensure JSON is properly formatted."
        }
    )

@app.exception_handler(json.JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    logger.error(f"Malformed JSON: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error_type": "MALFORMED_JSON",
            "error_message": f"Malformed JSON: {str(exc)}",
            "tips": "Check for missing quotes, commas, or brackets. Ensure your JSON is valid and does not exceed 50,000 characters."
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Generic error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "INTERNAL_SERVER_ERROR",
            "error_message": str(exc),
            "tips": "Unexpected error occurred. Please retry or contact support."
        }
    )

@app.post("/create-quote", response_model=None)
async def create_quote_endpoint(request: Request):
    try:
        body = await request.body()
        if len(body) > Config.MAX_INPUT_SIZE:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "success": False,
                    "error_type": "INPUT_TOO_LARGE",
                    "error_message": f"Input exceeds maximum allowed size ({Config.MAX_INPUT_SIZE} characters).",
                    "tips": "Reduce input size and retry."
                }
            )
        try:
            payload = json.loads(body.decode())
        except json.JSONDecodeError as exc:
            logger.error(f"Malformed JSON: {str(exc)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error_type": "MALFORMED_JSON",
                    "error_message": f"Malformed JSON: {str(exc)}",
                    "tips": "Check for missing quotes, commas, or brackets. Ensure your JSON is valid and does not exceed 50,000 characters."
                }
            )
        # Input sanitization
        payload = {k: v for k, v in payload.items() if v is not None}
        # Validate using Pydantic
        try:
            QuoteRequestModel(**payload)
        except ValidationError as exc:
            logger.error(f"Input validation error: {exc.errors()}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "error_type": "INPUT_VALIDATION_ERROR",
                    "error_message": "Input validation failed.",
                    "tips": "Check required fields, email format, ISO-8601 datetime, and line item structure. Ensure JSON is properly formatted."
                }
            )
        # Process request
        result = await agent.process_request(payload)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as exc:
        logger.error(f"Unhandled error in endpoint: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_type": "INTERNAL_SERVER_ERROR",
                "error_message": str(exc),
                "tips": "Unexpected error occurred. Please retry or contact support."
            }
        )

# -------------------- Main Execution Block --------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting CRM Quote Creation Agent...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
