mport logging
import json
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langsmith import traceable
from dotenv import load_dotenv

# === Structured Logging ===

#inheriting predefined class logging.Formatter
class JSONFormatter(logging.Formatter):
    """Format logs as JSON for log aggregation."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,#level- info, warning, error
            "message": record.getMessage(),
            "module": record.module,#file
            "function": record.funcName,#function
        }

        #if extra data passed eg- user_id, session_id etc
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging():
    """Setup structured JSON logging."""
    #create a logger named langgraph_app or use existing logger with this name
    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)
    
    #writes to console FileHandler would have written to file
    handler = logging.StreamHandler()
    #this tells it dont print as text use this formatter
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger