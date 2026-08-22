"""Secure logging utility with PII masking to prevent leakage of identity data in production logs."""

import logging
import re
import sys
from app.config import settings

# Sensitive patterns to mask (passport numbers, national IDs, dates of birth, full MRZ lines)
PII_PATTERNS = [
    (re.compile(r'([A-Z0-9]{8,12})', re.IGNORECASE), r'***ID***'),
    (re.compile(r'\b\d{2}[01]\d[0-3]\d\b'), r'***DOB***'),
    (re.compile(r'[P|V|I]<[A-Z0-9<]{30,44}', re.IGNORECASE), r'***MRZ_LINE***'),
]


class PIIMaskingFilter(logging.Filter):
    """Filter that sanitizes log records to mask PII."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        if not settings.MASK_PII_LOGS:
            return True
            
        if isinstance(record.msg, str):
            sanitized = record.msg
            for pattern, replacement in PII_PATTERNS:
                sanitized = pattern.sub(replacement, sanitized)
            record.msg = sanitized
            
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._sanitize_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._sanitize_value(a) for a in record.args)
                
        return True

    def _sanitize_value(self, val):
        if isinstance(val, str):
            for pattern, replacement in PII_PATTERNS:
                val = pattern.sub(replacement, val)
        return val


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a logger instance with PII masking."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        handler.addFilter(PIIMaskingFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)
    return logger
