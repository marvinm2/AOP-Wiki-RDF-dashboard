"""Configuration management for the AOP-Wiki RDF Dashboard."""
import os
from typing import Dict, Any

class Config:
    """Application configuration."""
    
    # SPARQL Configuration
    SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "http://localhost:8890/sparql")
    SPARQL_TIMEOUT = int(os.getenv("SPARQL_TIMEOUT", "30"))
    SPARQL_MAX_RETRIES = int(os.getenv("SPARQL_MAX_RETRIES", "3"))
    SPARQL_RETRY_DELAY = int(os.getenv("SPARQL_RETRY_DELAY", "2"))
    
    # Performance Configuration
    PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "5"))
    PLOT_TIMEOUT = int(os.getenv("PLOT_TIMEOUT", "60"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Flask Configuration
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    # Feature Flags
    ENABLE_HEALTH_CHECK = os.getenv("ENABLE_HEALTH_CHECK", "True").lower() == "true"
    ENABLE_PERFORMANCE_LOGGING = os.getenv("ENABLE_PERFORMANCE_LOGGING", "True").lower() == "true"
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        try:
            # Validate SPARQL endpoint URL
            from urllib.parse import urlparse
            parsed_url = urlparse(cls.SPARQL_ENDPOINT)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid SPARQL endpoint URL: {cls.SPARQL_ENDPOINT}")
            
            # Validate numeric settings
            if cls.SPARQL_TIMEOUT <= 0:
                raise ValueError("SPARQL_TIMEOUT must be positive")
            if cls.SPARQL_MAX_RETRIES <= 0:
                raise ValueError("SPARQL_MAX_RETRIES must be positive")
            if cls.PARALLEL_WORKERS <= 0:
                raise ValueError("PARALLEL_WORKERS must be positive")
            if cls.FLASK_PORT <= 0 or cls.FLASK_PORT > 65535:
                raise ValueError("FLASK_PORT must be between 1 and 65535")
                
            return True
            
        except Exception as e:
            import logging
            logging.error(f"Configuration validation failed: {str(e)}")
            return False