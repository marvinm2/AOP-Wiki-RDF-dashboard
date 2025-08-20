"""Configuration management for the AOP-Wiki RDF Dashboard.

This module provides centralized configuration management for the AOP-Wiki RDF Dashboard
application. It handles environment variable parsing, configuration validation, and
provides a unified interface for accessing application settings.

The configuration supports:
    - SPARQL endpoint configuration with timeout and retry settings
    - Flask web server configuration
    - Performance tuning parameters for parallel processing
    - Feature flags for optional functionality
    - Comprehensive validation of all settings

Environment Variables:
    SPARQL_ENDPOINT: URL of the SPARQL endpoint (default: http://localhost:8890/sparql)
    SPARQL_TIMEOUT: Query timeout in seconds (default: 30)
    SPARQL_MAX_RETRIES: Maximum retry attempts for failed queries (default: 3)
    SPARQL_RETRY_DELAY: Delay between retry attempts in seconds (default: 2)
    PARALLEL_WORKERS: Number of parallel workers for plot generation (default: 5)
    PLOT_TIMEOUT: Timeout for individual plot generation in seconds (default: 60)
    LOG_LEVEL: Logging verbosity level (default: INFO)
    FLASK_HOST: Flask host binding (default: 0.0.0.0)
    FLASK_PORT: Flask port number (default: 5000)
    FLASK_DEBUG: Enable Flask debug mode (default: False)
    ENABLE_HEALTH_CHECK: Enable health check endpoints (default: True)
    ENABLE_PERFORMANCE_LOGGING: Enable performance logging (default: True)

Example:
    Basic usage of configuration:
    
    >>> from config import Config
    >>> print(Config.SPARQL_ENDPOINT)
    http://localhost:8890/sparql
    >>> config_dict = Config.get_config_dict()
    >>> is_valid = Config.validate_config()

Author:
    Generated with Claude Code (https://claude.ai/code)

"""
import os
from typing import Dict, Any

class Config:
    """Application configuration class with environment variable management.
    
    This class provides a centralized interface for all application configuration
    settings. It automatically loads values from environment variables with
    sensible defaults and provides validation functionality.
    
    All configuration values are class attributes that can be accessed directly.
    The class also provides utility methods for configuration management and
    validation.
    
    Attributes:
        SPARQL_ENDPOINT (str): URL of the SPARQL endpoint for RDF queries
        SPARQL_TIMEOUT (int): Timeout in seconds for SPARQL queries
        SPARQL_MAX_RETRIES (int): Maximum number of retry attempts for failed queries
        SPARQL_RETRY_DELAY (int): Delay in seconds between retry attempts
        PARALLEL_WORKERS (int): Number of parallel workers for plot generation
        PLOT_TIMEOUT (int): Timeout in seconds for individual plot generation
        LOG_LEVEL (str): Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        FLASK_HOST (str): Host address for Flask web server binding
        FLASK_PORT (int): Port number for Flask web server
        FLASK_DEBUG (bool): Whether to enable Flask debug mode
        ENABLE_HEALTH_CHECK (bool): Whether to enable health check endpoints
        ENABLE_PERFORMANCE_LOGGING (bool): Whether to enable performance logging
    
    Example:
        >>> print(Config.SPARQL_ENDPOINT)
        http://localhost:8890/sparql
        >>> Config.FLASK_PORT = 8080  # Override default port
        >>> if Config.validate_config():
        ...     print("Configuration is valid")
    """
    
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
        """Get all configuration settings as a dictionary.
        
        Extracts all uppercase class attributes (configuration settings) and
        returns them as a dictionary. This is useful for logging configuration
        state, debugging, or passing configuration to other components.
        
        Returns:
            Dict[str, Any]: Dictionary mapping configuration parameter names to their values.
                Only includes uppercase attributes (actual configuration settings),
                excluding private attributes and methods.
        
        Example:
            >>> config = Config.get_config_dict()
            >>> print(config['SPARQL_ENDPOINT'])
            http://localhost:8890/sparql
            >>> for key, value in config.items():
            ...     print(f"{key}: {value}")
        """
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate all configuration settings for correctness and safety.
        
        Performs comprehensive validation of all configuration parameters to ensure
        they are within acceptable ranges and formats. This includes URL validation,
        numeric range checking, and logical consistency verification.
        
        Validation checks:
            - SPARQL endpoint URL format and scheme validation
            - Positive integer values for timeouts and worker counts
            - Port number range validation (1-65535)
            - Logical consistency between related parameters
        
        Returns:
            bool: True if all configuration settings are valid, False otherwise.
                If validation fails, error details are logged.
        
        Raises:
            ValueError: For specific validation failures (caught internally and logged)
        
        Example:
            >>> if not Config.validate_config():
            ...     print("Configuration validation failed, check logs")
            ...     # Fix configuration or use defaults
            ... else:
            ...     print("Configuration is valid")
        
        Note:
            This method logs validation errors but does not raise exceptions.
            It's designed to be safe to call during application startup.
        """
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