"""
Configuration management for CLLM.

Loads configuration from environment variables or .env file.
"""

import os
from pathlib import Path


def load_env_file():
    """Load .env file from current directory, parent directories, or package directory."""

    def load_from_path(env_file: Path) -> bool:
        """Load environment variables from a file. Returns True if loaded."""
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value
            return True
        return False

    # First check the package directory (where this config.py file is)
    package_dir = Path(__file__).parent
    if load_from_path(package_dir.parent / ".env"):
        return

    # Then check current working directory and up to 3 parent directories
    current = Path.cwd()
    for _ in range(4):
        if load_from_path(current / ".env"):
            return

        # Move to parent directory
        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent


# Load .env file on import
load_env_file()


class Config:
    """Configuration for CLLM tool."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # LLM provider selection
        self.llm_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

        # Anthropic configuration
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.anthropic_model = os.getenv(
            "ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"
        )

        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")

        # Shared configuration
        self.timeout = float(os.getenv("LLM_TIMEOUT", "600.0"))

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required configuration is missing
        """
        if self.llm_provider not in ["anthropic", "openai"]:
            raise ValueError(
                f"LLM_PROVIDER must be 'anthropic' or 'openai', got: {self.llm_provider}"
            )

        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required when using Anthropic. "
                "Please set it with: export ANTHROPIC_API_KEY=your_api_key"
            )

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when using OpenAI. "
                "Please set it with: export OPENAI_API_KEY=your_api_key"
            )


# Global configuration instance
config = Config()
