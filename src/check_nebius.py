"""Check the local Nebius API configuration and connection."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_BASE_URL = (
    "https://api.tokenfactory.nebius.com/v1/"
)


def mask_secret(secret: str) -> str:
    """Return a safely masked representation of a secret."""
    if len(secret) <= 10:
        return "*" * len(secret)

    return (
        secret[:5]
        + "*" * max(len(secret) - 10, 5)
        + secret[-5:]
    )


def print_api_error(error: Exception) -> None:
    """Display a useful explanation for a Nebius API error."""
    print("\n" + "=" * 70)
    print("NEBIUS CONNECTION FAILED")
    print("=" * 70)

    print(f"\nError type: {type(error).__name__}")
    print(f"Error details: {error}")

    if isinstance(error, AuthenticationError):
        print(
            "\nLikely cause: The API key is invalid, expired, "
            "revoked, or copied incorrectly."
        )
        print(
            "Create a new Nebius API key and place it in .env."
        )

    elif isinstance(error, PermissionDeniedError):
        print(
            "\nLikely cause: The API key does not have access "
            "to this model or service."
        )

    elif isinstance(error, NotFoundError):
        print(
            "\nLikely cause: The configured model ID does not "
            "exist or is unavailable to this account."
        )

    elif isinstance(error, RateLimitError):
        print(
            "\nLikely cause: The account reached a usage, credit, "
            "or request-rate limit."
        )

    elif isinstance(error, APIConnectionError):
        print(
            "\nLikely cause: Python could not connect to Nebius. "
            "Check your internet connection, firewall, or VPN."
        )

    elif isinstance(error, APIStatusError):
        print(
            "\nNebius returned an API status error. Review the "
            "status code and message printed above."
        )

    else:
        print(
            "\nThe request failed before a valid response was returned."
        )


def check_nebius_connection() -> None:
    """Validate environment variables, available models, and chat access."""
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f"The .env file was not found at: {ENV_PATH}"
        )

    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True,
    )

    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    base_url = os.getenv(
        "NEBIUS_BASE_URL",
        DEFAULT_BASE_URL,
    ).strip()
    configured_model = os.getenv(
        "NEBIUS_MODEL",
        "",
    ).strip()

    print("=" * 70)
    print("NEBIUS CONFIGURATION CHECK")
    print("=" * 70)

    print(f"\n.env file: {ENV_PATH}")
    print(f"API key loaded: {bool(api_key)}")

    if api_key:
        print(f"Masked API key: {mask_secret(api_key)}")

    print(f"Base URL: {base_url}")
    print(
        f"Configured model: "
        f"{configured_model or 'NOT CONFIGURED'}"
    )

    if not api_key:
        raise EnvironmentError(
            "NEBIUS_API_KEY is missing or blank in .env."
        )

    if not configured_model:
        raise EnvironmentError(
            "NEBIUS_MODEL is missing or blank in .env."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=60.0,
    )

    try:
        print("\nRequesting the available model list...")

        model_response = client.models.list()

        available_models = sorted(
            model.id
            for model in model_response.data
        )

        print(
            f"Models available to this account: "
            f"{len(available_models)}"
        )

        configured_model_available = (
            configured_model in available_models
        )

        print(
            "Configured model available: "
            f"{configured_model_available}"
        )

        if not configured_model_available:
            print(
                "\nThe configured model was not found. "
                "Some available chat-model IDs are:"
            )

            likely_chat_models = [
                model_id
                for model_id in available_models
                if any(
                    keyword in model_id.lower()
                    for keyword in [
                        "qwen",
                        "llama",
                        "kimi",
                        "gpt",
                        "deepseek",
                    ]
                )
            ]

            models_to_display = (
                likely_chat_models[:25]
                if likely_chat_models
                else available_models[:25]
            )

            for model_id in models_to_display:
                print(f"  - {model_id}")

            print(
                "\nCopy one exact available model ID into "
                "NEBIUS_MODEL in your .env file."
            )

            return

        print("\nSending a small test chat request...")

        response = client.chat.completions.create(
            model=configured_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Respond using only the word READY."
                    ),
                },
                {
                    "role": "user",
                    "content": "Test the API connection.",
                },
            ],
            temperature=0,
            max_tokens=10,
        )

        if not response.choices:
            raise RuntimeError(
                "The API returned no completion choices."
            )

        content = response.choices[0].message.content

        print(f"API response: {content}")
        print("\n" + "=" * 70)
        print("NEBIUS CONNECTION SUCCESSFUL")
        print("=" * 70)

    except (
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        RateLimitError,
        APIConnectionError,
        APIStatusError,
    ) as error:
        print_api_error(error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    try:
        check_nebius_connection()
    except (
        FileNotFoundError,
        EnvironmentError,
    ) as error:
        print("\n" + "=" * 70)
        print("LOCAL CONFIGURATION ERROR")
        print("=" * 70)
        print(f"\n{error}")
        raise SystemExit(1) from error