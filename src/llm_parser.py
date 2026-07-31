"""LLM-powered input parsing for the Adult Income application."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_NEBIUS_BASE_URL = (
    "https://api.tokenfactory.nebius.com/v1/"
)

USER_FEATURES = [
    "age",
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]

REQUIRED_USER_FEATURES = [
    "age",
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hours_per_week",
    "native_country",
]

MODEL_FEATURES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]

EDUCATION_TO_NUM = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}

VALID_CATEGORIES = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": list(EDUCATION_TO_NUM.keys()),
    "marital_status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    ],
    "sex": [
        "Female",
        "Male",
    ],
}

CATEGORY_ALIASES = {
    "private-sector": "Private",
    "private sector": "Private",
    "self-employed": "Self-emp-not-inc",
    "self employed": "Self-emp-not-inc",
    "federal government": "Federal-gov",
    "local government": "Local-gov",
    "state government": "State-gov",
    "high school": "HS-grad",
    "high-school": "HS-grad",
    "high school graduate": "HS-grad",
    "bachelor": "Bachelors",
    "bachelor's": "Bachelors",
    "bachelors degree": "Bachelors",
    "master": "Masters",
    "master's": "Masters",
    "masters degree": "Masters",
    "doctorate degree": "Doctorate",
    "never married": "Never-married",
    "married": "Married-civ-spouse",
    "executive": "Exec-managerial",
    "management": "Exec-managerial",
    "professional": "Prof-specialty",
    "administrative": "Adm-clerical",
    "tech support": "Tech-support",
    "not in family": "Not-in-family",
    "asian": "Asian-Pac-Islander",
    "native american": "Amer-Indian-Eskimo",
    "man": "Male",
    "woman": "Female",
}

SYSTEM_PROMPT = """
You extract structured information for an Adult Income prediction model.

Return one JSON object only. Do not include Markdown or explanations.

Use this exact structure:

{
  "features": {
    "age": null,
    "workclass": null,
    "education": null,
    "marital_status": null,
    "occupation": null,
    "relationship": null,
    "race": null,
    "sex": null,
    "capital_gain": null,
    "capital_loss": null,
    "hours_per_week": null,
    "native_country": null
  }
}

Rules:

1. Use null when information was not supplied.
2. Do not guess race, sex, marital status, occupation, or country.
3. Do not infer sensitive characteristics from names or other clues.
4. age, capital_gain, capital_loss, and hours_per_week must be numbers.
5. If the user says they have no capital gains or losses, use 0.
6. Convert United States, USA, or US to United-States.
7. Prefer the canonical Adult dataset category names below.

Workclass:
Private, Self-emp-not-inc, Self-emp-inc, Federal-gov,
Local-gov, State-gov, Without-pay, Never-worked

Education:
Preschool, 1st-4th, 5th-6th, 7th-8th, 9th, 10th,
11th, 12th, HS-grad, Some-college, Assoc-voc,
Assoc-acdm, Bachelors, Masters, Prof-school, Doctorate

Marital status:
Married-civ-spouse, Divorced, Never-married, Separated,
Widowed, Married-spouse-absent, Married-AF-spouse

Occupation:
Tech-support, Craft-repair, Other-service, Sales,
Exec-managerial, Prof-specialty, Handlers-cleaners,
Machine-op-inspct, Adm-clerical, Farming-fishing,
Transport-moving, Priv-house-serv, Protective-serv,
Armed-Forces

Relationship:
Wife, Own-child, Husband, Not-in-family,
Other-relative, Unmarried

Race:
White, Asian-Pac-Islander, Amer-Indian-Eskimo,
Other, Black

Sex:
Female, Male
""".strip()


def create_nebius_client() -> tuple[OpenAI, str]:
    """Create the Nebius client using environment variables."""
    load_dotenv(ENV_PATH)

    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv(
        "NEBIUS_BASE_URL",
        DEFAULT_NEBIUS_BASE_URL,
    )
    model = os.getenv("NEBIUS_MODEL")

    if not api_key:
        raise EnvironmentError(
            "NEBIUS_API_KEY is missing. Add it to the local .env file."
        )

    if not model:
        raise EnvironmentError(
            "NEBIUS_MODEL is missing. Add it to the local .env file."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    return client, model


def extract_json_object(response_text: str) -> dict[str, Any]:
    """Extract and decode a JSON object from an LLM response."""
    if not isinstance(response_text, str):
        raise TypeError("The LLM response must be text.")

    cleaned_text = response_text.strip()

    cleaned_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    )

    cleaned_text = re.sub(
        r"\s*```$",
        "",
        cleaned_text,
    )

    start_position = cleaned_text.find("{")
    end_position = cleaned_text.rfind("}")

    if start_position == -1 or end_position == -1:
        raise ValueError(
            "The LLM response did not contain a JSON object."
        )

    json_text = cleaned_text[
        start_position:end_position + 1
    ]

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The LLM returned invalid JSON."
        ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            "The decoded LLM response must be a JSON object."
        )

    return parsed


def category_key(value: str) -> str:
    """Create a normalized key for category comparison."""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.lower(),
    )


def canonicalize_category(
    field_name: str,
    value: object,
) -> str | None:
    """Convert a category or common alias into its canonical value."""
    if value is None:
        return None

    if not isinstance(value, str):
        return None

    cleaned_value = value.strip()

    if not cleaned_value:
        return None

    alias_value = CATEGORY_ALIASES.get(
        cleaned_value.lower()
    )

    if alias_value is not None:
        cleaned_value = alias_value

    valid_values = VALID_CATEGORIES[field_name]

    category_lookup = {
        category_key(category): category
        for category in valid_values
    }

    return category_lookup.get(
        category_key(cleaned_value)
    )


def normalize_native_country(
    value: object,
) -> str | None:
    """Normalize the country value for the Adult dataset."""
    if value is None or not isinstance(value, str):
        return None

    cleaned_value = value.strip()

    if not cleaned_value:
        return None

    country_aliases = {
        "us": "United-States",
        "u.s.": "United-States",
        "usa": "United-States",
        "u.s.a.": "United-States",
        "united states": "United-States",
        "united-states": "United-States",
    }

    alias = country_aliases.get(
        cleaned_value.lower()
    )

    if alias is not None:
        return alias

    return re.sub(
        r"\s+",
        "-",
        cleaned_value,
    )


def normalize_integer(
    field_name: str,
    value: object,
    minimum: int,
    maximum: int,
    errors: list[str],
) -> int | None:
    """Convert and validate one integer feature."""
    if value is None or value == "":
        return None

    try:
        numeric_value = int(float(value))
    except (TypeError, ValueError):
        errors.append(
            f"{field_name} must be a number."
        )
        return None

    if not minimum <= numeric_value <= maximum:
        errors.append(
            f"{field_name} must be between "
            f"{minimum} and {maximum}."
        )
        return None

    return numeric_value


def validate_parsed_features(
    parsed_response: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate, normalize, and classify the parsed LLM output.

    Returns a dictionary with:
    - status: complete, incomplete, or invalid
    - features: normalized user features
    - missing_fields: required values not supplied
    - errors: invalid values
    """
    raw_features = parsed_response.get(
        "features",
        parsed_response,
    )

    if not isinstance(raw_features, dict):
        return {
            "status": "invalid",
            "features": {},
            "missing_fields": REQUIRED_USER_FEATURES.copy(),
            "errors": [
                "The parsed features must be a JSON object."
            ],
        }

    errors: list[str] = []

    normalized_features: dict[str, Any] = {
        "age": normalize_integer(
            field_name="age",
            value=raw_features.get("age"),
            minimum=17,
            maximum=100,
            errors=errors,
        ),
        "workclass": None,
        "education": None,
        "marital_status": None,
        "occupation": None,
        "relationship": None,
        "race": None,
        "sex": None,
        "capital_gain": normalize_integer(
            field_name="capital_gain",
            value=raw_features.get("capital_gain", 0),
            minimum=0,
            maximum=99999,
            errors=errors,
        ),
        "capital_loss": normalize_integer(
            field_name="capital_loss",
            value=raw_features.get("capital_loss", 0),
            minimum=0,
            maximum=99999,
            errors=errors,
        ),
        "hours_per_week": normalize_integer(
            field_name="hours_per_week",
            value=raw_features.get("hours_per_week"),
            minimum=1,
            maximum=99,
            errors=errors,
        ),
        "native_country": normalize_native_country(
            raw_features.get("native_country")
        ),
    }

    category_fields = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
    ]

    for field_name in category_fields:
        raw_value = raw_features.get(field_name)

        if raw_value is None or raw_value == "":
            normalized_features[field_name] = None
            continue

        canonical_value = canonicalize_category(
            field_name=field_name,
            value=raw_value,
        )

        if canonical_value is None:
            errors.append(
                f"{field_name} contains an unsupported value: "
                f"{raw_value}"
            )

        normalized_features[field_name] = canonical_value

    if normalized_features["capital_gain"] is None:
        normalized_features["capital_gain"] = 0

    if normalized_features["capital_loss"] is None:
        normalized_features["capital_loss"] = 0

    missing_fields = [
        field_name
        for field_name in REQUIRED_USER_FEATURES
        if normalized_features.get(field_name) is None
    ]

    if errors:
        status = "invalid"
    elif missing_fields:
        status = "incomplete"
    else:
        status = "complete"

    return {
        "status": status,
        "features": normalized_features,
        "missing_fields": missing_fields,
        "errors": errors,
    }


def parse_user_input(
    user_text: str,
    client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Use an LLM to extract Adult Income features from natural language."""
    if not isinstance(user_text, str) or not user_text.strip():
        return {
            "status": "invalid",
            "features": {},
            "missing_fields": REQUIRED_USER_FEATURES.copy(),
            "errors": [
                "Please provide a description before requesting "
                "a prediction."
            ],
        }

    if client is None:
        client, configured_model = create_nebius_client()
        model = model or configured_model

    if not model:
        raise ValueError(
            "An LLM model name must be supplied."
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_text.strip(),
                },
            ],
        )
    except Exception as error:
        raise RuntimeError(
            "The LLM request failed."
        ) from error

    if not response.choices:
        raise RuntimeError(
            "The LLM response contained no choices."
        )

    response_text = response.choices[0].message.content

    if not response_text:
        raise RuntimeError(
            "The LLM returned an empty response."
        )

    parsed_response = extract_json_object(
        response_text
    )

    return validate_parsed_features(
        parsed_response
    )


def build_model_input(
    features: dict[str, Any],
    fnlwgt_default: int,
) -> pd.DataFrame:
    """
    Convert validated user features into the 14-column model input.

    The census sampling-weight feature, fnlwgt, is not meaningful
    for a normal user to provide, so the application supplies a
    training-data median.
    """
    missing_features = [
        feature_name
        for feature_name in REQUIRED_USER_FEATURES
        if features.get(feature_name) is None
    ]

    if missing_features:
        raise ValueError(
            "Cannot create model input because these fields "
            f"are missing: {missing_features}"
        )

    education = features["education"]

    if education not in EDUCATION_TO_NUM:
        raise ValueError(
            f"Unsupported education category: {education}"
        )

    model_row = {
        "age": int(features["age"]),
        "workclass": features["workclass"],
        "fnlwgt": int(fnlwgt_default),
        "education": education,
        "education_num": EDUCATION_TO_NUM[education],
        "marital_status": features["marital_status"],
        "occupation": features["occupation"],
        "relationship": features["relationship"],
        "race": features["race"],
        "sex": features["sex"],
        "capital_gain": int(
            features.get("capital_gain", 0)
        ),
        "capital_loss": int(
            features.get("capital_loss", 0)
        ),
        "hours_per_week": int(
            features["hours_per_week"]
        ),
        "native_country": features["native_country"],
    }

    return pd.DataFrame(
        [model_row],
        columns=MODEL_FEATURES,
    )