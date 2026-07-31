"""Tests for the Adult Income LLM input parser."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.llm_parser import (
    MODEL_FEATURES,
    build_model_input,
    parse_user_input,
)


class FakeCompletions:
    """Return a controlled LLM response for unit testing."""

    def __init__(
        self,
        response_content: str,
    ) -> None:
        self.response_content = response_content

    def create(
        self,
        **_: Any,
    ) -> SimpleNamespace:
        """Return a response shaped like an OpenAI chat response."""
        message = SimpleNamespace(
            content=self.response_content
        )

        choice = SimpleNamespace(
            message=message
        )

        return SimpleNamespace(
            choices=[choice]
        )


class FakeChat:
    """Provide the fake completions endpoint."""

    def __init__(
        self,
        response_content: str,
    ) -> None:
        self.completions = FakeCompletions(
            response_content
        )


class FakeClient:
    """Provide a fake OpenAI-compatible client."""

    def __init__(
        self,
        response_payload: dict[str, Any],
    ) -> None:
        response_content = json.dumps(
            response_payload
        )

        self.chat = FakeChat(
            response_content
        )


def complete_feature_payload() -> dict[str, Any]:
    """Return a complete valid parser response."""
    return {
        "features": {
            "age": 42,
            "workclass": "Private",
            "education": "Bachelors",
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 45,
            "native_country": "United-States",
        }
    }


def test_parser_extracts_complete_features() -> None:
    """The parser should return normalized complete features."""
    fake_client = FakeClient(
        complete_feature_payload()
    )

    result = parse_user_input(
        user_text=(
            "I am a 42-year-old married White man with a "
            "bachelor's degree. I work in management for a "
            "private company, work 45 hours per week, was born "
            "in the United States, and have no capital gains "
            "or losses."
        ),
        client=fake_client,
        model="test-model",
    )

    assert result["status"] == "complete"
    assert result["missing_fields"] == []
    assert result["errors"] == []

    assert result["features"]["age"] == 42
    assert result["features"]["education"] == "Bachelors"
    assert (
        result["features"]["occupation"]
        == "Exec-managerial"
    )
    assert result["features"]["hours_per_week"] == 45
    assert (
        result["features"]["native_country"]
        == "United-States"
    )


def test_parser_handles_incomplete_input() -> None:
    """Missing required features should trigger incomplete status."""
    fake_client = FakeClient(
        {
            "features": {
                "age": 35,
                "education": "Bachelors",
                "hours_per_week": 40,
            }
        }
    )

    result = parse_user_input(
        user_text=(
            "I am 35, have a bachelor's degree, "
            "and work 40 hours each week."
        ),
        client=fake_client,
        model="test-model",
    )

    assert result["status"] == "incomplete"
    assert "workclass" in result["missing_fields"]
    assert "occupation" in result["missing_fields"]
    assert "marital_status" in result["missing_fields"]
    assert "race" in result["missing_fields"]
    assert "sex" in result["missing_fields"]


def test_parser_rejects_invalid_numeric_values() -> None:
    """Invalid numeric ranges should produce useful errors."""
    payload = complete_feature_payload()
    payload["features"]["age"] = 140
    payload["features"]["hours_per_week"] = 150

    fake_client = FakeClient(payload)

    result = parse_user_input(
        user_text=(
            "I am 140 years old and work "
            "150 hours every week."
        ),
        client=fake_client,
        model="test-model",
    )

    assert result["status"] == "invalid"
    assert result["features"]["age"] is None
    assert result["features"]["hours_per_week"] is None

    assert any(
        "age must be between" in error
        for error in result["errors"]
    )

    assert any(
        "hours_per_week must be between" in error
        for error in result["errors"]
    )


def test_model_input_contains_all_required_columns() -> None:
    """Validated features should produce the expected model row."""
    features = complete_feature_payload()["features"]

    model_input = build_model_input(
        features=features,
        fnlwgt_default=180000,
    )

    assert model_input.shape == (1, 14)
    assert model_input.columns.tolist() == MODEL_FEATURES

    assert model_input.loc[0, "fnlwgt"] == 180000
    assert model_input.loc[0, "education_num"] == 13
    assert model_input.loc[0, "capital_gain"] == 0
    assert model_input.loc[0, "capital_loss"] == 0