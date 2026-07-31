"""Streamlit application for Adult Income prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow Streamlit to import modules from the project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_parser import (  # noqa: E402
    build_model_input,
    create_nebius_client,
    parse_user_input,
)


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_model_metadata.json"
)
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult.csv"


FIELD_LABELS = {
    "age": "age",
    "workclass": "type of employer or workclass",
    "education": "education level",
    "marital_status": "marital status",
    "occupation": "occupation",
    "relationship": "household relationship",
    "race": "race category",
    "sex": "sex category",
    "hours_per_week": "weekly working hours",
    "native_country": "country of origin",
}


EXPLANATION_SYSTEM_PROMPT = """
You explain predictions from an educational Adult Income
classification model.

The model estimates whether a record resembles the historical
income category <=50K or >50K.

Rules:
1. Clearly state the predicted category and estimated probability.
2. Explain that this is a statistical estimate, not a known fact.
3. Do not claim that any feature caused the prediction.
4. Do not provide employment, financial, legal, credit, insurance,
   immigration, or eligibility advice.
5. Mention that the model uses historical census information and
   may contain bias or limitations.
6. Keep the response between 80 and 140 words.
7. Use plain English.
""".strip()


@st.cache_resource
def load_trained_model() -> Any:
    """Load and cache the trained scikit-learn pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run "
            "'python -m src.train' before starting the app."
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise TypeError(
            "The saved object does not support predictions."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The saved model does not support probabilities."
        )

    return model


@st.cache_data
def load_fnlwgt_default() -> int:
    """
    Load the median Census sampling weight.

    fnlwgt is a Census sampling-weight field and is not practical
    for an application user to provide.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "The Adult Income dataset was not found. Run "
            "'python -m src.data_loader' before starting the app."
        )

    dataframe = pd.read_csv(
        DATA_PATH,
        usecols=["fnlwgt"],
    )

    weights = pd.to_numeric(
        dataframe["fnlwgt"],
        errors="coerce",
    )

    median_weight = weights.median()

    if pd.isna(median_weight):
        raise ValueError(
            "A valid median fnlwgt value could not be calculated."
        )

    return int(median_weight)


@st.cache_data
def load_model_metadata() -> dict[str, Any]:
    """Load information about the selected model."""
    if not METADATA_PATH.exists():
        return {}

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(metadata_file)

    if not isinstance(metadata, dict):
        return {}

    return metadata


def format_missing_fields(
    missing_fields: list[str],
) -> str:
    """Convert model feature names into readable labels."""
    readable_fields = [
        FIELD_LABELS.get(
            field,
            field.replace("_", " "),
        )
        for field in missing_fields
    ]

    if not readable_fields:
        return ""

    if len(readable_fields) == 1:
        return readable_fields[0]

    return (
        ", ".join(readable_fields[:-1])
        + f", and {readable_fields[-1]}"
    )


def create_fallback_explanation(
    predicted_class: int,
    probability_above_50k: float,
) -> str:
    """Create an explanation if the LLM service is unavailable."""
    predicted_label = (
        "> $50K"
        if predicted_class == 1
        else "<= $50K"
    )

    return (
        f"The model predicts the {predicted_label} income category. "
        f"It estimates a {probability_above_50k:.1%} probability "
        f"that the profile belongs to the above-$50K category. "
        f"This is a statistical estimate based on historical Census "
        f"patterns, not a confirmed fact about an individual. The "
        f"result may reflect limitations or bias in the historical "
        f"data and should not be used for employment, credit, "
        f"insurance, immigration, or eligibility decisions."
    )


def generate_prediction_explanation(
    features: dict[str, Any],
    predicted_class: int,
    probability_above_50k: float,
    client: Any | None = None,
    model_name: str | None = None,
) -> str:
    """Use Nebius to explain the trained model's prediction."""
    if client is None:
        client, configured_model = create_nebius_client()
        model_name = model_name or configured_model

    if not model_name:
        raise ValueError(
            "An LLM model name must be configured."
        )

    predicted_label = (
        "> $50K"
        if predicted_class == 1
        else "<= $50K"
    )

    explanation_context = {
        "predicted_category": predicted_label,
        "probability_above_50k": round(
            probability_above_50k,
            4,
        ),
        "provided_features": features,
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": EXPLANATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Explain this prediction:\n"
                    + json.dumps(
                        explanation_context,
                        indent=2,
                    )
                ),
            },
        ],
        temperature=0.2,
        max_tokens=250,
    )

    if not response.choices:
        raise RuntimeError(
            "The explanation response contained no choices."
        )

    explanation = response.choices[0].message.content

    if not explanation or not explanation.strip():
        raise RuntimeError(
            "The explanation response was empty."
        )

    return explanation.strip()


def make_prediction(
    model: Any,
    parsed_features: dict[str, Any],
    fnlwgt_default: int,
) -> dict[str, Any]:
    """Build the model input and produce a prediction."""
    model_input = build_model_input(
        features=parsed_features,
        fnlwgt_default=fnlwgt_default,
    )

    predicted_class = int(
        model.predict(model_input)[0]
    )

    probabilities = model.predict_proba(
        model_input
    )[0]

    probability_below_50k = float(
        probabilities[0]
    )

    probability_above_50k = float(
        probabilities[1]
    )

    predicted_label = (
        "> $50K"
        if predicted_class == 1
        else "<= $50K"
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probability_below_50k": probability_below_50k,
        "probability_above_50k": probability_above_50k,
        "model_input": model_input,
    }


def display_model_information(
    metadata: dict[str, Any],
) -> None:
    """Display selected-model information in the sidebar."""
    st.sidebar.header("Model information")

    run_name = metadata.get(
        "run_name",
        "Not available",
    )

    model_type = metadata.get(
        "model_type",
        "Not available",
    )

    roc_auc = metadata.get("roc_auc")
    accuracy = metadata.get("accuracy")
    f1_score = metadata.get("f1")

    st.sidebar.write(
        f"**Best run:** {run_name}"
    )

    st.sidebar.write(
        f"**Model:** {model_type}"
    )

    if isinstance(roc_auc, (int, float)):
        st.sidebar.write(
            f"**Test ROC-AUC:** {roc_auc:.4f}"
        )

    if isinstance(accuracy, (int, float)):
        st.sidebar.write(
            f"**Test accuracy:** {accuracy:.4f}"
        )

    if isinstance(f1_score, (int, float)):
        st.sidebar.write(
            f"**Test F1:** {f1_score:.4f}"
        )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Adult Income Prediction Assistant",
        page_icon="📊",
        layout="centered",
    )

    st.title(
        "Adult Income Prediction Assistant"
    )

    st.write(
        "Describe a fictional or sample profile in plain English. "
        "The application extracts the required information, runs "
        "the trained machine-learning model, and explains the result."
    )

    st.warning(
        "Educational demonstration only. Do not use this application "
        "for hiring, lending, insurance, immigration, benefits, or "
        "other consequential decisions."
    )

    try:
        trained_model = load_trained_model()
        fnlwgt_default = load_fnlwgt_default()
        metadata = load_model_metadata()
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
    ) as error:
        st.error(str(error))
        st.stop()

    display_model_information(metadata)

    example_input = (
        "I am 42 years old, work for a private company, have a "
        "bachelor's degree, and work in management. I am married, "
        "listed as a husband in my household, White, male, work "
        "45 hours per week, and I am from the United States. "
        "I have no capital gains or capital losses."
    )

    with st.form(
        key="income_prediction_form"
    ):
        user_text = st.text_area(
            "Describe the profile",
            value="",
            height=180,
            placeholder=example_input,
        )

        submitted = st.form_submit_button(
            "Analyze profile",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        with st.expander(
            "View an example request"
        ):
            st.write(example_input)

        return

    if not user_text.strip():
        st.error(
            "Enter a profile description before requesting "
            "a prediction."
        )
        return

    try:
        with st.spinner(
            "Extracting information from the description..."
        ):
            parsing_result = parse_user_input(
                user_text=user_text
            )
    except (
        EnvironmentError,
        RuntimeError,
        ValueError,
    ) as error:
        st.error(
            f"The LLM parser could not process the request: {error}"
        )
        return

    status = parsing_result["status"]
    parsed_features = parsing_result["features"]
    missing_fields = parsing_result["missing_fields"]
    parsing_errors = parsing_result["errors"]

    with st.expander(
        "Extracted information",
        expanded=True,
    ):
        if parsed_features:
            extracted_table = pd.DataFrame(
                {
                    "Feature": list(
                        parsed_features.keys()
                    ),
                    "Extracted value": [
                        parsed_features[feature]
                        for feature in parsed_features
                    ],
                }
            )

            st.dataframe(
                extracted_table,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write(
                "No usable features were extracted."
            )

    if status == "invalid":
        st.error(
            "The request contains invalid or unsupported values."
        )

        for error in parsing_errors:
            st.write(f"- {error}")

        return

    if status == "incomplete":
        readable_missing_fields = format_missing_fields(
            missing_fields
        )

        st.warning(
            "More information is required before the model can "
            f"make a prediction. Please provide: "
            f"{readable_missing_fields}."
        )

        return

    try:
        prediction_result = make_prediction(
            model=trained_model,
            parsed_features=parsed_features,
            fnlwgt_default=fnlwgt_default,
        )
    except (TypeError, ValueError) as error:
        st.error(
            f"The prediction could not be generated: {error}"
        )
        return

    predicted_label = prediction_result[
        "predicted_label"
    ]

    probability_above_50k = prediction_result[
        "probability_above_50k"
    ]

    probability_below_50k = prediction_result[
        "probability_below_50k"
    ]

    st.subheader("Model prediction")

    first_column, second_column = st.columns(2)

    with first_column:
        st.metric(
            label="Predicted category",
            value=predicted_label,
        )

    with second_column:
        st.metric(
            label="Probability of > $50K",
            value=f"{probability_above_50k:.1%}",
        )

    probability_table = pd.DataFrame(
        {
            "Income category": [
                "<= $50K",
                "> $50K",
            ],
            "Estimated probability": [
                probability_below_50k,
                probability_above_50k,
            ],
        }
    )

    st.dataframe(
        probability_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("LLM explanation")

    try:
        with st.spinner(
            "Generating an explanation..."
        ):
            explanation = generate_prediction_explanation(
                features=parsed_features,
                predicted_class=prediction_result[
                    "predicted_class"
                ],
                probability_above_50k=(
                    probability_above_50k
                ),
            )
    except (
        EnvironmentError,
        RuntimeError,
        ValueError,
    ):
        explanation = create_fallback_explanation(
            predicted_class=prediction_result[
                "predicted_class"
            ],
            probability_above_50k=(
                probability_above_50k
            ),
        )

        st.info(
            "The LLM explanation service was unavailable, so the "
            "application generated a standard explanation."
        )

    st.write(explanation)

    with st.expander(
        "Technical model input"
    ):
        st.dataframe(
            prediction_result["model_input"],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "This project uses historical Census data. Predictions may "
        "reflect limitations, outdated patterns, or bias in the data. "
        "The result describes model behavior and is not a verified "
        "statement about any person."
    )


if __name__ == "__main__":
    main()