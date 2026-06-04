from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ARTIFACT_DIR = APP_DIR / "artifacts"
SAMPLE_PREDICTION_PATH = DATA_DIR / "sample_bank_marketing_prediction.csv"
MODEL_PATH = ARTIFACT_DIR / "bank_logistic_regression_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "bank_logistic_regression_metadata.json"

st.set_page_config(page_title="Logistic Regression | Bank Marketing", layout="wide")

st.title("Logistic Regression — Bank Marketing Prediction")



@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_prediction_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PREDICTION_PATH)


def _coerce_numeric_like_columns(df: pd.DataFrame, min_ratio: float = 0.8) -> pd.DataFrame:
    """Convert numeric-looking text columns into numeric dtype when conversion is reliable."""
    converted = df.copy()

    for col in converted.columns:
        series = converted[col]
        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        s = series.astype("string").str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA, "NA": pd.NA})
        non_missing = s.notna()
        if int(non_missing.sum()) == 0:
            continue

        numeric_like_ratio = s[non_missing].str.match(r"^[+-]?[0-9\s.,]+$", na=False).mean()
        if numeric_like_ratio < min_ratio:
            continue

        candidates = {
            "plain": pd.to_numeric(s, errors="coerce"),
            "us": pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce"),
            "eu": pd.to_numeric(
                s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce",
            ),
        }

        best_values, best_ratio = None, -1.0
        for values in candidates.values():
            ratio = values[non_missing].notna().mean()
            if ratio > best_ratio:
                best_values, best_ratio = values, ratio

        if best_values is not None and best_ratio >= min_ratio:
            converted[col] = best_values

    return converted


def read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Read CSV files with delimiter and encoding fallback."""
    separators = [None, ";", ",", "\t", "|"]
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    best_df = None
    best_score = (-1, -1)
    last_exc = None

    for encoding in encodings:
        for sep in separators:
            try:
                uploaded_file.seek(0)
                candidate = pd.read_csv(uploaded_file, sep=sep, engine="python", encoding=encoding)
                score = (candidate.shape[1], candidate.shape[0])
                if score > best_score:
                    best_df = candidate
                    best_score = score
            except Exception as exc:
                last_exc = exc

    if best_df is None:
        raise ValueError(f"Could not parse CSV file. Last error: {last_exc}")

    return _coerce_numeric_like_columns(best_df)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return _coerce_numeric_like_columns(pd.read_excel(uploaded_file))
    return read_csv_flexible(uploaded_file)


def clean_bank_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").copy()
    index_like_columns = [col for col in cleaned.columns if str(col).lower().startswith("unnamed")]
    return cleaned.drop(columns=index_like_columns, errors="ignore")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def yes_no_input_to_model(value: str) -> str:
    return "yes" if value == "Yes" else "no"


def normalize_yes_no_columns(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Keep yes/no values consistent for model scoring."""
    normalized = df.copy()
    value_map = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "0": "no",
    }

    for col in features:
        if col in normalized.columns:
            normalized[col] = (
                normalized[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .map(value_map)
                .fillna(normalized[col])
            )

    return normalized


def predict_with_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def score_dataset(df: pd.DataFrame, model, features: list[str], threshold: float) -> tuple[pd.DataFrame, list[str]]:
    result = clean_bank_frame(df)
    missing_features = [feature for feature in features if feature not in result.columns]
    if missing_features:
        return result, missing_features

    result = normalize_yes_no_columns(result, features)
    probabilities = model.predict_proba(result[features])[:, 1]
    result["positive_response_probability"] = probabilities
    result["positive_response_flag"] = predict_with_threshold(probabilities, threshold)
    return result, []


def highlight_prediction_columns(df: pd.DataFrame):
    def _highlight(_row):
        styles = []
        for col in df.columns:
            if col == "positive_response_probability":
                styles.append("background-color: #E8F1FF; font-weight: 600;")
            elif col == "positive_response_flag":
                value = _row.get(col)
                if value == 1:
                    styles.append("background-color: #E8F5E9; color: #166534; font-weight: 700;")
                else:
                    styles.append("background-color: #F3F4F6; color: #374151; font-weight: 600;")
            else:
                styles.append("")
        return styles

    return _highlight


def render_model_context(metadata: dict, threshold: float) -> None:
    st.caption(
        f"Model: {metadata['model_name']}. Inputs: {', '.join(metadata['features'])}. "
        f"Output: probability of positive campaign response. Decision threshold: {threshold:.2f}."
    )


if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model artifacts were not found. Run train_bank_logistic_regression_model.ipynb first.")
    st.stop()

if not SAMPLE_PREDICTION_PATH.exists():
    st.error("Sample prediction dataset was not found.")
    st.stop()

model = load_model()
metadata = load_metadata()
features = metadata["features"]
fixed_threshold = float(metadata.get("threshold_default", 0.5))

if "single_prediction_result" not in st.session_state:
    st.session_state["single_prediction_result"] = None
if "batch_uploader_key" not in st.session_state:
    st.session_state["batch_uploader_key"] = 0
if "batch_df" not in st.session_state:
    st.session_state["batch_df"] = None
if "batch_scored_df" not in st.session_state:
    st.session_state["batch_scored_df"] = None

main_tab_single, main_tab_batch = st.tabs(["Individual prediction", "Batch prediction"])

# =========================================================
# TAB 1 - Individual prediction
# =========================================================
with main_tab_single:
    st.subheader("Single customer prediction")
    render_model_context(metadata, fixed_threshold)

    form_col, result_col = st.columns([1, 2], gap="large")

    with form_col:
        with st.container(border=True):
            with st.form("single_prediction_form", border=False):
                st.markdown("#### Customer profile")
                job = st.selectbox("Has job?", ["Yes", "No"], index=0, help="Model input: job")
                education = st.selectbox("Has education indicator?", ["Yes", "No"], index=0, help="Model input: education")
                housing = st.selectbox("Has housing loan?", ["Yes", "No"], index=0, help="Model input: housing")
                loan = st.selectbox("Has personal loan?", ["Yes", "No"], index=1, help="Model input: loan")

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    submitted = st.form_submit_button(
                        "Predict response",
                        type="primary",
                        width="stretch",
                        icon=":material/play_arrow:",
                    )
                with btn_col2:
                    clear_single = st.form_submit_button(
                        "Clear",
                        width="stretch",
                        icon=":material/clear:",
                    )

            if clear_single:
                st.session_state["single_prediction_result"] = None
                st.rerun()

            if submitted:
                input_df = pd.DataFrame([
                    {
                        "job": yes_no_input_to_model(job),
                        "education": yes_no_input_to_model(education),
                        "housing": yes_no_input_to_model(housing),
                        "loan": yes_no_input_to_model(loan),
                    }
                ])
                scored_df, missing_features = score_dataset(input_df, model, features, fixed_threshold)

                if missing_features:
                    st.error(f"Prediction unavailable. Missing model inputs: {', '.join(missing_features)}")
                else:
                    st.session_state["single_prediction_result"] = scored_df
                    st.rerun()

    with result_col:
        with st.container(border=True):
            st.markdown("#### Prediction result")
            result = st.session_state["single_prediction_result"]
            if result is None:
                st.info("Click 'Predict response' to score the customer profile.")
            else:
                probability = float(result.loc[0, "positive_response_probability"])
                decision = int(result.loc[0, "positive_response_flag"])
                k1, k2 = st.columns(2)
                k1.metric("Probability", f"{probability:.1%}")
                k2.metric("Prediction", "Positive" if decision == 1 else "No")

                display_df = result[features + ["positive_response_probability", "positive_response_flag"]]
                st.dataframe(
                    display_df.style.apply(highlight_prediction_columns(display_df), axis=1).format(
                        {"positive_response_probability": "{:.3f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

# =========================================================
# TAB 2 - Batch prediction
# =========================================================
with main_tab_batch:
    st.subheader("Batch prediction from file")
    render_model_context(metadata, fixed_threshold)

    input_col, preview_col = st.columns([1, 2], gap="large")

    with input_col:
        with st.container(border=True):
            st.markdown("#### 1. Input data")
            uploaded = st.file_uploader(
                "Upload CSV or Excel",
                type=["csv", "xlsx", "xls"],
                key=f"file_uploader_{st.session_state['batch_uploader_key']}",
                help=f"Expected columns: {', '.join(features)}. The target variable is not required.",
            )

            if uploaded is not None:
                try:
                    st.session_state["batch_df"] = clean_bank_frame(read_uploaded_file(uploaded))
                    st.session_state["batch_scored_df"] = None
                except Exception as exc:
                    st.error(f"Could not read the file: {exc}")

           

            df = st.session_state["batch_df"]
            data_ready = df is not None and not df.empty
            run_batch = st.button(
                "Run prediction",
                type="primary",
                width="stretch",
                icon=":material/play_arrow:",
                disabled=not data_ready,
            )


            sample_df = load_prediction_sample()
            st.download_button(
                "Download sample CSV",
                data=to_csv_bytes(sample_df),
                file_name="sample_bank_marketing_prediction.csv",
                mime="text/csv",
                width="stretch",
                icon=":material/download:",
            )

            clear_batch = st.button(
                "Clear / reset",
                width="stretch",
                icon=":material/clear:",
            )

            if clear_batch:
                st.session_state["batch_df"] = None
                st.session_state["batch_scored_df"] = None
                st.session_state["batch_uploader_key"] += 1
                st.rerun()

            if not data_ready:
                st.info("Upload a CSV/XLSX file to enable prediction.")

            if run_batch and data_ready:
                scored_df, missing_features = score_dataset(df, model, features, fixed_threshold)
                if missing_features:
                    st.error(f"Prediction unavailable. Missing model inputs: {', '.join(missing_features)}")
                else:
                    st.session_state["batch_scored_df"] = scored_df

    with preview_col:
        with st.container(border=True):
            st.markdown("#### 2. Dataset preview")
            df = st.session_state["batch_df"]
            if df is None:
                st.caption("Preview will appear after a dataset is uploaded.")
            elif df.empty:
                st.error("The dataset is empty after removing fully blank rows.")
            else:
                st.dataframe(df.head(20), width="stretch", hide_index=True)

    scored_df = st.session_state["batch_scored_df"]
    if scored_df is not None:
        st.markdown("### Processed output preview")
        output_cols = features + ["positive_response_probability", "positive_response_flag"]
        output_cols = [col for col in output_cols if col in scored_df.columns]
        output_preview = scored_df[output_cols].head(30)

        st.dataframe(
            output_preview.style.apply(highlight_prediction_columns(output_preview), axis=1).format(
                {"positive_response_probability": "{:.3f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download scored dataset",
            data=to_csv_bytes(scored_df),
            file_name="bank_marketing_scored_logistic_regression.csv",
            mime="text/csv",
            width="stretch",
            icon=":material/download:",
        )
