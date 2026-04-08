from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from config import ARTIFACTS_DIR, EXAMPLES_DIR, MIN_TEXT_LENGTH
from inference import MathProblemInferenceService

st.set_page_config(
    page_title="Math Problem Analyzer",
    page_icon="📘",
    layout="wide",
)


@st.cache_resource
def load_service(artifacts_dir: str) -> MathProblemInferenceService:
    return MathProblemInferenceService(artifacts_dir=artifacts_dir)


def read_example(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def main() -> None:
    st.title("Math Problem Analyzer")
    st.caption("Multitask transformer classifier: topic + difficulty")

    with st.sidebar:
        st.subheader("Settings")
        artifacts_dir = st.text_input(
            "Artifacts directory",
            value=str(ARTIFACTS_DIR),
        )

        st.subheader("Example Problems")
        if st.button("Geometry example", use_container_width=True):
            st.session_state["input_problem"] = read_example(
                EXAMPLES_DIR / "geometry.txt"
            )
        if st.button("Number theory example", use_container_width=True):
            st.session_state["input_problem"] = read_example(
                EXAMPLES_DIR / "number_theory.txt"
            )
        if st.button("Algebra example", use_container_width=True):
            st.session_state["input_problem"] = read_example(
                EXAMPLES_DIR / "algebra.txt"
            )

    if "input_problem" not in st.session_state:
        st.session_state["input_problem"] = ""

    problem_text = st.text_area(
        "Enter math problem text",
        value=st.session_state["input_problem"],
        height=260,
        placeholder="Paste a full math problem statement here...",
    )

    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    )

    if not analyze_clicked:
        return

    if not problem_text.strip():
        st.error("Input is empty. Please enter a math problem.")
        return

    if len(problem_text.strip()) < MIN_TEXT_LENGTH:
        st.error(
            "Input is too short. Please provide at least "
            f"{MIN_TEXT_LENGTH} characters."
        )
        return

    try:
        service = load_service(artifacts_dir)
    except FileNotFoundError:
        st.error(
            "Trained artifacts are missing. Run training first: "
            "python train.py --artifacts-dir artifacts"
        )
        return
    except Exception as exc:
        st.error(f"Failed to load model artifacts: {exc}")
        return

    try:
        result = service.predict(problem_text)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    st.subheader("Results")

    confidence = (
        max(result.topic_probabilities.values())
        if result.topic_probabilities
        else 0.0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Topic", result.best_topic)
    col2.metric(
        "Difficulty",
        f"{result.best_difficulty} (level {result.difficulty_level})",
    )
    col3.metric("Confidence", f"{confidence:.2%}")
    st.progress(int(confidence * 100))
    st.caption(
        f"Predicted raw difficulty score: {result.difficulty_score:.3f}"
    )

    st.markdown("### Top-95% Topics")
    top95_df = pd.DataFrame(result.top95_topics)
    if not top95_df.empty:
        top95_df["probability"] = top95_df["probability"].map(
            lambda x: f"{x:.2%}"
        )
        st.dataframe(top95_df, use_container_width=True, hide_index=True)

    st.markdown("### Topic Probabilities")
    topic_df = pd.DataFrame(
        [
            {"topic": topic, "probability": prob}
            for topic, prob in result.topic_probabilities.items()
        ]
    )
    chart_df = topic_df.set_index("topic")
    st.bar_chart(chart_df)
    topic_df["probability"] = topic_df["probability"].map(lambda x: f"{x:.2%}")
    st.dataframe(topic_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
