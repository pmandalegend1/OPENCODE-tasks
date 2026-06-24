from __future__ import annotations

from pathlib import Path

import streamlit as st

from prompt_injection_detector.config import load_config
from prompt_injection_detector.model import Detector
from prompt_injection_detector.redteam import run_redteam

st.set_page_config(page_title="Prompt Injection Detector", layout="wide")

cfg = load_config()
DEFAULT_MODEL_PATH = str(Path(cfg["paths"]["artifacts_dir"]) / "detector.joblib")


@st.cache_resource
def get_detector(model_path: str) -> Detector | None:
    try:
        return Detector(model_path)
    except FileNotFoundError:
        return None


st.title("🛡️ Prompt Injection Detector")
st.caption("Catch attacks on LLMs — then watch the red team try to break the detector.")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    min_variants = st.slider("Min red-team variants", 5, 15, 5)

detector = get_detector(model_path)
if detector is None:
    st.error(
        f"No trained model found at `{model_path}`. Run `pid build-data` then "
        f"`pid train` first."
    )
    st.stop()

tab1, tab2 = st.tabs(["🔍 Detector", "⚔️ Red-Team View"])

with tab1:
    st.subheader("Paste a prompt to classify")
    user_input = st.text_area("Prompt input", height=150, placeholder="Paste any prompt here...")
    if st.button("Classify", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            result = detector.predict(user_input)
            if result["is_injection"]:
                st.error(f"🚨 Injection Detected — category: **{result['category']}**")
            else:
                st.success("✅ Clean")
            st.metric("Confidence (injection probability)", f"{result['confidence']:.1%}")
            if result["top_features"]:
                st.write("**Top features that triggered detection:**")
                st.write(", ".join(result["top_features"]))
            st.session_state["last_text"] = user_input
            st.session_state["last_result"] = result

with tab2:
    st.subheader("Generate evasion variants for a detected injection")
    default_text = st.session_state.get("last_text", "")
    redteam_input = st.text_area(
        "Injection text to attack", value=default_text, height=120,
        placeholder="Paste an injection (or use the text classified as injection in the Detector tab)...",
    )
    if st.button("Run Red-Team Generator"):
        if not redteam_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating evasion variants and scoring against the detector..."):
                result = run_redteam(redteam_input, detector, min_variants=min_variants)

            n_bypassed = len(result.successful_evasions)
            st.write(f"**{n_bypassed} / {len(result.variants)}** variants bypassed the detector.")

            for v in result.variants:
                status = "🟢 BYPASSED" if v.bypassed else "🔴 caught"
                with st.expander(f"{status} — strategy: `{v.strategy}` (confidence {v.confidence:.1%})"):
                    st.write(v.text)

            best = result.best_evasion
            if best:
                st.warning(
                    f"⚠️ Most successful evasion strategy: **{best.strategy}** "
                    f"(confidence dropped to {best.confidence:.1%})"
                )
            else:
                st.success("The detector held up against every variant.")

st.divider()
st.caption(
    "Prompt Injection Detector — OWASP LLM Top 10 #1 risk. Built with scikit-learn, "
    "a custom red-team generator, and an adversarial training loop."
)
