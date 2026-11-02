import os

# ── Environment loader ─────────────────────────────────────────────────────────
# Tries Streamlit secrets first (cloud deployment), falls back to .env (local dev)
def _get_secret(key: str, default: str = "") -> str:
    """
    Resolution order:
      1. Streamlit st.secrets  — when running on Streamlit Cloud
      2. OS environment / .env — when running locally via python-dotenv
      3. default               — empty string unless explicitly overridden
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass  # Streamlit not available or secrets not configured — fall through

    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv(key, default).strip()

# LLM
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

# Hugging face
HF_TOKEN = _get_secret("HF_TOKEN")

#Ml Flow
MLFLOW_TRACKING_URI    = _get_secret("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = _get_secret("MLFLOW_EXPERIMENT_NAME", "travel-agent")

#Validation

# Fail fast at startup if critical keys are missing rather than failing mid-request
_REQUIRED_KEYS = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "HF_TOKEN":     HF_TOKEN,
}

for _key, _val in _REQUIRED_KEYS.items():
    if not _val:
        raise ValueError(
            f"[config] Required secret '{_key}' is not set.\n"
            f"  Local  : add it to your .env file\n"
            f"  Cloud  : add it to Streamlit Cloud → App Settings → Secrets"
        )

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

print(f"Key loaded: {GROQ_API_KEY[:8]}...")  # Debug check
print(f"Model: {LLM_MODEL}")

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
# ML FLow
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "travel-agent")



# ── Debug (remove in production) ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"GROQ key loaded : {GROQ_API_KEY[:8]}...")
    print(f"HF token loaded : {HF_TOKEN[:8]}..." if HF_TOKEN else "HF_TOKEN: not set")
    print(f"Model           : {LLM_MODEL}")
    print(f"MLflow URI      : {MLFLOW_TRACKING_URI}")
    print(f"MLflow Exp      : {MLFLOW_EXPERIMENT_NAME}")
