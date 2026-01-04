from faster_whisper.utils import download_model

# Standard Whisper models
model_names = [
    "small",
    "medium",
    "large-v2",
    "turbo",
]

# Language override models (HuggingFace model IDs)
# These are specialized models used when specific languages are detected.
# IMPORTANT: This list must be kept in sync with LANGUAGE_OVERRIDES in src/predict.py (lines 35-40)
# When adding a new language override, add the model ID here as well.
language_override_models = [
    "ivrit-ai/whisper-large-v3-turbo-ct2",  # Hebrew (he) - from LANGUAGE_OVERRIDES["he"]
    # Add future language override models here as they're added to LANGUAGE_OVERRIDES in predict.py
]


def download_model_weights(selected_model):
    """
    Download model weights.
    """
    print(f"Downloading {selected_model}...")
    download_model(selected_model, cache_dir=None)
    print(f"Finished downloading {selected_model}.")


# Download standard Whisper models
print("Downloading standard Whisper models...")
for model_name in model_names:
    download_model_weights(model_name)

# Download language override models (HuggingFace models)
print("\nDownloading language override models...")
for model_name in language_override_models:
    download_model_weights(model_name)

print("\nFinished downloading all models.")
