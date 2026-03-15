"""
NeuTTS Gradio Demo
------------------
A simple web UI for running NeuTTS text-to-speech inference.
Launch with:  python gradio_app.py
"""

import os
import tempfile

import gradio as gr
import soundfile as sf
import torch

from neutts import NeuTTS

# ---------------------------------------------------------------------------
# Available backbone models
# ---------------------------------------------------------------------------
MODELS = {
    "NeuTTS-Nano (English)": "neuphonic/neutts-nano",
    "NeuTTS-Air (English)": "neuphonic/neutts-air",
    "NeuTTS-Nano French": "neuphonic/neutts-nano-french",
    "NeuTTS-Nano German": "neuphonic/neutts-nano-german",
    "NeuTTS-Nano Spanish": "neuphonic/neutts-nano-spanish",
}

SAMPLE_RATE = 24_000

# Cache one loaded model to avoid reloading on every click
_loaded: dict = {}


def _get_tts(backbone_repo: str) -> NeuTTS:
    if backbone_repo not in _loaded:
        _loaded.clear()
        _loaded[backbone_repo] = NeuTTS(
            backbone_repo=backbone_repo,
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu",
        )
    return _loaded[backbone_repo]


# ---------------------------------------------------------------------------
# Inference function called by Gradio
# ---------------------------------------------------------------------------
def generate_speech(
    model_name: str,
    input_text: str,
    ref_audio,
    ref_text: str,
):
    if not input_text.strip():
        return None, "⚠️ Please enter some text to synthesise."

    backbone = MODELS[model_name]

    try:
        tts = _get_tts(backbone)
    except Exception as exc:
        return None, f"❌ Failed to load model: {exc}"

    # Encode reference audio if provided
    ref_codes = None
    if ref_audio is not None and ref_text.strip():
        try:
            ref_codes = tts.encode_reference(ref_audio)
        except Exception as exc:
            return None, f"❌ Failed to encode reference audio: {exc}"

    try:
        wav = tts.infer(input_text, ref_codes, ref_text.strip() if ref_codes is not None else None)
    except Exception as exc:
        return None, f"❌ Inference failed: {exc}"

    # Write to a temporary file so Gradio can serve it.
    # delete=False is required because Gradio reads the file after this function
    # returns; the OS will reclaim /tmp on reboot / its own sweep policy.
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, wav, SAMPLE_RATE)
    return tmp.name, "✅ Audio generated successfully."


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="NeuTTS Live Demo") as demo:
    gr.Markdown(
        """
        # 🗣 NeuTTS Live Demo
        On-device, real-time text-to-speech with instant voice cloning.
        Select a model, enter your text and (optionally) upload a reference audio clip for voice cloning,
        then click **Generate**.

        > 💡 Models are downloaded on first use — this may take a minute.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="NeuTTS-Nano (English)",
                label="Model",
            )
            input_text = gr.Textbox(
                label="Text to synthesise",
                placeholder="Hello! This is NeuTTS speaking.",
                lines=4,
            )

        with gr.Column(scale=1):
            ref_audio = gr.Audio(
                label="Reference audio for voice cloning (optional, ≥ 3 s)",
                type="filepath",
            )
            ref_text = gr.Textbox(
                label="Transcription of reference audio (optional)",
                placeholder="Exact words spoken in the reference clip…",
                lines=3,
            )

    generate_btn = gr.Button("🎙 Generate", variant="primary")

    with gr.Row():
        audio_out = gr.Audio(label="Generated speech", type="filepath")
        status_msg = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_speech,
        inputs=[model_dropdown, input_text, ref_audio, ref_text],
        outputs=[audio_out, status_msg],
    )

    gr.Markdown(
        """
        ---
        Built by [Neuphonic](https://neuphonic.com/) · [GitHub](https://github.com/neuphonic/neutts)
        · [HuggingFace](https://huggingface.co/neuphonic)
        """
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTS Gradio Demo")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")
    args = parser.parse_args()

    demo.launch(share=args.share, server_port=args.port)
