"""Smoke-test the free Gradio Space before porting the full app.

Hugging Face's free ZeroGPU hardware requires at least one ``@spaces.GPU``
function to exist at startup ("No @spaces.GPU function detected during
startup"), so the test declares one even though the real model is CPU-only.

If this reaches RUNNING, the free ZeroGPU tier works for this account and the
full app can be deployed there. If it fails with a quota/payment error, Hugging
Face hosting needs PRO.
"""

import gradio as gr
import spaces


@spaces.GPU
def gpu_probe(name: str) -> str:
    """Runs on the ZeroGPU worker; the model itself does not need a GPU."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = name or "there"
    return f"Hello {name} - the Gradio Space is running on {device}."


def greet(name: str) -> str:
    return f"Hello {name or 'there'} - the Gradio Space is running."


with gr.Blocks(title="SuperKart Space smoke test") as demo:
    gr.Markdown(
        "# SuperKart Space smoke test\n"
        "Temporary check that this Space can start on the free ZeroGPU tier."
    )
    with gr.Row():
        name_box = gr.Textbox(label="Your name", placeholder="Ruksana")
    with gr.Row():
        plain_btn = gr.Button("Plain greeting (CPU)")
        gpu_btn = gr.Button("GPU probe (ZeroGPU)")
    out = gr.Textbox(label="Response")
    plain_btn.click(greet, inputs=name_box, outputs=out)
    gpu_btn.click(gpu_probe, inputs=name_box, outputs=out)

demo.launch()
