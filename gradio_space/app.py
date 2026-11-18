"""Smoke-test the free Gradio Space before porting the full app.

If this tiny app reaches RUNNING, the free ZeroGPU tier works for your account
and the real app can be deployed there. If it fails with a quota/payment error,
Hugging Face hosting is not available without PRO.
"""

import gradio as gr


def greet(name: str) -> str:
    return f"Hello {name or 'there'} - the Gradio Space is running."


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Your name", placeholder="Ruksana"),
    outputs=gr.Textbox(label="Response"),
    title="SuperKart Space smoke test",
    description="Temporary check that this Space can start on the free tier.",
)

demo.launch()
