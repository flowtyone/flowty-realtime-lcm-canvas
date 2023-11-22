import os
import argparse
import gradio as gr
from main import load_models, cache_path
from PIL import Image
from os import path

canvas_size = 512

if not path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)

with gr.Blocks() as demo:
    infer = load_models()

    with gr.Column():
        with gr.Row():
            with gr.Column():
                s = gr.Slider(label="steps", minimum=4, maximum=8, step=1, value=4, interactive=True)
                c = gr.Slider(label="cfg", minimum=0.1, maximum=3, step=0.1, value=1, interactive=True)
                i_s = gr.Slider(label="sketch strength", minimum=0.1, maximum=0.9, step=0.1, value=0.9, interactive=True)
            with gr.Column():
                mod = gr.Text(label="Model Hugging Face id (after changing this wait until the model downloads in the console)", value="Lykon/dreamshaper-7", interactive=True)
                t = gr.Text(label="Prompt", value="Scary warewolf, 8K, realistic, colorful, long sharp teeth, splash art", interactive=True)
                se = gr.Number(label="seed", value=1337, interactive=True)
        with gr.Row(equal_height=True):
            i = gr.Image(source="canvas", tool="color-sketch", shape=(canvas_size, canvas_size), width=canvas_size, height=canvas_size, type="pil")
            o = gr.Image(width=canvas_size, height=canvas_size)

            def process_image(p, im, steps, cfg, image_strength, seed):
                if not im:
                    return Image.new("RGB", (canvas_size, canvas_size))
                return infer(
                    prompt=p,
                    image=im,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    strength=image_strength,
                    seed=int(seed)
                )

            reactive_controls = [t, i, s, c, i_s, se]

            for control in reactive_controls:
                control.change(fn=process_image, inputs=reactive_controls, outputs=o)

            def update_model(model_name):
                global infer
                infer = load_models(model_name)

            mod.change(fn=update_model, inputs=mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # If the option python ui.py --share is attached, it will be deployed to Gradio
    parser.add_argument("--share", action="store_true", help="Deploy on Gradio for sharing", default=False)
    args = parser.parse_args()
    demo.launch(share=args.share)
