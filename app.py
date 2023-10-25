import os
import gradio as gr

from utils.constants import choices, title
from EasyOcrImageToTextGradio.actions.image_to_ocr import inference

os.environ["CUDA_VISIBLE_DEVICES"] = ""

block = gr.Blocks(css="footer {visibility: hidden}", theme='freddyaboulton/dracula_revamped', title=title).queue()

with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## <p align='center'> OCR: Image ⮕ Text </p>")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy")
            input_lan = gr.CheckboxGroup(choices, type="value", value=['en'], label='language')
            button = gr.Button("Submit")
        with gr.Column():
            output_image = gr.Image(type="numpy")
            output_dataframe = gr.Dataframe(headers=['Coordinates', 'Text', 'Confidence'])

        button.click(fn=inference,
                     inputs=[input_image, input_lan],
                     outputs=[output_image, output_dataframe])

block.launch(share=True)
