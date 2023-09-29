import os
import subprocess
import gradio as gr
from inference_pipeline import Pipeline



weight_path = "/home/naserwin/hamze/General_Generative_Defect/multi_subject_SD/results/plastic/plastic_chip_21sep2023.ckpt"
converted_path = "/home/naserwin/hamze/General_Generative_Defect/multi_subject_SD/results/plastic/tst5"

pipeline = Pipeline(weight_path=weight_path, converted_path=converted_path)

def defect_generate(prompt, dict, padding, blur_len, strength_slider, CFG_Scale_slider, transparency, num_inference_steps):
    init_img = dict['image'].convert("RGB")
    mask_img = dict['mask'].convert("RGB")
    image = pipeline.generate(prompt, init_img, mask_img, padding, blur_len, strength_slider, CFG_Scale_slider, transparency=transparency, num_inference_steps=num_inference_steps)
    return image

def grpu_proccess_kill():
    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    utilization = subprocess.check_output(cmd, shell=True)
    utilization = utilization.decode("utf-8").strip().split("\n")
    utilization = [int(x.replace(" %", "")) for x in utilization]
    os.system(f"kill -9 {utilization[-1]}")
    return utilization


css = '''
#image_upload{min-height:800px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 800px}
'''
with gr.Blocks(css=css) as demo:
    mark_down = gr.Markdown(
        """
        # General Generative Defect
        Start typing below to see the output.
        """
    )
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="prompt..")
        greet_btn = gr.Button("Generate").style(full_width=False)
        close_btn = gr.Button("close").style(full_width=False)
    with gr.Row():
        strength_slider = gr.Slider(0, 1, 0.75, label="Denoising strength")
        CFG_Scale_slider = gr.Slider(1, 300, 13, label="CFG Scale")
        transparency = gr.Slider(0, 1, 0.5, label="transparency")
    with gr.Row():
        padding_slider = gr.Slider(0, 256, 32, label="Mask Padding")
        blur_slider = gr.Slider(1, 256, 9, label="Mask Blur")
        num_inference_steps = gr.Slider(1, 300, 150, label="num_inference_steps")
    input_img = gr.Image(label="Image", elem_id="image_upload", type='pil', tool='sketch').style(height=800)
    output = gr.Image(label="Generated Image")
    
    greet_btn.click(fn=defect_generate, inputs=[prompt, input_img, padding_slider, blur_slider, strength_slider, CFG_Scale_slider, transparency, num_inference_steps], outputs=output, api_name="General Generative Defect")
    close_btn.click(grpu_proccess_kill)


demo.launch(share=False)
