import gradio as gr
from inference_pipeline import Pipeline


weight_path = "/home/rteam1/faryad/General_Generative_Defect/results/v1-5-pruned-emaonly.ckpt"
converted_path = "/home/rteam1/faryad/General_Generative_Defect/results/save_converted_model"

pipeline = Pipeline(weight_path=weight_path, converted_path=converted_path)

def defect_generate(prompt, dict, padding, blur_len, strength_slider, CFG_Scale_slider):
    init_img =  dict['image'].convert("RGB")
    mask_img = dict['mask'].convert("RGB")
    # image = generator.generate(prompt)
    image = pipeline.generate(prompt, 
                              init_img, 
                              mask_img, padding, blur_len, strength_slider, CFG_Scale_slider, transparency=0.5,transparency="")
    return image
css = '''
#image_upload{min-height:800px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 800px}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown(
    """
    # General Generative Defect
    Start typing below to see the output.
    """)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="prompt..")
        greet_btn = gr.Button("Generate").style(full_width=False)
    with gr.Row():
        strength_slider = gr.Slider(0, 1, 0.75, label="Denoising strength")
        CFG_Scale_slider = gr.Slider(1, 50, 13, label="CFG Scale")
    with gr.Row():
        padding_slider = gr.Slider(0, 256, 32,label="Mask Padding")
        blur_slider = gr.Slider(1, 256, 9,label="Mask Blur")
    input_img = gr.Image(label="Image", elem_id="image_upload",type='pil', tool='sketch').style(height=800)
    output = gr.Image(label="Generated Image")
    
    greet_btn.click(fn=defect_generate, inputs=[prompt, input_img, padding_slider, blur_slider, strength_slider, CFG_Scale_slider], outputs=output, api_name="General Generative Defect", )


demo.launch(share=False)