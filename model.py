import os
import torch
import subprocess
import re
import glob
import gc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import export_to_gif, load_image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from preprocess import gif_to_images, stitch_images, resize_image, split_image, images_to_gif

def generate_video(prompt="A girl dancing", output_gif="output.gif", frame_dir="raw_frames"):
    device = "cuda"
    dtype = torch.float16

    step = 8  # Options: [1,2,4,8]
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
    
    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)
    export_to_gif(output.frames[0], output_gif)
    gif_to_images(output_gif, frame_dir)


def key_frame_img2img(input_dir="bin/key_frames",
                      key_frame_indices=[4, 8, 12, 16],
                      prompt="4k, 1girl, dancing, realistic, clear face, high quality", 
                      negative_prompt="bad anatomy, poorly drawn hands, extra limbs",
                      strength=0.3, guidance_scale=100):
    stitch_images(input_dir=input_dir, output_image="bin/key_frames.png",grid_size=(2,2))
    resize_image("bin/key_frames.png", "bin/key_frames.png", (2048, 2048))

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    # use from_pipe to avoid consuming additional memory when loading a checkpoint
    pipeline = AutoPipelineForImage2Image.from_pipe(pipeline_text2image).to("cuda")
    #pipeline = AutoPipelineForImage2Image.from_pretrained(
    #    "stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    #).to("cuda")
    init_image = load_image("bin/key_frames.png")
    image = pipeline(prompt, negative_prompt=negative_prompt,image=init_image, strength=strength, guidance_scale=guidance_scale).images[0]
    image.save("bin/refined_key_frames.png")
    files = glob.glob(os.path.join("bin/refined_key_frames", '*'))
    for f in files:
        os.remove(f)
    split_image(image_path="bin/refined_key_frames.png", output_dir="bin/refined_key_frames", grid_size=(2, 2),frame_indices=key_frame_indices)


def ebsynth_frame(input_dir="bin/raw_frames", output_dir="bin/refined_frames", style_dir="bin/refined_key_frames"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    key_frame_list = [filename for filename in os.listdir(style_dir) if filename.endswith(".png")]
    key_frame_list.sort(key=lambda f: int(re.search(r'frame_(\d+).png', f).group(1)))

    frames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    frames.sort(key=lambda f: int(re.search(r'frame_(\d+).png', f).group(1)))

    for i, key_frame in enumerate(key_frame_list):
        output_dir_i = os.path.join(output_dir, f"{i}")
        if not os.path.exists(output_dir_i):
            os.makedirs(output_dir_i)

        imgpath = os.path.join(input_dir, key_frame)
        resize_image(imgpath, imgpath, (1024, 1024))
        for filename in frames:
            imgpath = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir_i, filename)
            resize_image(imgpath, imgpath, (1024, 1024))
            command = f"export PATH=ebsynth/bin:$PATH\nebsynth -style {imgpath} -guide {os.path.join(input_dir, key_frame)} {os.path.join(style_dir, key_frame)} -output {output_path}"
            process = subprocess.Popen(command, shell=True, executable="/bin/bash")
            output, error = process.communicate()
            if error:
                print("Error:", error)
        images_to_gif(input_dir=output_dir_i, output_gif=f"bin/output_{i}.gif")
s

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


def show_masks_on_image(raw_image, masks):
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=False)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭图像，以释放内存
    del mask
    gc.collect()


def segment_image(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        raw_image = Image.open(img_path)
        generator = pipeline("mask-generation", model="/share/lab5/sam/", device=0)
        outputs = generator(raw_image, points_per_batch=64)
        masks = outputs["masks"]

        
    
