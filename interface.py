import os
import torch
import subprocess
import re
import glob
import gc
import argparse
import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mp
import gradio
from PIL import Image
from transformers import pipeline
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import export_to_gif, load_image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from preprocess import *
from preprocess import resize_image, key_frame_extraction, stitch_images, images_to_gif, gif_to_images

def import_video(input_gif, frame_dir="bin/raw_frames"):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    else:
        files = glob.glob(os.path.join(frame_dir, '*'))
        for f in files:
            os.remove(f)
    gif_to_images(input_gif, frame_dir)
    for filename in os.listdir(frame_dir):
        img_path = os.path.join(frame_dir, filename)
        resize_image(img_path, img_path, (512, 512))


def generate_video(prompt="A girl dancing, poor quality", output_gif="bin/raw.gif", frame_dir="bin/raw_frames"):
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
                      lora=None,
                      prompt="4k, 1girl, dancing, realistic, clear face, high quality", 
                      negative_prompt="bad anatomy, poorly drawn hands, extra limbs",
                      strength=0.2, guidance_scale=50):
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
    if lora is not None:
        pipeline.load_lora_weights(lora)
    init_image = load_image("bin/key_frames.png")
    image = pipeline(prompt, negative_prompt=negative_prompt,image=init_image, strength=strength, guidance_scale=guidance_scale).images[0]
    image.save("bin/refined_key_frames.png")
    files = glob.glob(os.path.join("bin/refined_key_frames", '*'))
    for f in files:
        os.remove(f)
    split_image(image_path="bin/refined_key_frames.png", output_dir="bin/refined_key_frames", grid_size=(2, 2),frame_indices=key_frame_indices)


def ebsynth_frame(input_dir="bin/raw_frames", output_dir="bin/refined_frames", style_dir="bin/refined_key_frames", single=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        files = glob.glob(os.path.join(output_dir, '*'))
        for f in files:
            os.remove(f)
    gray_input_dir = "bin/gray_raw_key_frames"
    if not os.path.exists(gray_input_dir): 
        os.makedirs(gray_input_dir)
    else:
        files = glob.glob(os.path.join(gray_input_dir, '*'))
        for f in files:
            os.remove(f)

    key_frame_list = [filename for filename in os.listdir(style_dir) if filename.endswith(".png")]
    if single:
        key_frame_list = [key_frame_list[0]]
    key_frame_list.sort(key=lambda f: int(re.search(r'frame_(\d+).png', f).group(1)))
    key_frame_idx_list = [int(re.search(r'frame_(\d+).png', f).group(1)) for f in key_frame_list]

    for frame in os.listdir(input_dir):
        imgpath = os.path.join(input_dir, frame)
        resize_image(imgpath, imgpath, (1024, 1024))
        convert_to_grayscale(imgpath, os.path.join(gray_input_dir, frame))

    

    frames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    frames.sort(key=lambda f: int(re.search(r'frame_(\d+).png', f).group(1)))

    for i, frame in enumerate(frames):
        raw_path = os.path.join(input_dir, frame)
        output_path = os.path.join(output_dir, frame)
        
        resize_image(raw_path, raw_path, (1024, 1024))

        nearest_key_frame_idx = min(key_frame_idx_list, key=lambda x: abs(x - i))
        nearest_key_frame = f"frame_{nearest_key_frame_idx}.png"

        command = f"export PATH=ebsynth/bin:$PATH\nebsynth -style {os.path.join(style_dir, nearest_key_frame)} "

        guide = f"-guide {os.path.join(input_dir, nearest_key_frame)} {raw_path} -weight {2} "
        gray_guide = f"-guide {os.path.join(gray_input_dir, nearest_key_frame)} {os.path.join(gray_input_dir, frame)} -weight {1.5} "
        command += guide
        command += gray_guide 
        command += f"-output {output_path} -patchsize 7 -searchvoteiters 10 -extrapass3x3"

        process = subprocess.Popen(command, shell=True, executable="/bin/bash")
        output, error = process.communicate()
        if error:
            print("Error:", error)
        if frame in key_frame_list and i < len(key_frame_list) - 1:
            i += 1
    return images_to_gif(input_dir=output_dir, output_gif=f"bin/output.gif")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.2])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


def show_masks_on_image(raw_image, masks, output_path):
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
        output_path = os.path.join(output_dir, filename)
        raw_image = Image.open(img_path)
        generator = pipeline("mask-generation", model="/share/lab5/sam/", device=0)
        outputs = generator(raw_image, points_per_batch=64)
        masks = outputs["masks"]
        show_masks_on_image(raw_image, masks, output_path)

def Generate_video(Import, Import_video, video_prompt, style, lora):
    os.environ['HTTP_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890"
    os.environ['HTTPS_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890"
    os.environ['ALL_PROXY']="socks5://Clash:QOAF8Rmd@10.1.0.213:7893"
    video_path = "bin/raw.gif"
    if Import:
        # save Import_video to bin/raw.mp4
        # transform video to gif
        clip = mp.VideoFileClip(Import_video)
        clip.write_gif(video_path)

        import_video(input_gif=video_path, frame_dir="bin/raw_frames")
        print("Imported video successfully.")

    else:
        print("Generating video...")
        generate_video(prompt=video_prompt, output_gif=f"bin/raw.gif", frame_dir="bin/raw_frames")
        print("Generated video successfully.")
    interval = len(os.listdir("bin/raw_frames")) // 4
    key_frame_indices = [interval * i for i in range(1, 5) if interval * i < len(os.listdir("bin/raw_frames")) + 1]
    print("Extracting key frames...")
    key_frame_extraction(input_dir="bin/raw_frames", output_dir="bin/key_frames", key_frame_indices=key_frame_indices)
    print("Extracted key frames successfully.")
    if not lora:
        print("Running img2img...")
        key_frame_img2img(input_dir="bin/key_frames",
                        key_frame_indices=key_frame_indices,
                        prompt=style,
                        negative_prompt="bad anatomy, poorly drawn hands, extra limbs",
                        strength=0.4, guidance_scale=50)
    else:
        print("Running img2img with lora...")
        key_frame_img2img(input_dir="bin/key_frames",
                        key_frame_indices=key_frame_indices,
                        prompt=style,
                        negative_prompt="bad anatomy, poorly drawn hands, extra limbs",
                        strength=0.3, guidance_scale=50)
    print("Running img2img successfully.")
    ebsynth_frame(input_dir="bin/raw_frames", output_dir="bin/refined_frames", style_dir="bin/refined_key_frames")
    clip = mp.VideoFileClip("bin/raw.gif")
    clip.write_videofile("bin/raw.mp4")
    clip2 = mp.VideoFileClip("bin/output.gif")
    clip2.write_videofile("bin/output.mp4")
    return "bin/raw.mp4", "bin/output.mp4"

if __name__ == "__main__":
    demo = gradio.Interface(
        fn=Generate_video,
        inputs=["checkbox", "video", "text", "text", "checkbox"],
        outputs=["video","video"],
    )

    demo.launch(share=True)
