import os
import torch
import subprocess
import re
import glob
import gc
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import pipeline
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import export_to_gif, load_image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from preprocess import *

def import_video(input_gif, frame_dir="bin/raw_frames"):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
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
                      prompt="4k, 1girl, dancing, realistic, clear face, high quality", 
                      negative_prompt="bad anatomy, poorly drawn hands, extra limbs",
                      strength=0.5, guidance_scale=100):
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


def ebsynth_frame(input_dir="bin/raw_frames", output_dir="bin/refined_frames", style_dir="bin/refined_key_frames", single=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gray_input_dir = "bin/gray_raw_key_frames"
    if not os.path.exists(gray_input_dir): 
        os.makedirs(gray_input_dir)
    else:
        files = glob.glob(os.path.join(gray_input_dir, '*'))
        for f in files:
            os.remove(f)
    gray_style_dir = "bin/gray_refined_key_frames"
    if not os.path.exists(gray_style_dir):
        os.makedirs(gray_style_dir)
    else:
        files = glob.glob(os.path.join(gray_style_dir, '*'))
        for f in files:
            os.remove(f)

    key_frame_list = [filename for filename in os.listdir(style_dir) if filename.endswith(".png")]
    key_frame_list.sort(key=lambda f: int(re.search(r'frame_(\d+).png', f).group(1)))

    if single:
        key_frame_list = [key_frame_list[0]]

    for frame in key_frame_list:
        imgpath = os.path.join(input_dir, frame)
        resize_image(imgpath, imgpath, (1024, 1024))
        convert_to_grayscale(imgpath, os.path.join(gray_input_dir, frame))
        convert_to_grayscale(imgpath, os.path.join(gray_style_dir, frame))


    frames = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    frames.sort(key=lambda f: int(re.search(r'frame_(\d+).png', f).group(1)))

    for i, filename in enumerate(frames):
        imgpath = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        resize_image(imgpath, imgpath, (1024, 1024))

        command = f"export PATH=ebsynth/bin:$PATH\nebsynth -style {imgpath} "
        for key_frame in key_frame_list:
            idx = int(re.search(r'frame_(\d+).png', key_frame).group(1)) - 1
            weight = 2 * np.exp(-(i - idx) ** 2 / 16)
            guide = f"-guide {os.path.join(input_dir, key_frame)} {os.path.join(style_dir, key_frame)} -weight {weight} "
            gray_guide = f"-guide {os.path.join(gray_input_dir, key_frame)} {os.path.join(gray_style_dir, key_frame)} -weight {0.5*weight} "
            command += guide
            command += gray_guide
        command += f"-output {output_path} -patchsize 3 -searchvoteiters 10 -extrapass3x3"

        process = subprocess.Popen(command, shell=True, executable="/bin/bash")
        output, error = process.communicate()
        if error:
            print("Error:", error)
        if filename in key_frame_list and i < len(key_frame_list) - 1:
            i += 1
    images_to_gif(input_dir=output_dir, output_gif=f"bin/output.gif")

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

if __name__ == "__main__":

    # Open VPN
    os.environ['HTTP_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890"
    os.environ['HTTPS_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890"
    os.environ['ALL_PROXY']="socks5://Clash:QOAF8Rmd@10.1.0.213:7893"
    
    parser = argparse.ArgumentParser(description='Smooth Video.')
    parser.add_argument('--mode', type=str, required=True, help='Mode for the video. Options: [import, generate]')
    parser.add_argument('--video_path', type=str, help='Path for the video if it is import mode.')
    parser.add_argument('--video_prompt', type=str, help='Prompt for the video if it is generate mode.')
    parser.add_argument('--image_prompt', type=str, required=True, help='Prompt for the img2img.')
    args = parser.parse_args()

    if args.mode == "import":
        import_video(input_gif=args.video_path, frame_dir="bin/raw_frames")
        print("Imported video successfully.")
        if args.video_path is None:
            parser.print_help()
            exit(1)

    elif args.mode == "generate":
        print("Generating video...")
        generate_video(prompt=args.video_prompt, output_gif="bin/raw.gif", frame_dir="bin/raw_frames")
        print("Generated video successfully.")
        if args.video_prompt is None:
            parser.print_help()
            exit(1)
    else:
        parser.print_help()
        exit(1)

    if args.image_prompt is None:
        parser.print_help()
        exit(1)
    interval = len(os.listdir("bin/raw_frames")) // 4
    key_frame_indices = [interval * i for i in range(1, 5) if interval * i < len(os.listdir("bin/raw_frames")) + 1]
    print("Extracting key frames...")
    key_frame_extraction(input_dir="bin/raw_frames", output_dir="bin/key_frames", key_frame_indices=key_frame_indices)
    print("Extracted key frames successfully.")

    print("Running img2img...")
    key_frame_img2img(input_dir="bin/key_frames",
                      key_frame_indices=key_frame_indices,
                      prompt=args.image_prompt,
                      negative_prompt="bad anatomy, poorly drawn hands, extra limbs",
                      strength=0.3, guidance_scale=50)
    print("Running img2img successfully.")
    
    print("Running ebsynth...")
    ebsynth_frame(input_dir="bin/raw_frames", output_dir="bin/refined_frames", style_dir="bin/refined_key_frames")
    print("Running ebsynth successfully.")
    print("Finished!")
        
    
