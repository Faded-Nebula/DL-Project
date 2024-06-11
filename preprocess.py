from PIL import Image
import os
import imageio
import re
import glob

def get_image_resolution(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        print(f'The image resolution is {width}x{height}')
    
def resize_image(image_path, output_path, new_size=(1024, 1024)):
    with Image.open(image_path) as img:
        resized_img = img.resize(new_size)
        resized_img.save(output_path)

def key_frame_extraction(input_dir="bin/raw_frames", output_dir="bin/key_frames", key_frame_indices=[4, 8, 12, 16]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        files = glob.glob(os.path.join(output_dir, '*'))
        for f in files:
            os.remove(f)
    for i in key_frame_indices:
        img_path = os.path.join(input_dir, f"frame_{i}.png")
        img = Image.open(img_path)
        img.save(os.path.join(output_dir, f"frame_{i}.png"))

def stitch_images(input_dir, output_image, image_size=(512, 512), grid_size=(2, 2)):
    # 创建一个新的空白图像，大小为image_size * grid_size
    new_img = Image.new('RGB', (image_size[0]*grid_size[0], image_size[1]*grid_size[1]))

    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    files.sort(key=lambda f: int(re.search(r'frame_(\d+).png', f).group(1)))

    # 遍历输入目录中的所有图像
    for i, file_name in enumerate(files):
        img_path = os.path.join(input_dir, file_name)
        # 加载图像并粘贴到正确的位置
        img = Image.open(img_path)
        x = i % grid_size[0] * image_size[0]
        y = i // grid_size[0] * image_size[1]
        new_img.paste(img, (x, y))

    # 保存新图像
    new_img.save(output_image)

def images_to_gif(input_dir, output_gif):
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    files.sort(key=lambda f: int(re.search(r'frame_(\d+).png', f).group(1)))
    images = []
    for i, filename in enumerate(files):
        img_path = os.path.join(input_dir, filename)
        images.append(imageio.imread(img_path))
    imageio.mimsave(output_gif, images, duration=1)

def gif_to_images(input_gif, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        files = glob.glob(os.path.join(output_dir, '*'))
        for f in files:
            os.remove(f)
    with imageio.get_reader(input_gif) as reader:
        for i, img in enumerate(reader):
            imageio.imsave(os.path.join(output_dir, f"frame_{i + 1}.png"), img)

def split_image(image_path, output_dir, grid_size=(2, 2),frame_indices=[4, 8, 12, 16]):
    assert len(frame_indices) <= grid_size[0] * grid_size[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = Image.open(image_path)
    width, height = img.size
    grid_width = width // grid_size[0]
    grid_height = height // grid_size[1]
    k = 0
    for j in range(grid_size[1]):
        for i in range(grid_size[0]):

            if k >= len(frame_indices):
                break

            left = i * grid_width
            upper = j * grid_height
            right = (i + 1) * grid_width
            lower = (j + 1) * grid_height
            cropped_img = img.crop((left, upper, right, lower))
            cropped_img.save(f"{output_dir}/frame_{frame_indices[k]}.png")
            resize_image(f"{output_dir}/frame_{frame_indices[k]}.png", f"{output_dir}/frame_{frame_indices[k]}.png", (1024, 1024))
            k +=1


def convert_to_grayscale(input_img, output_img):
    image = Image.open(input_img)
    grayscale_image = image.convert("L")
    grayscale_image.save(output_img)