import torch
from make_a_video_pytorch import PseudoConv3d, SpatioTemporalAttention, SpaceTimeUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conv = PseudoConv3d(
    dim = 256,
    kernel_size = 3
).to(device)

attn = SpatioTemporalAttention(
    dim = 256,
    dim_head = 64,
    heads = 8
).to(device)

images = torch.randn(1, 256, 16, 16).to(device) # (batch, features, height, width)

conv_out = conv(images) # (1, 256, 16, 16)
attn_out = attn(images) # (1, 256, 16, 16)

unet = SpaceTimeUnet(
    dim = 64,
    channels = 3,
    dim_mult = (1, 2, 4, 8),
    resnet_block_depths = (1, 1, 1, 2),
    temporal_compression = (False, False, False, True),
    self_attns = (False, False, False, True),
    condition_on_timestep = False,
    attn_pos_bias = False,
    flash_attn = True
).to(device)

# train on images

images = torch.randn(1, 3, 128, 128).to(device)
images_out  = unet(images)

assert images.shape == images_out.shape

# then train on videos

video = torch.randn(1, 3, 16, 128, 128).to(device)
video_out = unet(video)

assert video_out.shape == video.shape

# or even treat your videos as images

video_as_images_out = unet(video, enable_time = False)
print("Success!")