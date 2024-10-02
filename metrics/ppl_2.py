import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import lpips

# from LPNet import LPNet

from models import parse_layer_string


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def get_omega(x, y):
    return torch.acos((normalize(x) * normalize(y)).sum(1))

def slerp(x, y, t):
    omega = get_omega(x, y)[:, None]
    c1 = torch.sin(omega * (1 - t)) / torch.sin(omega)
    c2 = torch.sin(omega * t) / torch.sin(omega)
    return c1 * x + c2 * y

def lerp(a, b, t):
    return a + (b - a) * t


def calc_ppl(args, g, sampler):
    device = "cuda"

    latent_dim = args.latent_dim
    g.eval()

    # percept = sampler.calc_loss
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    distances = []

    n_sample = 1024
    eps = 1e-4

    blocks = parse_layer_string(args.dec_blocks)
    res = sorted(set([s[0] for s in blocks if s[0] <= args.max_hierarchy]))

    with torch.no_grad():
        for batch in tqdm(range(n_sample // args.n_batch)):
            z_1 = torch.randn(args.n_batch, latent_dim).to(device)
            z_2 = torch.randn(args.n_batch, latent_dim).to(device)
            # Sample num_samples points along the interpolated lines
            t = torch.rand(args.n_batch)[:, None].to(device)

            interpolated_1 = slerp(z_1, z_2, t)
            interpolated_2 = slerp(z_1, z_2, t + eps)
            # Generated the interpolated images
            y_1, y_2 = g(interpolated_1, spatial_noise=None), g(interpolated_2, spatial_noise=None)
            # Calculate the per-sample LPIPS
            cur_lpips = loss_fn_vgg(y_1, y_2) 
            cur_lpips /= (eps ** 2)
            distances.append(cur_lpips.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print(f"{args.restore_path}, ppl:", filtered_dist.mean())
