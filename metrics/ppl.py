import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import imageio

# from LPNet import LPNet

from models import parse_layer_string


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


def calc_ppl(args, g, sampler):
    device = "cuda"

    latent_dim = args.latent_dim
    g.eval()

    percept = sampler.calc_loss

    distances = []

    n_batch = args.n_sample // args.n_batch
    resid = args.n_sample - (n_batch * args.n_batch)
    batch_sizes = [args.n_batch] * n_batch
    if resid:
        batch_sizes.append(resid)

    blocks = parse_layer_string(args.dec_blocks)
    res = sorted(set([s[0] for s in blocks if s[0] <= args.max_hierarchy]))


    num_batch = 0

    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            snoise = [torch.randn([batch * 2, 1, s, s], dtype=torch.float32).cuda() for s in res]
            snoise_e = [torch.randn([batch * 2, 1, s, s], dtype=torch.float32) for s in res]

            latent = torch.randn([batch * 2, latent_dim], device=device)
            if args.sampling == "full":
                lerp_t = torch.rand(batch, device=device)
            else:
                lerp_t = torch.zeros(batch, device=device)

            if args.space == "w":
                latent = g.module.decoder.mapping_network(latent)[0]

            # snoise_t0, snoise_t1 = [sn[::2] for sn in snoise], [sn[1::2] for sn in snoise]
            latent_t0, latent_t1 = latent[::2], latent[1::2]
            latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
            latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + args.eps)
            latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

            for i in range(len(snoise)):
                snoise_t0, snoise_t1 = snoise[i][::2], snoise[i][1::2]
                snoise_e0 = lerp(snoise_t0, snoise_t1, lerp_t[:, None, None, None])
                snoise_e1 = lerp(snoise_t0, snoise_t1, lerp_t[:, None, None, None] + args.eps)
                snoise_p = torch.stack([snoise_e0, snoise_e1], 1).view(*snoise[i].shape)
                snoise_e[i] = snoise_p
            
            if args.ppl_snoise == 0:
                for i in range(len(snoise)):
                    snoise[i][::2] = snoise[i][1::2]
                snoise_e = snoise


            image = g(latent_e, None, input_is_w=args.space=="w")
            

            if args.crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode="bilinear", align_corners=False
                )

            dist = percept(image[::2], image[1::2], use_mean=False).view(image.shape[0] // 2) / (
                args.eps ** 2
            )
            distances.append(dist.to("cpu").numpy())

            image_0 = g(latent_e0, None, input_is_w=args.space=="w")
            image_1 = g(latent_e1, None, input_is_w=args.space=="w")


            for i in range(batch):
                imageio.imwrite(
                    f"dubi/interpolated_{num_batch}_{i}.png",
                    np.concatenate(
                        [
                            image_0[i].cpu().numpy().transpose(1, 2, 0),
                            image_1[i].cpu().numpy().transpose(1, 2, 0),
                        ],
                        1,
                    ),
                )
            
            num_batch += 1

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    print(f"{args.restore_path}, ppl:", filtered_dist.mean())
