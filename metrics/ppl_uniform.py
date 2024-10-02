import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
# from LPNet import LPNet
from collections import defaultdict

from models import parse_layer_string
import pandas as pd


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


def calc_ppl_uniform(args, g, sampler):
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

    steps = np.arange(0, 1+args.step, args.step)
    print(steps)
    output_dists = defaultdict(list)

    blocks = parse_layer_string(args.dec_blocks)
    res = sorted(set([s[0] for s in blocks if s[0] <= args.max_hierarchy]))

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

            for i in range(len(steps) - 1):
                prev_step, step = steps[i], steps[i + 1]

                latent_t0, latent_t1 = latent[::2], latent[1::2]
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None] + prev_step)
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + step)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

                if args.ppl_snoise > 0:
                    for j in range(len(snoise)):
                        snoise_t0, snoise_t1 = snoise[j][::2], snoise[j][1::2]
                        snoise_e0 = lerp(snoise_t0, snoise_t1, lerp_t[:, None, None, None] + prev_step)
                        snoise_e1 = lerp(snoise_t0, snoise_t1, lerp_t[:, None, None, None] + args.eps)
                        snoise_p = torch.stack([snoise_e0, snoise_e1], 1).view(*snoise[j].shape)
                        snoise_e[j] = snoise_p

                if args.ppl_snoise == 0:
                    for j in range(len(snoise)):
                        snoise[j][::2] = snoise[j][1::2]
                    snoise_e = snoise

                image = g(latent_e, snoise, input_is_w=args.space=="w")

                if args.crop:
                    c = image.shape[2] // 8
                    image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

                if i==0:
                    start = image[::2]
                if i==len(steps)-2:
                    end = image[1::2]
                    endpoint_dist = percept(start, end, use_mean=False).view(image.shape[0] // 2) / (
                            args.step ** 2
                    )
                    output_dists[str(i+1)].append(endpoint_dist.to("cpu").numpy())


                factor = image.shape[2] // 256

                if factor > 1:
                    image = F.interpolate(
                        image, size=(256, 256), mode="bilinear", align_corners=False
                    )

                dist = percept(image[::2], image[1::2], use_mean=False).view(image.shape[0] // 2) / (
                    args.step ** 2
                )
                output_dists[str(i)].append(dist.to("cpu").numpy())

    distances = dict()
    for i in range(len(steps)):
        temp = np.concatenate(output_dists[str(i)], 0)

        # lo = np.percentile(temp, 1, interpolation="lower")
        # hi = np.percentile(temp, 99, interpolation="higher")
        # filtered_dist = np.extract(
        #     np.logical_and(lo <= temp, temp <= hi), temp
        # )
        distances[str(i)] = temp
    # print(distances)
    distances['endpoint'] = distances[f"{len(steps)-1}"]
    del distances[f"{len(steps)-1}"]
    # output_dists.append(filtered_dist.mean())
    dist_df = pd.DataFrame(distances, index=list(range(len(distances['0']))))
    # print(f"ppls: {dist_df}")
    means = dist_df.drop(columns=['endpoint']).mean(axis=1)
    stds = dist_df.drop(columns=['endpoint']).std(axis=1)
    assert len(stds) == len(distances['0'])
    # print(sum(stds)/len(stds))
    print(means.shape)
    print("Mean: ", means.mean())
    print("Std.Dev: ", stds.mean())
    print("Endpoint Mean: ", dist_df['endpoint'].mean())
    # ckpt_num = int(args.ckpt.split("/")[-1][:-3])
    # save_dir = "/".join(args.ckpt.split("/")[:-2])+f"/ppl_uniform_at_{ckpt_num}.csv"
    dist_df.to_csv(args.save_dir + f'/{args.ppl_save_name}.csv')
    # output_dists = [dist.astype(np.float64) for dist in distances.items()]

    # print("ppl:", filtered_dist.mean())
