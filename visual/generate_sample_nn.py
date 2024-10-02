import torch
import numpy as np
import imageio
from torch import nn as nnn

def generate_sample_nn(H, data, sampler, shape, ema_imle, fname, logprint, preprocess_fn):
    mb = H.num_rows_visualize
    batches = []
    n_rows = mb
    temp_latent_rnds = torch.randn([mb, H.latent_dim], dtype=torch.float32).cuda()
    tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
    temp_latent_rnds = torch.randn([mb, H.latent_dim], dtype=torch.float32).cuda()
    tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
    out = ema_imle(temp_latent_rnds, tmp_snoise)
    batches.append(out)
    to_s = []
    nns = []
    for b in batches:
        for i in range(mb):
            to_s.append((b[i:i+1], None, np.inf))
    print(data.shape, len(to_s))
    loss = nnn.MSELoss()
    for i in range(data.shape[0]):
        x = data[i:i+1]
        _, target = preprocess_fn([x])
        for j, x in enumerate(to_s):
            d = x[0]
            # cur = sampler.calc_loss(target.permute(0, 3, 1, 2).cuda(), d.cuda()).item()
            cur = loss(target.permute(0, 3, 1, 2).cuda(), d.cuda()).item()
            if cur < x[2]:
                to_s[j] = (d, target, cur)

    for a in to_s:
        nn = sampler.sample_from_out(a[0].cpu())
        nns.append(nn)
    for a in to_s:
        real = sampler.sample_from_out(a[1].permute(0, 3, 1, 2).cpu())
        nns.append(real)
        
    print(len(nns))
    batches = nns
    n_rows = 2
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
