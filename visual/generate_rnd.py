import torch
import numpy as np
import imageio

def generate_rnd(H, sampler, shape, ema_imle, fname, logprint):
    mb = H.num_rows_visualize
    batches = []
    n_rows = mb
    temp_latent_rnds = torch.randn([mb, H.latent_dim], dtype=torch.float32).cuda()
    for t in range(H.num_rows_visualize):
        temp_latent_rnds.normal_()
        if(H.use_snoise == True):
            tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
        else:
            tmp_snoise = [s[:mb] for s in sampler.neutral_snoise]
        # tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
        out = ema_imle(temp_latent_rnds, tmp_snoise)
        batches.append(sampler.sample_from_out(out))

    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
