from selectors import BaseSelector
import torch
import numpy as np
import imageio

def spatial_vissual(H, sampler, shape, imle, fname, logprint, lat1=None, lat2=None, sn1=None, sn2=None):
    num_lin = 1
    mb = 8

    batches = []
    # step = (-f_latent + s_latent) / num_lin
    base_latent = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda()
    base_snosie = [torch.randn([1, 1, s, s], dtype=torch.float32).cuda() for s in sampler.res]
    # for t in range(len(sampler.res)):
    #     snoise = [base_snosie[i] for i in range(t)] + [torch.ones([1, 1, s, s], dtype=torch.float32).cuda() for s in sampler.res[t:]]
    #     out = imle(base_latent, spatial_noise=snoise)
    #     batches.append(sampler.sample_from_out(out))

    # for t in range(10):
    #     snoise = [base_snosie[i] for i in range(len(sampler.res) - 1)] + [torch.randn([1, 1, s, s], dtype=torch.float32).cuda() for s in sampler.res[len(sampler.res) - 1:]]
    #     out = imle(base_latent, spatial_noise=snoise)
    #     batches.append(sampler.sample_from_out(out))

    # for t in range(10):
    #     snoise = [base_snosie[i] for i in range(len(sampler.res) - 1)] + [torch.randn([1, 1, s, s], dtype=torch.float32).cuda() for s in sampler.res[len(sampler.res) - 1:]]
    #     base_latent.normal_()
    #     out = imle(base_latent, spatial_noise=base_snosie)
    #     batches.append(sampler.sample_from_out(out))

    base_latent = [torch.zeros([1, H.latent_dim], dtype=torch.float32).cuda() for i in range(10)]
    base_snosie = [torch.zeros([1, 1, s, s], dtype=torch.float32).cuda() for s in sampler.res]
    out = imle(base_latent[0], spatial_noise=base_snosie)
    batches.append(sampler.sample_from_out(out))
        
    
    base_latent = [torch.randn([1, H.latent_dim], dtype=torch.float32).cuda() for i in range(10)]
    base_snosie = [torch.randn([1, 1, s, s], dtype=torch.float32).cuda() for s in sampler.res]
    for i in range(10):
        out = imle(base_latent[i], spatial_noise=base_snosie)
        batches.append(sampler.sample_from_out(out))
        
    lat_dim = base_latent[0].shape[1]
    print(lat_dim)
    for i in range(10):
        out = imle(base_latent[i], spatial_noise=base_snosie)
        batches.append(sampler.sample_from_out(out))
        lat = base_latent[i]
        for j in range(10):
            lat2 = base_latent[j]
            cur = torch.cat((lat[:, 0:lat_dim//2], lat2[:, lat_dim//2:]), dim=1)
            out = imle(cur, spatial_noise=base_snosie)
            batches.append(sampler.sample_from_out(out))

    print(len(batches))
    mb = 11
    n_rows = 11
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)