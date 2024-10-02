import torch
import numpy as np
import imageio

def generate_rnd_nn(H, data, sampler, shape, imle, fname, logprint, preprocess_fn):
    mb = 10
    batches = []
    temp_latent_rnds = torch.randn([mb, H.latent_dim], dtype=torch.float32).cuda()
    for t in range(H.num_rows_visualize):
        temp_latent_rnds.normal_()
        tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
        out = imle(temp_latent_rnds, tmp_snoise)
        batches.append(out)
    to_s = []
    nns = []
    nns_pairs = []
    for b in batches:
        for i in range(mb):
            to_s.append(b[i:i+1])
    print(len(to_s))
    for i in range(data.shape[0]):
        x = data[i:i+1]
        _, target = preprocess_fn([x])
        bst_loss = np.inf
        bst_ind = -1
        for j, d in enumerate(to_s):
            cur = sampler.calc_loss(target.permute(0, 3, 1, 2).cuda(), d.cuda()).item()
            if cur < bst_loss:
                bst_loss = cur
                bst_ind = j
        real = sampler.sample_from_out(target.permute(0, 3, 1, 2).cpu())
        nn = sampler.sample_from_out(to_s[bst_ind].cpu())
        nns_pairs.append((bst_loss, real, nn, bst_ind))

    print(len(nns))
    nns_pairs = sorted(nns_pairs)[::-1]
    for a in nns_pairs:
        nns.append(a[1])
        nns.append(a[2])
    batches = nns
    mb = 10
    n_rows = 20
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])
    logprint(f'printing samples to {fname}/rnd-nn.png')
    imageio.imwrite(f'{fname}/rnd-nn.png', im)

    # used = [x[3] for x in nns_pairs]
    # others = [(np.inf, x, None) for i, x in enumerate(to_s) if i not in used]
    # print('others', len(others))
    # for i in range(data.shape[0]):
    #     x = data[i:i+1]
    #     _, target = preprocess_fn([x])
    #     for j, x in enumerate(others):
    #         d = x[1]
    #         cur = sampler.calc_loss(target.permute(0, 3, 1, 2).cuda(), d.cuda()).item()
    #         if cur < x[0]:
    #             others[j] = (cur, d, target)
    # others = sorted(others)[::-1]

    # for i in range(len(others)//10):
    #     nns = []
    #     for a in others[i*10:(i+1)*10]:
    #         nn = sampler.sample_from_out(a[1].cpu())
    #         nns.append(nn)
    #     for a in others[i*10:(i+1)*10]:
    #         real = sampler.sample_from_out(a[2].permute(0, 3, 1, 2).cpu())
    #         nns.append(real)

    #     print(len(nns))
    #     batches = nns
    #     mb = 10
    #     n_rows = 2
    #     im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
    #         [n_rows * shape[1], mb * shape[2], 3])
    #     logprint(f'printing samples to {fname}')
    #     imageio.imwrite(f'{fname}/rnd-nn-rem-{i}.png', im)
        