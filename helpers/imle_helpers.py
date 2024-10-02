import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import imageio
from visual.utils import get_sample_for_visualization, generate_for_NN, generate_images_initial
from torch.utils.data import DataLoader, TensorDataset
from helpers.utils import ZippedDataset, get_cpu_stats_over_ranks


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma, eps):
    return torch.exp(logsigma) * eps + mu


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def discretized_mix_logistic_loss(x, l, low_bit=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10)  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = torch.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = torch.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = torch.cat([torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(x < -0.999,
                                log_cdf_plus,
                                torch.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)
    res = -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])
    return res


def sample_from_discretized_mix_logistic(l, nr_mix, eps=None, u=None):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    if eps is None:
        eps = torch.empty(logit_probs.shape, device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    log_scales = const_max((l[:, :, :, :, nr_mix:nr_mix * 2] * sel).sum(dim=4), -7.)
    coeffs = (torch.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    if u is None:
        u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return torch.cat([torch.reshape(x0, xs[:-1] + [1]), torch.reshape(x1, xs[:-1] + [1]), torch.reshape(x2, xs[:-1] + [1])], dim=3), eps, u


def backtrack(H, sampler, imle, preprocess_fn, data, logprint, training_step_imle):
    latents = torch.randn([data.shape[0], H.latent_dim], requires_grad=True, dtype=torch.float32, device='cuda')
    snoise = [torch.randn([data.shape[0], s.shape[1], s.shape[2], s.shape[3]], dtype=torch.float32, device='cuda') for s in sampler.snoise_tmp]

    if H.restore_latent_path:
        logprint('restoring latent path')
        latents = torch.tensor(torch.load(f'{H.restore_latent_path}/latent-best.npy'), requires_grad=True, dtype=torch.float32, device='cuda')
        snoise = [torch.tensor(torch.load(f'{H.restore_latent_path}/snoise-best-{s.shape[2]}.npy'), requires_grad=True, dtype=torch.float32, device='cuda') for s in sampler.snoise_tmp]

    latent_optimizer = AdamW([latents], lr=H.latent_lr)
    if H.space == 'w':
        latent_optimizer = AdamW([latents] + snoise, lr=H.latent_lr)
    # latent_optimizer = SGD([latents] + snoise, lr=H.latent_lr)
    dists = torch.empty([data.shape[0]], dtype=torch.float32).cuda()

    sampler.calc_dists_existing(data, imle, dists=dists, latents=latents, snoise=snoise)
    print(f'initial dists: {dists.mean()}')

    best_loss = np.inf
    num_iters = 0

    while num_iters < H.reconstruct_iter_num:
        comb_dataset = ZippedDataset(data, TensorDataset(latents))
        data_loader = DataLoader(comb_dataset, batch_size=H.n_batch)
        for cur, indices in data_loader:
            x = cur
            lat = cur[1][0]
            _, target = preprocess_fn(x)
            cur_snoise = [s[indices] for s in snoise]
            training_step_imle(H, target.shape[0], target, lat, cur_snoise, imle, None, latent_optimizer, sampler.calc_loss)
            latents.grad.zero_()
            [s.grad.zero_() for s in snoise]
        num_iters += len(data)

        logprint(f'iteration: {num_iters}')
        # torch.save(latents.detach(), f'{H.save_dir}/latent-latest.npy')
        # for s in snoise:
        #     torch.save(s.detach(), f'{H.save_dir}/snoise-latest-{s.shape[2]}.npy')

        sampler.calc_dists_existing(data, imle, dists=dists, latents=latents, snoise=snoise)
        cur_mean = dists.mean()
        logprint(f'cur mean: {cur_mean}, best: {best_loss}')
        if cur_mean < best_loss:
            torch.save(latents.detach(), f'{H.save_dir}/latent-best.npy')
            for s in snoise:
                torch.save(s.detach(), f'{H.save_dir}/snoise-best-{s.shape[2]}.npy')
            logprint(f'improved: {cur_mean}')
            best_loss = cur_mean
            for i in range(data.shape[0]):
                samp = sampler.sample(latents[i:i+1], imle, [s[i:i+1] for s in snoise])
                imageio.imwrite(f'{H.save_dir}/{i}.png', samp[0])
                imageio.imwrite(f'{H.save_dir}/{i}-real.png', data[i])

        if num_iters >= H.reconstruct_iter_num:
            break


def reconstruct(H, sampler, imle, preprocess_fn, images, latents, snoise, name, logprint, training_step_imle):
    latent_optimizer = AdamW([latents], lr=H.latent_lr)
    generate_for_NN(sampler, images, latents.detach(), snoise, images.shape, imle,
                    f'{H.save_dir}/{name}-initial.png', logprint)
    for i in range(H.latent_epoch):
        for iter in range(H.reconstruct_iter_num):
            _, target = preprocess_fn([images])
            stat = training_step_imle(H, target.shape[0], target, latents, snoise, imle, None, latent_optimizer, sampler.calc_loss)

            latents.grad.zero_()
            if iter % 50 == 0:
                print('loss is: ', stat['loss'])
                generate_for_NN(sampler, images, latents.detach(), snoise, images.shape, imle,
                                f'{H.save_dir}/{name}-{iter}.png', logprint)

                torch.save(latents.detach(), '{}/reconstruct-latest.npy'.format(H.save_dir))