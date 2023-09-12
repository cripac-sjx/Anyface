import os
from argparse import Namespace
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import time
import clip
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")
from mapper.MemoryBlock import MemoryBlock
from mapper.datasets.textdatasets import TextDataset

text_dataset = TextDataset
from mapper.training.train_utils import convert_s_tensor_to_list

from mapper.datasets.latents_dataset import LatentsDataset, StyleSpaceLatentsDataset

from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper
import dnnlib
import legacy
import pickle

def run(test_opts):
    memory=MemoryBlock(test_opts.memory_path)
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)
    with open('models/ffhq.pkl','rb') as f:
        G_net=pickle.load(f)['G_ema'].cuda()
    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = StyleCLIPMapper(opts)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    net.cuda()
    device = 'cuda:0'
    clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)
    # upsample = torch.nn.Upsample(scale_factor=7)
    # avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    # Initialize dataset
    test_dataset = text_dataset(opts.datapath, 'test')

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=test_opts.test_batch_size,
                                 shuffle=False,
                                 num_workers=int(test_opts.test_workers),
                                 drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(test_dataset)

    global_i = 0
    global_time = []
    for input_batch in tqdm(test_dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            text, w_text, key, w_key, sent_ix = input_batch
            text = clip.tokenize(text).cuda()
            text_features = clip_model.encode_text(text)
            # input_cuda = input_cuda.cuda()

            tic = time.time()
            image_feat=memory(text_features.unsqueeze(1).repeat(1,26,1),text_features.unsqueeze(1).repeat(1,26,1))
            image_features=image_feat['memory']
            result_batch = run_on_batch(image_features, net,G_net, opts.couple_outputs, opts.work_in_stylespace)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            # im_path = str(global_i).zfill(5)
            im_path = os.path.join(out_path_results, str(int(key[i])) + '.jpg')
            if test_opts.couple_outputs:
                couple_output = torch.cat([result_batch[2][i].unsqueeze(0), result_batch[0][i].unsqueeze(0)])
                torchvision.utils.save_image(couple_output, im_path, normalize=True, range=(-1, 1))
            else:
                torchvision.utils.save_image(result_batch[i], im_path, normalize=True, range=(-1, 1))
            # torch.save(result_batch[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))

            global_i += 1
    #
    # stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)


# with open(stats_path, 'w') as f:
# 	f.write(result_str)


def run_on_batch(text, net,G_net, couple_outputs=False, stylespace=False):
    with torch.no_grad():
        w_hat, _ = net.I_map(text.float())
        noise = torch.randn(len(text), 512).cuda()
        w_z = G_net.mapping(noise, c=None, truncation_psi=0.5, truncation_cutoff=8)
        # w_z = w_z.unsqueeze(dim=1).repeat(1, 4, 1)
        w_z_t = torch.cat((w_z[:,:2,:], w_hat[:, 2:, :]), 1)
        # w_z_t=w_hat
        x_hat=G_net.synthesis(w_z_t, noise_mode='const', force_fp32=True)
        # x_hat, w_hat, _ = net.decoder([w_z_t], input_is_latent=True, return_latents=True,
        #                               randomize_noise=False, truncation=1)
        result_batch = x_hat
        # if couple_outputs:
        #     x, _ = net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1,
        #                        input_is_stylespace=stylespace)
        #     result_batch = (x_hat, w_hat, x)
    return result_batch


if __name__ == '__main__':
    test_opts = TestOptions().parse()
    run(test_opts)
