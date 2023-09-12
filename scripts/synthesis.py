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
from mapper_all.MemoryBlock import MemoryBlock
from mapper_all.datasets.textdatasets import TextDataset

text_dataset = TextDataset
from mapper_all.training.train_utils import convert_s_tensor_to_list

from mapper_all.datasets.latents_dataset import LatentsDataset, StyleSpaceLatentsDataset

from mapper_all.options.test_options import TestOptions
from mapper_all.styleclip_mapper import StyleCLIPMapper
import dnnlib
import legacy
import pickle


def run(test_opts):
    memory=MemoryBlock(test_opts.memory_path)
    out_path_results = os.path.join(test_opts.exp_dir, 'synthesis_results')
    os.makedirs(out_path_results, exist_ok=True)
    with open('../models/ffhq.pkl','rb') as f:
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

    text=opts.descrip
    im_path=text.replace(' ','')
    text = clip.tokenize(text).cuda()
    text_features = clip_model.encode_text(text)
    image_feat=memory(text_features.unsqueeze(1).repeat(1,26,1),text_features.unsqueeze(1).repeat(1,26,1))
    image_features=image_feat['memory']
    for i in range(100):
        result_batch = run_on_batch(image_features, net, G_net)
        torchvision.utils.save_image(result_batch, os.path.join(out_path_results, im_path+str(i)+'.jpg'),
                                 normalize=True, range=(-1, 1))


def run_on_batch(text, net, G_net):
    with torch.no_grad():
        w_hat, _ = net.I_map(text.float())
        noise = torch.randn(len(text), 512).cuda()
        w_z = G_net.mapping(noise, c=None, truncation_psi=0.5, truncation_cutoff=8)
        w_z_t = torch.cat((w_z[:,:2,:], w_hat[:, 2:, :]), 1)
        x_hat=G_net.synthesis(w_z_t, noise_mode='const', force_fp32=True)
        result_batch = x_hat
    return result_batch


if __name__ == '__main__':
    test_opts = TestOptions().parse()
    run(test_opts)
