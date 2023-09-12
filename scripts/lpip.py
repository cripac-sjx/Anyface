import os
import sys
sys.path.append(".")
sys.path.append("..")

import lpips
import time
import random
import argparse
import numpy as np
import pprint
import datetime
import dateutil.tz
from PIL import Image
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import warnings
from torch import nn
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Caculate the R precision')
    parser.add_argument('--textpath',default='data/test_caps/', type=str, help='Path to dataset')
    parser.add_argument('--imgpath',default='data/CelebAimg/', type=str, help='Path to dataset')
    parser.add_argument('--Attn_path',default='../results/CelebAText/AttnGAN/valid/', type=str, help='Path to dataset')
    parser.add_argument('--Con_path',default='../results/CelebAText/ControlGAN/valid/', type=str, help='Path to dataset')
    parser.add_argument('--SEA_path',default='../results/CelebAText/SEA-T2F/valid/', type=str, help='Path to dataset')
    parser.add_argument('--Mut_path', default='../mut_0noise/inference_results/', type=str, help='Path to dataset')

    args = parser.parse_args()
    return args


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


if __name__ == "__main__":
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, datapath, Attn_path, Con_path, SEA_path,Mut_path,imgpath):
            self.img_path=imgpath
            self.datapath=datapath
            self.Attn_path=Attn_path
            self.Con_path=Con_path
            self.SEA_path=SEA_path
            self.Mut_path=Mut_path
            self.transform = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.namelist = os.listdir(self.datapath)

        def __getitem__(self, index):
            textname = self.namelist[index]
            imgname=textname.split('.')[0]+'.png'
            imgname2 = textname.split('.')[0] + '.jpg'
            text_path=self.datapath+textname
            f=open(text_path)
            line=f.readlines()[int(textname.split('.')[0])%10]

            img_path=self.img_path+imgname2
            img=Image.open(img_path).convert('RGB')
            img = self.transform(img)

            attn_path=self.Attn_path+imgname
            attn_img=Image.open(attn_path).convert('RGB')
            attn_img = self.transform(attn_img)
            con_path=self.Con_path+imgname
            con_img=Image.open(con_path).convert('RGB')
            con_img = self.transform(con_img)
            sea_path=self.SEA_path+imgname
            sea_img=Image.open(sea_path).convert('RGB')
            sea_img = self.transform(sea_img)

            mut_path=self.Mut_path+imgname2
            mut_img=Image.open(mut_path).convert('RGB')
            mut_img = self.transform(mut_img)
            return line,attn_img,con_img,sea_img,mut_img, img

        def __len__(self):
            return len(self.namelist)
    device = 'cuda:0'
    args = parse_args()

    imgs = IgnoreLabelDataset(args.textpath,args.Attn_path, args.Con_path, args.SEA_path, args.Mut_path, args.imgpath)

    test_dataloader = torch.utils.data.DataLoader(
        imgs, batch_size=10)

    P_rates = [0]*4
    loss_fn = lpips.LPIPS(net='alex')
    for step, test_data in enumerate(test_dataloader, 0):

        line, img0,img1,img2,img5, ori_img=test_data
        imgs=[img0,img1,img2,img5]
        for j in range(len(imgs)):
            P_rates[j]+=torch.sum(loss_fn.forward(imgs[j],ori_img)) .detach()
        print(step)
        print(P_rates)
    print('Finished!')
    print(P_rates)
