import torch
import torch.utils.data as data
import pdb
import os
import numpy.random as random
from PIL import Image
from torchvision import transforms

class TextDataset(data.Dataset):

    def __init__(self, data_dir, split):
        self.embeddings_num = 10
        self.split=split

        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split+'_caps2')
        self.filenames=os.listdir(split_dir)
        self.text_dict=dict()

        for textfile in self.filenames:
            filename=os.path.join(split_dir,textfile)
            f=open(filename,'r')
            lines = [line.strip() for line in f.readlines()]
            self.text_dict[textfile.split('.')[0]]=lines

    def __getitem__(self, index):
        wrong_index=random.randint(0,len(self.filenames))
        key = self.filenames[index].split('.')[0]
        wrong_key = self.filenames[wrong_index].split('.')[0]
        sent_ix=int(key)%10
        # sent_ix = random.randint(0, self.embeddings_num)
        text=self.text_dict[key][sent_ix]
        wrong_text=self.text_dict[wrong_key][sent_ix]

        #latent=self.latents[int(key)]

        return text,wrong_text,int(key), int(wrong_key),sent_ix

    def __len__(self):
        return len(self.filenames)

class TextDataset(data.Dataset):

    def __init__(self, data_dir, split):
        self.embeddings_num = 10
        self.split=split

        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split+'_caps2')
        self.filenames=os.listdir(split_dir)
        self.text_dict=dict()
        self.file_dir=[]

        for textfile in self.filenames:
            filename=os.path.join(split_dir,textfile)
            f=open(filename,'r')
            lines = [line.strip() for line in f.readlines()]
            self.text_dict[textfile.split('.')[0]]=lines
            self.file_dir.append(filename)

    def __getitem__(self, index):
        wrong_index=random.randint(0,len(self.filenames))
        key = self.filenames[index].split('.')[0]
        wrong_key = self.filenames[wrong_index].split('.')[0]
        sent_ix=int(key)%10
        # sent_ix = random.randint(0, self.embeddings_num)
        text=self.text_dict[key][sent_ix]
        wrong_text=self.text_dict[wrong_key][sent_ix]
        file=self.file_dir[index]
        #latent=self.latents[int(key)]

        return text,wrong_text,int(key), int(wrong_key),sent_ix

    def __len__(self):
        return len(self.filenames)


class ImageDataset(data.Dataset):

    def __init__(self, data_dir):
        filename=os.path.join(data_dir,'face.txt')
        f=open(filename)
        files=f.readlines()
        self.file_dir=[]
        self.latent_dir=[]
        self.transform = transforms.Compose([
            transforms.ToTensor()])

        for textfile in files:
            filename=os.path.join(data_dir,textfile.split('\n')[0])
            self.file_dir.append(filename)
            self.latent_dir.append(filename.replace('data_all','latents_all').replace('.png','.pt').replace('.jpg','.pt'))

    def __getitem__(self, index):
        wrong_index=random.randint(0,len(self.file_dir))
        # image=Image.open(self.file_dir[index]).convert('RGB')
        # wrong_image=Image.open(self.file_dir[wrong_index]).convert('RGB')
        latent_path=self.latent_dir[index]
        wrong_latent_path=self.latent_dir[wrong_index]
        # latent=torch.load(self.latent_dir[index])
        # wrong_latent=torch.load(self.latent_dir[wrong_index])
        # image=self.transform(image)
        # wrong_image=self.transform(wrong_image)

        return latent_path,wrong_latent_path

    def __len__(self):
        return len(self.file_dir)

