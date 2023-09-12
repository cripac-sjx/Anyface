import torch
import os
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.embeddings_num = 10
        self.split=split

        self.data_dir = data_dir
        split_dir = os.path.join(data_dir, split+'_caps')
        self.filenames=os.listdir(split_dir)
        self.number_example = len(self.filenames)
        self.text_dict=dict()
        for textfile in self.filenames:
            filename=os.path.join(split_dir,textfile)
            f=open(filenames,'r')
            lines = [line.strip() for line in f.readlines()]
            self.text_dict[filename]=lines

    def __len__(self):
        return len(filenames)
    def __getitem__(self, index):
        # key = self.filenames[index]
        # img_name = '%sCelebAimg/%s' %(self.data_dir, key)
        # img = Image.open(img_name).convert('RGB')
        # sent_ix = random.randint(0, self.embeddings_num)
        # new_sent_ix = index * self.embeddings_num + sent_ix
        # caps = self.caption([new_sent_ix])

        # return img, caps,  key
        return self.text_dict

class LatentsDataset(Dataset):

	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):

		return self.latents[index]

class StyleSpaceLatentsDataset(Dataset):

	def __init__(self, latents, opts):
		padded_latents = []
		for latent in latents:
			latent = latent.cpu()
			if latent.shape[2] == 512:
				padded_latents.append(latent)
			else:
				padding = torch.zeros((latent.shape[0], 1, 512 - latent.shape[2], 1, 1))
				padded_latent = torch.cat([latent, padding], dim=2)
				padded_latents.append(padded_latent)
		self.latents = torch.cat(padded_latents, dim=2)
		self.opts = opts

	def __len__(self):
		return len(self.latents)

	def __getitem__(self, index):
		return self.latents[index]
