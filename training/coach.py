import os
import warnings
warnings.filterwarnings("ignore")
import math
import pdb
import clip
import torch
import torchvision
from torch.nn import functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import criteria.clip_loss as clip_loss
from criteria import id_loss
from mapper.datasets.textdatasets import TextDataset
# from mapper.datasets.latents_dataset import LatentsDataset, StyleSpaceLatentsDataset
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.training.ranger import Ranger
from mapper.training import train_utils
from mapper.training.train_utils import convert_s_tensor_to_list
from distance_consistency import rel_loss
text_dataset=TextDataset
class Coach:
	def __init__(self, opts):
		self.opts = opts
		self.neg=8
		self.global_step = 0
		self.device = 'cuda:0'
		self.opts.device = self.device

		# Initialize network
		self.net = StyleCLIPMapper(self.opts).to(self.device)

		# Initialize loss
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss(self.opts).to(self.device).eval()
		if self.opts.clip_lambda > 0:
			self.clip_loss = clip_loss.CLIPLoss(opts)
		if self.opts.latent_l2_lambda > 0:
			self.latent_l2_loss = nn.MSELoss().to(self.device).eval()
			# self.latent_l2_loss=nn.TripletMarginLoss(margin=0.1)
		if self.opts.kl_lambda > 0:
			self.kl_loss = nn.KLDivLoss(reduction='batchmean')
		if self.opts.recon_lambda>0:
			self.recon_loss=nn.L1Loss()


		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		self.clip_model, preprocess = clip.load("ViT-B/32", device=self.device)
		self.upsample = torch.nn.Upsample(scale_factor=7)
		self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

		# Initialize dataset
		self.latents=torch.load(self.opts.latents_path)
		self.train_dataset=text_dataset('./data/', 'train')
		self.test_dataset = text_dataset(self.opts.datapath, 'test')
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		#self.text_inputs = torch.cat([clip.tokenize(self.opts.description)]).cuda()

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.log_dir = log_dir
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				text, wrong_text, key, key_w, sent_ix = batch
				self.w_mean = self.latents.mean(0).unsqueeze(0)
				w = self.latents[key].to(self.device)
				w_w = self.latents[key_w].to(self.device)
				text = clip.tokenize(text).to(self.device)
				wrong_text = clip.tokenize(wrong_text).to(self.device)

				with torch.no_grad():
					# x_mean, _ = self.net.decoder([w_mean], input_is_latent=True, randomize_noise=False, truncation=1,
					# 						input_is_stylespace=self.opts.work_in_stylespace)
					x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1,
											input_is_stylespace=self.opts.work_in_stylespace)
					image = self.avg_pool(self.upsample(x))
					image_features = self.clip_model.encode_image(image)
					x_w, _ = self.net.decoder([w_w], input_is_latent=True, randomize_noise=False, truncation=1,
											  input_is_stylespace=self.opts.work_in_stylespace)
					image_w = self.avg_pool(self.upsample(x_w))
					# wrong_image_features = self.clip_model.encode_image(image_w)
					text_features = self.clip_model.encode_text(text)
					wrong_text_features = self.clip_model.encode_text(wrong_text)

				w_i,feat_i = self.net.I_map(image_features.float())
				# w_i_w, feat_i_w=self.net.I_map(wrong_image_features.float())
				w_t,feat_t = self.net.T_map(text_features.float())
				w_t_w,feat_t_w = self.net.T_map(wrong_text_features.float())

				# Noise input
				with torch.no_grad():
					noise = torch.randn(len(text), 512).cuda()
					w_z = self.net.decoder.style(noise)
					w_z = w_z.unsqueeze(dim=1).repeat(1, 4, 1)
					w_z_t = torch.cat((w_z,w_t[:, 4:, :]), 1)
					# w_z_i = torch.cat((w_z,w_i[:, 4:, :]), 1)
					x_t, w_z_t, _ = self.net.decoder([w_z_t], input_is_latent=True, return_latents=True,
												   randomize_noise=False,
												   truncation=1)
					x_i, w_i, _ = self.net.decoder([w_i], input_is_latent=True, return_latents=True,
													 randomize_noise=False,
													 truncation=1)
				# if self.global_step>3000:
				# 	self.opts.kl_lambda=0
				loss, t_loss_dict = self.calc_loss(w, x, w_t, w_t_w, self.w_mean, x_t, text)

				if self.opts.kl_lambda>0:
					loss_i, i_loss_dict = self.calc_imgloss(w, x, w_i, x_i)
					loss_dict = dict(i_loss_dict, **t_loss_dict)
					loss_kl = (self.kl_loss(F.log_softmax(feat_t, dim=1), F.softmax(feat_i, dim=1)) + self.kl_loss(
						F.log_softmax(feat_i, dim=1), F.softmax(feat_t, dim=1))) / 2
					loss+=loss_i
					loss+=loss_kl*self.opts.kl_lambda
					loss_dict['loss_kl'] = float(loss_kl)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 1000 == 0):
					self.parse_and_log_images(x, x_t, title='images_train')

				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				val_loss_dict = None
				# if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
				# 	val_loss_dict = self.validate()
				# 	if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
				# 		self.best_val_loss = val_loss_dict['loss']
				# 		self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			if batch_idx > 100:
				break
			text, wrong_text, key, key_w, sent_ix = batch
			w = self.latents[key].to(self.device)
			text = clip.tokenize(text).to(self.device)
			wrong_text = clip.tokenize(wrong_text).to(self.device)

			with torch.no_grad():
				x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1,
										input_is_stylespace=self.opts.work_in_stylespace)
				image = self.avg_pool(self.upsample(x))
				image_features = self.clip_model.encode_image(image)

				text_features = self.clip_model.encode_text(text)
				wrong_text_features=self.clip_model.encode_text(wrong_text)

				# noise = torch.randn(len(text), 512).cuda()
				# text_noise = torch.cat((text_features.float(), noise), 1)
				# w_t = self.net.T_map(text_noise)

				w_t,_ = self.net.T_map(text_features.float())
				w_t_w,_=self.net.T_map(wrong_text_features.float())


				noise = torch.randn(len(text), 512).cuda()
				w_z = self.net.decoder.style(noise)
				w_z = w_z.unsqueeze(dim=1).repeat(1, 4, 1)
				w_z_t = torch.cat((w_z, w_t[:, 4:, :]), 1)

				x_t, w_z_t, _ = self.net.decoder([w_z_t], input_is_latent=True, return_latents=True, randomize_noise=False,
											   truncation=1)
				loss, cur_loss_dict = self.calc_loss(w, x, w_t, w_t_w.repeat(self.neg,1,1), self.w_mean, x_t,text)

			agg_loss_dict.append(cur_loss_dict)
			self.parse_and_log_images(x, x_t, title='images_val', index=batch_idx)

			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		params = list(self.net.T_map.mapping.parameters())+list(self.net.I_map.mapping.parameters())

		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def calc_loss(self, w, x, w_hat, w_hat_w, w_mean, x_hat,text):
		loss_dict = {}
		loss = 0.0
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement = self.id_loss(x_hat, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.clip_lambda > 0:
			loss_clip = self.clip_loss(x_hat, text).mean()
			loss_dict['loss_clip'] = float(loss_clip)
			loss += loss_clip * self.opts.clip_lambda
		if self.opts.latent_l2_lambda > 0:
			# loss_l2_latent=self.latent_l2_loss(w,w_hat)
			loss_l2_latent=0
			# for i in range(len(w)):
			# 	pos=self.latent_l2_loss(w_hat[i],w[i])/self.latent_l2_loss(w_hat[i],w_mean[0])
			# 	neg=self.latent_l2_loss(w_hat_w[i],w[i])/self.latent_l2_loss(w_hat_w[i],w_mean[0])
			# 	loss_l2_latent+=torch.max(pos-neg+0.1, torch.tensor(0.0).to(self.device))
			# loss_l2_latent/=len(w)
			for i in range(len(w)):
				pos = torch.sum(1 - torch.cosine_similarity(w[i], w_hat[i])) / torch.sum \
					(1 - torch.cosine_similarity(w_hat[i], w_mean[0]))
				neg=0
				for j in range(self.neg):
					neg+= torch.sum(1 - torch.cosine_similarity(w[i], w_hat_w[j])) / torch.sum(
						1 - torch.cosine_similarity(w_hat_w[i], w_mean[0]))
				neg/=self.neg
				loss_l2_latent += torch.max(pos - neg + 0.2, torch.tensor(0.0).to(self.device))
			loss_l2_latent/=len(w)
			loss_dict['loss_T_latent'] = float(loss_l2_latent)
			loss += loss_l2_latent * self.opts.latent_l2_lambda
		loss_dict['loss'] = float(loss)
		return loss, loss_dict

	def calc_imgloss(self, w, x, w_hat, x_hat):
		loss_dict = {}
		loss = 0.0
		if self.opts.recon_lambda>0:
			loss_recon=self.recon_loss(x,x_hat)
			loss_dict['loss_recon']=float(loss_recon)
			loss+=loss_recon*self.opts.recon_lambda
		if self.opts.latent_l2_lambda > 0:
			# loss_l2_latent=self.latent_l2_loss(w,w_hat)
			loss_l2_latent = self.latent_l2_loss(w, w_hat)
			loss_dict['loss_I_latent'] = float(loss_l2_latent)
			loss += loss_l2_latent
		loss_dict['loss'] = float(loss)
		return loss, loss_dict

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			#pass
			print(f"step: {self.global_step} \t metric: {prefix}/{key} \t value: {value}")
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, x, x_hat, title, index=None):
		if index is None:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}.jpg')
		else:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu()]), path,
									 normalize=True, scale_each=True, range=(-1, 1), nrow=self.opts.batch_size)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict