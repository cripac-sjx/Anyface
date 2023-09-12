from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--num_caps', default=10, type=int, help='Number of captions')
		self.parser.add_argument('--ref_image', default='scripts/00415.png', type=str, help='Path of reference image')
		self.parser.add_argument('--descrip', default=None, type=str, help='Text description')
		self.parser.add_argument('--degree', default=4, type=int, help='The degree to be preserved')
		self.parser.add_argument('--key', default=0, type=int, help='Image name')
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment oexputput directory')
		self.parser.add_argument('--datapath', default='../mapper/data/', type=str, help='Path to dataset')
		self.parser.add_argument('--checkpoint_path', default='checkpoints/iteration_42000.pt', type=str, help='Path to model checkpoint')
		self.parser.add_argument('--memory_path', default='checkpoints/memory_42000.pt', type=str, help='Path to model checkpoint')
		# self.parser.add_argument('--checkpoint_path', default='/mnt/workspace/workgroup/zhibing/Anyface/mapper_all/out_worcycle/checkpoints/iteration_34000.pt', type=str, help='Path to model checkpoint')
		# self.parser.add_argument('--memory_path', default='/mnt/workspace/workgroup/zhibing/Anyface/mapper_all/out_worcycle/memory/memory_34000.pt', type=str, help='Path to model checkpoint')
        # self.parser.add_argument('--memory_path', default='workgroup/zhibing/Anyface/mapper_all/out_rel10_CLIP0.1_D50_m0.15/memory/memory_50000.pt', type=str, help='Path to memory')
		self.parser.add_argument('--couple_outputs', default=False,  action='store_true', help='Whether to also save inputs + outputs side-by-side')

		self.parser.add_argument('--stylegan_size', default=1024, type=int)
		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--latents_test_path', default='../mapper/data/latents.pt', type=str, help="The latents for the validation")
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--work_in_stylespace', default=False, action='store_true')

		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')

	def parse(self):
		opts = self.parser.parse_args()
		return opts