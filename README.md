# Anyface++ -- Official implementation
AnyFace++: A Unified Framework for Free-style Text-to-Face Synthesis and Manipulation

Jianxin Sun, Qiyao Deng, Qi Li, Muyi Sun, Yunfan Liu, Zhenan Sun

![Teaser image](.framework.png)

# Requirements
Our code is based on the implementation of StyleGAN2, to run our code, you need to meet all the requirements of StyleGAN2 and download the “ffhq.pkl” file from the [StyleGAN2 repository](https://github.com/NVlabs/stylegan2-ada-pytorch), and put it into "./models/" Then, run:

```.bash
pip install -r requirements.txt
```

# Datasets

Download the CelebAText-HQ and Multi-modal CelebA-HQ from [SEA-T2F](https://github.com/cripac-sjx/SEA-T2F) and [TediGAN](https://github.com/IIGROUP/TediGAN).
Download the [FFText-HQ](https://drive.google.com/file/d/1ZhkrUXFkVVPHAHR0HhcYhxPuBeAENO86/view?usp=sharing) Dataset.

# Pretained Models
Down the [Memory model](https://drive.google.com/file/d/1db6VmuMVzW50uHdGwMrv4HK9c3FbaoQ4/view?usp=sharing) and [pretrained model](https://drive.google.com/file/d/1GbMavKgNVqXgWcmqxp0R6n8nmOmLfooS/view?usp=sharing) and put them into "./moels"

# Inference

```.bash
python scripts/synthesis.py --descrip "A girl with curly black hair is smiling." \
--memory_path "<path to memory model>" \
--checkpoints_path "<path to checkpoint path>" \
--exp_dir "outputs"
```

