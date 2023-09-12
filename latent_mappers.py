import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable
from models.stylegan2.model import EqualLinear, PixelNorm

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 768
        self.c_dim = 512
        layers = []
        layers.append(
            EqualLinear(
                self.t_dim, self.c_dim*2, lr_mul=0.01, activation='fused_lrelu'
            )
        )
        for i in range(3):
            layers.append(
                EqualLinear(
                    self.c_dim*2, self.c_dim*2, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.fc = nn.Sequential(*layers)


    def encode(self, text_embedding):
        x = self.fc(text_embedding)
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar
class Mapper(Module):

    def __init__(self, opts, latent_dim=512):
        super(Mapper, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x

class T_map(Module):

    def __init__(self, opts, latent_dim=512):
        super(T_map, self).__init__()
        self.opts = opts
        self.latent_dim=latent_dim
        #layers = [PixelNorm()]
        layers=[]
        layers.append(
            EqualLinear(
                512, latent_dim*2, lr_mul=0.01, activation='fused_lrelu'
            )
        )
        layers.append(
            EqualLinear(
                latent_dim*2, latent_dim*4, lr_mul=0.01, activation='fused_lrelu'
            )
        )
        layers.append(
            EqualLinear(
                latent_dim*4, latent_dim*8, lr_mul=0.01, activation='fused_lrelu'
            )
        )
        layers.append(
            EqualLinear(
                latent_dim*8, latent_dim*16, lr_mul=0.01, activation='fused_lrelu'
            )
        )
        layers.append(
            EqualLinear(
                latent_dim*16, latent_dim*18, lr_mul=0.01, activation='fused_lrelu'
            )
        )
        # for i in range(5):
        #     layers.append(
        #         EqualLinear(
        #             latent_dim*18, latent_dim*18, lr_mul=0.01, activation='fused_lrelu'
        #         )
        #     )
        # for i in range(4):
        #     layers.append(
        #         EqualLinear(
        #             latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
        #         )
        #     )
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        # self.linear=EqualLinear(
        #     latent_dim*2, latent_dim, lr_mul=0.01, activation='fused_lrelu'
        # )
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        # out=self.linear(x)
        out=self.transformer_encoder(x.unsqueeze(dim=1))
        feat = out.squeeze(dim=1)
        out = self.mapping(out.squeeze(dim=1))
        out = torch.stack(out.split(512, 1),dim=1)
        # out=out.unsqueeze(dim=1).repeat(1,18,1)
        return out, feat

class I_map(Module):

    def __init__(self, opts, latent_dim=512):
        super(I_map, self).__init__()
        self.opts = opts
        self.latent_dim = latent_dim
        # layers = [PixelNorm()]
        layers = []
        layers.append(
            EqualLinear(
                768*26, latent_dim * 18, lr_mul=0.01, activation='fused_lrelu'
            )
        )
        # layers.append(
        #     EqualLinear(
        #         latent_dim * 2, latent_dim * 4, lr_mul=0.01, activation='fused_lrelu'
        #     )
        # )
        # layers.append(
        #     EqualLinear(
        #         latent_dim * 4, latent_dim * 8, lr_mul=0.01, activation='fused_lrelu'
        #     )
        # )
        # layers.append(
        #     EqualLinear(
        #         latent_dim * 8, latent_dim * 16, lr_mul=0.01, activation='fused_lrelu'
        #     )
        # )
        # layers.append(
        #     EqualLinear(
        #         latent_dim * 16, latent_dim * 18, lr_mul=0.01, activation='fused_lrelu'
        #     )
        # )
        for i in range(14):
            layers.append(
                EqualLinear(
                    latent_dim*18, latent_dim*18, lr_mul=0.01, activation='fused_lrelu'
                )
            )
        # for i in range(4):
        #     layers.append(
        #         EqualLinear(
        #             latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
        #         )
        #     )
        # encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # self.linear = EqualLinear(
        #     latent_dim * 2, latent_dim, lr_mul=0.01, activation='fused_lrelu'
        # )
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        # out=self.linear(x)
        # out = self.transformer_encoder(x.unsqueeze(dim=1).float())
        # feat=out.squeeze(dim=1)
        # out = self.mapping(out.squeeze(dim=1))
        x=x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        out=self.mapping(x.float())
        out = torch.stack(out.split(512, 1), dim=1)
        # out=out.unsqueeze(dim=1).repeat(1,18,1)
        feat=out
        return out,feat


class SingleMapper(Module):

    def __init__(self, opts):
        super(SingleMapper, self).__init__()

        self.opts = opts

        self.mapping = Mapper(opts)

    def forward(self, x):
        out = self.mapping(x)
        return out


class LevelsMapper(Module):

    def __init__(self, opts):
        super(LevelsMapper, self).__init__()

        self.opts = opts

        if not opts.no_coarse_mapper:
            self.course_mapping = Mapper(opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = Mapper(opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = Mapper(opts)

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

class FullStyleSpaceMapper(Module):

    def __init__(self, opts):
        super(FullStyleSpaceMapper, self).__init__()

        self.opts = opts

        for c, c_dim in enumerate(STYLESPACE_DIMENSIONS):
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=c_dim))

    def forward(self, x):
        out = []
        for c, x_c in enumerate(x):
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            out.append(x_c_res)

        return out


class WithoutToRGBStyleSpaceMapper(Module):

    def __init__(self, opts):
        super(WithoutToRGBStyleSpaceMapper, self).__init__()

        self.opts = opts

        indices_without_torgb = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
        self.STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in indices_without_torgb]

        for c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
            setattr(self, f"mapper_{c}", Mapper(opts, latent_dim=STYLESPACE_DIMENSIONS[c]))

    def forward(self, x):
        out = []
        for c in range(len(STYLESPACE_DIMENSIONS)):
            x_c = x[c]
            if c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            else:
                x_c_res = torch.zeros_like(x_c)
            out.append(x_c_res)

        return out