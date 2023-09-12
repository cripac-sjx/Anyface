import torch
from torch import nn
def rel_loss(batch, src_feat, trg_feat):
    sfm = nn.Softmax(dim=1)
    sim = nn.CosineSimilarity()
    kl_loss = nn.KLDivLoss()
    dist_source=torch.zeros([batch,batch-1]).cuda()
    dist_target = torch.zeros([batch, batch - 1]).cuda()
    for pair1 in range(batch):
        tmpc = 0
        # comparing the possible pairs
        for pair2 in range(batch):
            if pair1 != pair2:
                anchor_feat = torch.unsqueeze(
                    src_feat[pair1].reshape(-1), 0)
                compare_feat = torch.unsqueeze(
                    src_feat[pair2].reshape(-1), 0)
                dist_source[pair1, tmpc] = sim(
                    anchor_feat, compare_feat)
                tmpc += 1
    dist_source = sfm(dist_source)

    for pair1 in range(batch):
        tmpc = 0
        # comparing the possible pairs
        for pair2 in range(batch):
            if pair1 != pair2:
                anchor_feat = torch.unsqueeze(
                    trg_feat[pair1].reshape(-1), 0)
                compare_feat = torch.unsqueeze(
                    trg_feat[pair2].reshape(-1), 0)
                dist_target[pair1, tmpc] = sim(
                    anchor_feat, compare_feat)
                tmpc += 1
    dist_target = sfm(dist_target)

    loss=1000*kl_loss(torch.log(dist_source), dist_target)
    
    return loss