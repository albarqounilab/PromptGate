import torch
import torch.nn.functional as F
import numpy as np

import torch.distributions as dist
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


# Helper for FEAL 
def fl_duc(global_model, local_model, dataloader):
    global_model.eval()
    local_model.eval()

    g_u_data_list = []
    l_u_data_list = []
    u_dis_list = []
    l_feature_list = []

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()

            # Global
            g_logit = global_model(image)[0]
            alpha = F.relu(g_logit) + 1
            # alpha = F.softplus(g_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)
            g_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            dirichlet = dist.Dirichlet(alpha)
            g_u_dis = dirichlet.entropy()

            # Local
            l_logit, _, _, block_features = local_model(image)
            l_feature = F.adaptive_avg_pool2d(block_features[-1], 3).flatten(start_dim=1)
            l_feature_list.append(l_feature)
            
            alpha_l = F.relu(l_logit) + 1
            # alpha_l = F.softplus(l_logit) + 1
            total_alpha_l = torch.sum(alpha_l, dim=1, keepdim=True)
            l_u_data = torch.sum((alpha_l / total_alpha_l) * (torch.digamma(total_alpha_l + 1) - torch.digamma(alpha_l + 1)), dim=1)

            g_u_data_list.append(g_u_data)
            l_u_data_list.append(l_u_data)
            u_dis_list.append(g_u_dis)

    return (torch.cat(g_u_data_list), torch.cat(l_u_data_list), 
            torch.cat(u_dis_list), torch.cat(l_feature_list))

def relaxation(u_rank_arg, l_feature_list, args, query_num):
    """
    Returns indices of selected samples (chosen_idx) based on diversity.
    """
    query_flag = torch.zeros(len(u_rank_arg)).cuda()
    chosen_idx = []
    
    for i in u_rank_arg: 
        if len(chosen_idx) == query_num:
            break

        cos_sim = pairwise_cosine_similarity(l_feature_list[i:i+1,:], l_feature_list)[0]
        neighbor_arg = torch.argsort(-cos_sim)
        valid_neighbors = neighbor_arg[cos_sim[neighbor_arg] > args.cosine][1 : 1 + args.n_neighbor]

        if query_flag[valid_neighbors].sum() == 0:
            query_flag[i] = 1
            chosen_idx.append(i.item())

    # Fill remainder if strict relaxation left us short
    if len(chosen_idx) < query_num:
        remain_idx = list(set(u_rank_arg.tolist()) - set(chosen_idx))
        chosen_idx.extend(remain_idx[:query_num - len(chosen_idx)])
        
    return chosen_idx