import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from snf.compute import _find_dominate_set
import numpy as np
import networkx as nx
import time

from integrao.IntegrAO_unsupervised import IntegrAO
from integrao.dataset import GraphDataset
import torch_geometric.transforms as T


def tsne_loss(P, activations, *, sample_size=None, rng=None):
    """
    KL(P ‖ Q) with optional subset sampling.

    • If `sample_size` < N, the loss is estimated on a random subset of
      that many rows/columns to save memory.

    • After sub‑sampling we *renormalise* both P and Q so that
      their entries sum to 1, ensuring the KL term is ≥ 0.
    """

    eps = 1e-12

    device = activations.device
    n = activations.size(0)

    # ---------- optional sub‑sampling ---------------------------------
    if sample_size is not None and sample_size < n:
        if rng is None:
            rng = torch.Generator(device=device)
        idx = torch.randperm(n, generator=rng, device=device)[:sample_size]
        activations = activations[idx]          # (s, d)
        P = P[idx][:, idx]                      # (s, s)
        n = sample_size

    # ---------- compute Q ---------------------------------------------
    alpha = 1.0
    sum_act = torch.sum(activations.pow(2), dim=1)            # (n,)
    Q = (
                sum_act
                + sum_act.view([-1, 1])
                - 2 * activations @ activations.t()
        ) / alpha
    Q = torch.pow(1 + Q, -(alpha + 1) / 2)
    Q *= (1.0 - torch.eye(n, device=device))                  # zero diag
    Q = Q.clamp_min_(eps)

    # ---------- *renormalise* both matrices ---------------------------
    P = P.clamp_min(eps)
    P /= P.sum() + eps
    Q /= Q.sum() + eps

    # ---------- KL divergence -----------------------------------------
    C = torch.sum(P * torch.log((P + eps) / (Q + eps)))
    return C


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 100 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 100))
    lr = max(lr, 1e-3)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_model(net, device, restore):
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    else:
        pass

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.to(device)

    return net


def P_preprocess(P):
    # Make sure P-values are set properly
    np.fill_diagonal(P, 0)  # set diagonal to zero
    P = P + np.transpose(P)  # symmetrize P-values
    P = P / np.sum(P)  # make sure P-values sum to one
    P = P * 4.0  # early exaggeration
    P = np.maximum(P, 1e-12)
    return P


def tsne_p_deep(dicts_commonIndex, dict_sampleToIndexs, data, P=np.array([]), neighbor_size=20, embedding_dims=64,
                alighment_epochs=1000):
    """
    Runs t-SNE on the dataset in the NxN matrix P to extract embedding vectors
    to no_dims dimensions.
    """

    # Check inputs
    if isinstance(embedding_dims, float):
        print("Error: array P should have type float.")
        return -1
    if round(embedding_dims) != embedding_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    print("\nStarting unsupervised embedding extraction!")
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_channels = 128  # TODO: change to using ymal file
    dataset_num = len(P)
    feature_dims = []
    transform = T.Compose([
        T.ToDevice(str(device)),
    ])

    x_dict = {}
    edge_index_dict = {}
    for i in range(dataset_num):
        # preprocess the inputs into PyG graph format
        dataset = GraphDataset(neighbor_size, data[i], P[i], transform=transform)
        x_dict[i] = dataset[0].x
        edge_index_dict[i] = dataset[0].edge_index

        feature_dims.append(np.shape(data[i])[1])
        print("Dataset {}:".format(i), np.shape(data[i]))

        # preprocess similarity matrix for t-sne loss
        P[i] = P_preprocess(P[i])
        P[i] = torch.from_numpy(P[i]).float().to(device)

    net = IntegrAO(feature_dims, hidden_channels, embedding_dims)
    Project_GNN = init_model(net, device, restore=None)
    Project_GNN.train()

    optimizer = torch.optim.Adam(Project_GNN.parameters(), lr=1e-1)
    c_mse = nn.MSELoss()

    for epoch in range(alighment_epochs):
        adjust_learning_rate(optimizer, epoch)

        loss = 0
        embeddings = []

        kl_loss = np.array(0)
        kl_loss = torch.from_numpy(kl_loss).to(device).float()

        embeddings = Project_GNN(x_dict, edge_index_dict)
        embeddings = list(embeddings.values())
        for i, X_embedding in enumerate(embeddings):
            n = P[i].shape[0]
            sample_size = 10000 if n > 10000 else None  # use sampling only if too large
            kl_loss += tsne_loss(P[i], X_embedding)

        # pairwise alignment loss between each pair of networks
        alignment_loss = np.array(0)
        alignment_loss = torch.from_numpy(alignment_loss).to(device).float()

        for i in range(dataset_num - 1):
            for j in range(i + 1, dataset_num):
                low_dim_set1 = embeddings[i][dicts_commonIndex[(i, j)]]
                low_dim_set2 = embeddings[j][dicts_commonIndex[(j, i)]]
                alignment_loss += c_mse(low_dim_set1, low_dim_set2)

        loss += kl_loss + alignment_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                "epoch {}: loss {}, align_loss:{:4f}".format(
                    epoch, loss.data.item(), alignment_loss.data.item()
                )
            )
        if epoch == 100:
            for i in range(dataset_num):
                P[i] = P[i] / 4.0

    # get the final embeddings for all samples
    embeddings = Project_GNN(x_dict, edge_index_dict)
    for i in range(dataset_num):
        embeddings[i] = embeddings[i].detach().cpu().numpy()

        # compute the average embedding for each sample
    final_embedding = np.array([]).reshape(0, embedding_dims)
    for key in dict_sampleToIndexs:
        sample_embedding = np.zeros((1, embedding_dims))

        for (dataset, index) in dict_sampleToIndexs[key]:
            sample_embedding += embeddings[dataset][index]
        sample_embedding /= len(dict_sampleToIndexs[key])

        final_embedding = np.concatenate((final_embedding, sample_embedding), axis=0)

    end_time = time.time()
    print(f"Manifold alignment ends! Times: {end_time - start_time}s")

    return final_embedding, Project_GNN
