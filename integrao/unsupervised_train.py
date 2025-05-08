import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import os
import numpy as np
import time

from integrao.IntegrAO_unsupervised import IntegrAO
from integrao.dataset import GraphDataset
import torch_geometric.transforms as T

cudnn.benchmark = True


def tsne_loss(P, activations, threshold_mb=32000, sample_size=10000):
    """
    Computes KL divergence loss between similarity matrix P and low-dim activations.

    Parameters:
    - P: torch.Tensor (n, n) — high-dim similarities
    - activations: torch.Tensor (n, d) — low-dim embeddings
    - threshold_mb: float — max memory in MB before offloading to CPU (default: 8GB)
    - sample_size: int — number of points to sample for approximation

    Returns:
    - Scalar tensor (KL divergence), computed on GPU if possible
    """
    eps = 1e-12
    n = P.shape[0]
    device = activations.device

    # Use sampling if desired
    if sample_size and sample_size < n:
        idx = torch.randperm(n)[:sample_size]
        P = P[idx][:, idx]
        activations = activations[idx]
        n = sample_size

    # Estimate memory: n x n float32 matrix (4 bytes per element)
    estimated_bytes = (n ** 2) * 4
    estimated_mb = estimated_bytes / (1024 ** 2)

    use_cpu = estimated_mb > threshold_mb
    if use_cpu:
        P = P.cpu()
        activations = activations.detach().cpu()
        device = torch.device("cpu")

    # Compute pairwise distances
    sum_act = torch.sum(activations ** 2, dim=1)
    Q = (
            sum_act.view([-1, 1]) + sum_act.view([1, -1])
            - 2.0 * activations @ activations.T
    )
    Q = (1 + Q).pow(-1.0)
    Q.fill_diagonal_(0.0)

    # Normalize Q
    Q_sum = Q.sum().item()
    Q = Q / Q_sum
    Q = torch.clamp(Q, min=eps)

    # Normalize P
    P_sum = P.sum().item()
    P = P / P_sum

    # KL divergence
    log_div = torch.log((P + eps) / (Q + eps))
    if torch.isnan(log_div).any():
        print("[tsne_loss] Warning: NaNs detected in log(P/Q)")

    C = torch.sum(P * log_div)

    if torch.isnan(C):
        print("[tsne_loss] KL loss is NaN")

    return C.to(device)


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

    print("Starting unsupervised embedding extraction!")
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
            sample_size = 5000 if n > 10000 else None  # use sampling only if too large
            kl_loss += tsne_loss(P[i], X_embedding, sample_size=sample_size)

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
    print("Manifold alignment ends! Times: {}s".format(end_time - start_time))

    return final_embedding, Project_GNN
