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

from scipy.sparse import csr_matrix

from integrao.IntegrAO_unsupervised import IntegrAO
from integrao.dataset import GraphDataset
import torch_geometric.transforms as T


def tsne_loss(P, activations, threshold=10000, sample_size=3000):
    """
    Computes KL divergence loss between P and Q.
    Uses sampling if sample_size < n. Falls back to CPU if effective size >= threshold.

    Parameters:
    - P: torch.Tensor, shape (n, n) — similarity matrix
    - activations: torch.Tensor, shape (n, d) — low-dimensional embeddings
    - threshold: int — if effective size >= threshold, move to CPU
    - sample_size: int — if set and < n, compute KL on a subset

    Returns:
    - KL divergence (scalar tensor) on GPU
    """
    eps = 1e-12
    n = P.shape[0]
    device = activations.device

    # Decide whether to sample
    effective_n = sample_size if sample_size and sample_size < n else n
    use_cpu = effective_n >= threshold

    if use_cpu:
        print(f"Using CPU for tsne_loss (effective N={effective_n})")
        P = P.cpu()
        activations = activations.detach().cpu()
    else:
        print(f"Using GPU for tsne_loss (effective N={effective_n})")
        P = P.to(device)
        activations = activations  # already on GPU

    # If using sampling, select subset of rows/cols
    if sample_size and sample_size < n:
        idx = torch.randperm(n)[:sample_size]
        P = P[idx][:, idx]
        activations = activations[idx]
        n = sample_size

    # Compute pairwise squared distances
    sum_act = torch.sum(activations**2, dim=1)
    Q = (
        sum_act.view([-1, 1]) + sum_act.view([1, -1])
        - 2.0 * activations @ activations.T
    )
    Q = (1 + Q).pow(-1.0)
    Q.fill_diagonal_(0.0)
    Q = Q / Q.sum()
    Q = torch.clamp(Q, min=eps)

    P = P / P.sum()
    C = torch.sum(P * torch.log((P + eps) / (Q + eps)))

    return C.to(device)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 100 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 100))
    lr = max(lr, 1e-3)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_model(net, device, restore):
    if restore is not None and os.path.exits(restore):
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


def tsne_p_deep(dicts_commonIndex, dict_sampleToIndexs, data, P=np.array([]), neighbor_size=20, embedding_dims=50, alighment_epochs=1000):
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

    hidden_channels = 128 # TODO: change to using ymal file
    dataset_num = len(P)
    feature_dims = []
    transform = T.Compose([
        T.ToDevice(device), 
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
            sample_size = 3000 if n > 10000 else None  # use sampling only if too large
            kl_loss += tsne_loss(P[i], X_embedding, threshold=10000, sample_size=sample_size)

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

        if (epoch) % 100 == 0:
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
