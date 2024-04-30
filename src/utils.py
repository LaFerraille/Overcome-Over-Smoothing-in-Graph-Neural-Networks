import numpy as np
import torch

def adjacency_matrix(edge_index, num_nodes):
    # Create an adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj

def cosine_distance_matrix(H):
    # Normalize the representation matrix H to unit vectors
    norm_H = H / H.norm(dim=1, keepdim=True)
    # Compute the cosine similarity matrix
    cosine_sim = torch.mm(norm_H, norm_H.t())
    # Convert to cosine distance matrix
    # Cosine distance is not affected by the absolute value of the node vector
    D = 1 - cosine_sim
    return D

def MAD(D, mask=None):
    # Element-wise multiplication of D and M_tgt
    if mask is None:  # Global MAD calculation
        mask = torch.ones_like(D)
    D_tgt = D * mask
    
    # Compute the average distance for non-zero values along each row in D_tgt
    row_sums = D_tgt.sum(dim=1)
    row_counts = (D_tgt != 0).sum(dim=1).float()
    
    # Avoid division by zero
    row_counts[row_counts == 0] = 1
    
    D_avg_tgt = row_sums / row_counts
    
    # Calculate MAD value by averaging the non-zero values in D_avg_tgt
    MAD = D_avg_tgt.sum() / (D_avg_tgt != 0).float().sum()
    
    return MAD.item()

def dirichlet_energy(X, edge_index,adj_matrix, device):
    # X: [N, F], edge_index: [2, E]
    N = X.shape[0]
    
    # Compute the degree of each node
    degree_matrix = adj_matrix.clone().detach().to(device)
    # Compute the energy
    i, j = edge_index
    energy = torch.norm(X[i]/torch.sqrt(degree_matrix[i,i].unsqueeze(1) + 1) - X[j]/torch.sqrt(degree_matrix[j,j].unsqueeze(1) + 1), dim=1) ** 2
    energy = energy.sum()

    return energy.item()

def calculate_mat(adj_matrix, order_near, order_far):
    """
    As in the paper, we calculate MAD_neib based on nodes with orders ≤ 3 and MAD_far based on nodes with orders ≥ 8.
    MAGGap = MAD_far - MAD_neib
    """

    adj_matrix_simple = adj_matrix.clone().detach().float()
    adj_matrix_simple.fill_diagonal_(0)
    
    # Matrix to accumulate paths of different lengths
    mat_sum = adj_matrix_simple.clone()
    product = adj_matrix_simple.clone()

    # Accumulate paths up to 'order_near'
    for power in range(2, order_near + 1):
        product = torch.matmul(product, adj_matrix_simple).clamp_max_(1)
        mat_sum += product

    mat_neib = mat_sum.clamp_max_(1).int()

    # Continue accumulating paths to reach 'order_far'
    for power in range(order_near + 1, order_far + 1):
        product = torch.matmul(product, adj_matrix_simple).clamp_max_(1)
        mat_sum += product

    mat_far = (torch.ones_like(mat_sum) - mat_sum.clamp_max_(1)).int()
    
    return mat_neib, mat_far

def MADGap(H, mat_neib, mat_far):    
    mat_neib = mat_neib.float()  # Convert to float for division
    mat_far = mat_far.float()  # Convert to float for division

    cos_dist = cosine_distance_matrix(H)

    cos_dist_near = cos_dist * mat_neib
    cos_dist_far = cos_dist * mat_far

    # Calculate sums and avoid division by zero
    mat_neib_line_sum = torch.clamp(mat_neib.sum(dim=1), min=1)
    mat_far_line_sum = torch.clamp(mat_far.sum(dim=1), min=1)

    cos_dist_near_line = torch.sum(cos_dist_near, dim=1) / mat_neib_line_sum
    cos_dist_far_line = torch.sum(cos_dist_far, dim=1) / mat_far_line_sum

    mad_near = cos_dist_near_line.mean()
    mad_far = cos_dist_far_line.mean()

    mad_gap = mad_far - mad_near
    return mad_near.item(), mad_far.item(), mad_gap.item()

def compute_mad_and_madgap(model, dataset, device, order_near=3, order_far=8):
    model.eval()
    with torch.no_grad():
        data = dataset[0].to(device)
        embeddings = model(data.x, data.edge_index)
        adj_matrix = adjacency_matrix(data.edge_index, embeddings.shape[0]).to(device)

        # Compute cosine distance matrix, MAD, and MADGap
        cosine_matrix = cosine_distance_matrix(embeddings)
        mad = MAD(cosine_matrix)
        mat_neib, mat_far = calculate_mat(adj_matrix, order_near, order_far)
        mad_near, mad_far, mad_gap = MADGap(embeddings, mat_neib, mat_far)

    return mad, mad_gap

def compute_energy(model, dataset, device):
    model.eval()
    with torch.no_grad():
        data = dataset[0].to(device)
        embeddings = model(data.x, data.edge_index)
        adj_matrix = adjacency_matrix(data.edge_index, embeddings.shape[0]).to(device)
        energy = dirichlet_energy(embeddings, data.edge_index, adj_matrix, device)

    return energy
