import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx, from_networkx
from itertools import combinations
import numpy as np
import copy
from .training import training_pipeline_cora, test_cora

def adjust_graph_topology(model, data, device):
    data = data[0].to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        probs = F.softmax(out, dim=1)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

    G = to_networkx(data, to_undirected=True)
    train_nodes = data.train_mask.nonzero(as_tuple=True)[0].tolist()  # Get list of training node indices
    
    # Track changes
    edge_changes = False
    edges_removed = 0
    edges_added = 0

    # Iterate only over combinations of training nodes
    for u, v in combinations(train_nodes, 2):
        if preds[u] != preds[v] and G.has_edge(u, v) and confidences[u] > 0.15 and confidences[v] > 0.15:
            G.remove_edge(u, v)
            edges_removed += 1
            edge_changes = True
        elif preds[u] == preds[v] and not G.has_edge(u, v) and confidences[u] > 0.15 and confidences[v] > 0.15:
            G.add_edge(u, v)
            edges_added += 1
            edge_changes = True

    # Calculate rates
    total_edges = G.number_of_edges()
    removal_rate = edges_removed / total_edges if total_edges > 0 else 0
    addition_rate = edges_added / total_edges if total_edges > 0 else 0


    # Only update edge_index if there were changes
    if edge_changes:
        print("Graph topology adjusted")
        new_edge_index = from_networkx(G).edge_index
        data.edge_index = new_edge_index

    return data.edge_index, removal_rate, addition_rate



def AdaEdge(gnn_constructor, loss_fcn, optimizer_constructor, dataset, device, lambda_reg=0, max_epochs=100, adaedge_iters=5):
    original_data = dataset.to(device)
    data = copy.deepcopy(original_data)

    removal_rate_list = []
    addition_rate_list = []

    for ada_iter in range(adaedge_iters):
        data = copy.deepcopy(data) 
        model = gnn_constructor().to(device)  # Instantiate a new model for each AdaEdge iteration
        optimizer = optimizer_constructor(model.parameters())  # Create new optimizer

        print(f"AdaEdge iteration {ada_iter + 1}/{adaedge_iters}")

        training_pipeline_cora(model, loss_fcn, optimizer, data, device, lambda_reg=lambda_reg, max_epochs=max_epochs, verbose=False)

        if ada_iter < adaedge_iters - 1:  # Adjust graph topology in all but last iteration
            data.edge_index, removal_rate, addition_rate = adjust_graph_topology(model, data, device)
            removal_rate_list.append(removal_rate)
            addition_rate_list.append(addition_rate)

    removal_rate = np.mean(removal_rate_list)
    addition_rate = np.mean(addition_rate_list)

    # Optionally reinitialize for final training if needed
    print("Final training on adjusted graph...")
    final_model = gnn_constructor().to(device)
    final_optimizer = optimizer_constructor(final_model.parameters())
    training_pipeline_cora(final_model, loss_fcn, final_optimizer, original_data, device, max_epochs=max_epochs)
    test_cora(final_model, original_data, device)

    return final_model, removal_rate, addition_rate
