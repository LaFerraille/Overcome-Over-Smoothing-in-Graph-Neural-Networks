import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv

class GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_size, hidden_size))
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_size, hidden_size))
        self.convs.append(GCNConv(hidden_size, output_size))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(input_size, hidden_size))
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_size, hidden_size))
        self.convs.append(GATConv(hidden_size, output_size))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_size, hidden_size))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.convs.append(SAGEConv(hidden_size, output_size))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class ChebNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, K=5):
        super(ChebNet, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(ChebConv(input_size, hidden_size, K))
        for i in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_size, hidden_size, K))
        self.convs.append(ChebConv(hidden_size, output_size, K))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    

class EnhancedGAT(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, heads=4, mlp_hidden=64):
        super(EnhancedGAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()

        # First GAT layer
        self.convs.append(GATConv(input_size, hidden_size, heads=heads, dropout=0.6))

        # Additional GAT layers
        for _ in range(1, num_layers - 1):
            self.convs.append(GATConv(hidden_size * heads, hidden_size, heads=heads, dropout=0.6))
            self.mlps.append(Sequential(
                Linear(hidden_size * heads, mlp_hidden),
                ReLU(),
                BatchNorm1d(mlp_hidden),
                Linear(mlp_hidden, hidden_size * heads),
                ReLU(),
                BatchNorm1d(hidden_size * heads)
            ))

        # Final GAT layer to produce output
        self.convs.append(GATConv(hidden_size * heads, output_size, concat=False, heads=heads, dropout=0.6))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            if i < len(self.mlps):
                x = self.mlps[i](x)  # Apply MLP after each GAT layer except the last one to fight against over-smoothing
        x = self.convs[-1](x, edge_index)  # Apply final GAT layer
        return x