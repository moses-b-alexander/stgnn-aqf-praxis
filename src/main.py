


from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas import concat, DataFrame, read_csv, set_option
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from time import perf_counter, sleep
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
import tsl
from tsl.data import SpatioTemporalDataset
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE
from tsl.nn.blocks.encoders import *
from tsl.nn.blocks.decoders import *
from tsl.nn.layers import DiffConv, GATConv, GraphConv
from tsl.nn.models import BaseModel


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


torch.set_float32_matmul_precision("high")


beijing_subset = True

window_length, horizon_length = 20, 4

num_epochs = 100
batch_size = 64
learning_rate = 0.001
hidden_units = 512
encoding_dim = 128
dropout = 0.2

rnn_layers = 2
gnn_layers = 2

gnn_kernel = 4

num_attention_heads = 4

connectivity_threshold = 0.4
max_path_distance = 18


num_workers = 8

DataLoader.persistent_workers = True


model_types = [
    "CR-1", "CR-2", "CR-3", "CR-4", "CR-5",
    "CL-1", "CL-2", "CL-3", "CL-4", "CL-5",
    "CG-1", "CG-2", "CG-3", "CG-4", "CG-5",
    "DR-1", "DR-2", "DR-3", "DR-4", "DR-5",
    "DL-1", "DL-2", "DL-3", "DL-4", "DL-5",
    "DG-1", "DG-2", "DG-3", "DG-4", "DG-5",
    "AR-1", "AR-2", "AR-3", "AR-4", "AR-5",
    "AL-1", "AL-2", "AL-3", "AL-4", "AL-5",
    "AG-1", "AG-2", "AG-3", "AG-4", "AG-5"
]

name_mappings = {
    "GraphConv": "graph convolutional network",
    "DiffConv": "diffusion convolutional network",
    "GATConv": "attention convolutional network",
    "RNN": "recurrent neural network",
    "GRU": "gated recurrent unit network",
    "LSTM": "long short-term memory network",
}
id_mappings = {
    "GraphConv": "C",
    "DiffConv": "D",
    "GATConv": "A",
    "RNN": "R",
    "GRU": "G",
    "LSTM": "L",
}

num_cols = [
    "test_mae", "test_mae_lag_01", "test_mae_lag_02", "test_mae_lag_03", "test_mape",
    "number_of_parameters", "training_time_seconds",
]
cat_cols = [
    "tmp0",
    "spatial_component_id", "temporal_component_id", "spatial_component_name", "temporal_component_name",
    "tmp1",
    "additive", "concatenative", "positional_encoding", "attention",
    "model_name", "spatiotemporal_id", "model_id",
]

cols = [
    "tmp0",
    "spatial_component_id", "temporal_component_id", "spatial_component_name", "temporal_component_name",
    "tmp1",
    "test_mae", "test_mae_lag_01", "test_mae_lag_02", "test_mae_lag_03", "test_mape",
    "additive", "concatenative", "positional_encoding", "attention",
    "model_name", "spatiotemporal_id", "model_id",
    "number_of_parameters", "training_time_seconds",
]


dataset_dir = os.path.join(os.getcwd(), "dataset/")
results_dir = os.path.join(os.getcwd(), "results/")
plots_dir = os.path.join(os.getcwd(), "plots/")
if not os.path.exists(dataset_dir):  os.makedirs(dataset_dir)
if not os.path.exists(results_dir):  os.makedirs(results_dir)
if not os.path.exists(plots_dir):  os.makedirs(plots_dir)


def get_model(
    flag,
    spatial_type, temporal_type,
    additive, concatenative,
    *args,
):

    if additive == concatenative:
        return (None, None, None, None)

    class STGNNAdd(BaseModel):
        def __init__(
            self,
            input_size,
            covariate_size,
            gnn_type,
            rnn_type,
            n_nodes,
            horizon,
            hidden_size,
            encoding_size,
            num_gnn_layers,
            num_rnn_layers,
            num_layers,
            kernel_size,
            dropout_rate,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.encoding_size = encoding_size

            self.dropout = nn.Dropout(dropout_rate)

            self.spatial_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                if i == 0:
                    in_channels = input_size + covariate_size
                else:
                    in_channels = hidden_size
                if gnn_type == "GCN":
                    self.spatial_layers.append(
                        GraphConv(
                            input_size=in_channels,
                            output_size=hidden_size,
                            activation="leaky_relu",
                        )
                    )
                if gnn_type == "GDN":
                    self.spatial_layers.append(
                        DiffConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            activation="leaky_relu",
                            k=kernel_size,
                        )
                    )
                if gnn_type == "GAT":
                    self.spatial_layers.append(
                        GATConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            heads=kernel_size,
                        )
                    )

            self.rnn = None
            if rnn_type == "RNN":
                self.rnn = nn.RNN(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "LST":
                self.rnn = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "GRU":
                self.rnn = nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )

            self.fc = nn.Linear(hidden_size * 2, horizon)

        def forward(
            self, x,
            maximum_monthly_temperature, minimum_monthly_temperature,
            mean_monthly_temperature_extremity,
            mean_monthly_temperature, mean_annual_temperature,
            edge_index, edge_weight=None, **kwargs,
        ):
            batch_size, time_steps, n_nodes, input_size = x.shape

            covariates = torch.stack([
                maximum_monthly_temperature,
                minimum_monthly_temperature,
                mean_monthly_temperature_extremity,
                mean_monthly_temperature,
                mean_annual_temperature,
            ], dim=-1).unsqueeze(2).repeat(1, 1, n_nodes, 1)

            x_encoded = torch.cat([x, covariates], dim=-1)

            x_flat = x_encoded.view(batch_size * time_steps, n_nodes, -1)

            for i, spatial_layer in enumerate(self.spatial_layers):
                x_flat = spatial_layer(x_flat, edge_index)
                if isinstance(x_flat, tuple):
                    x_flat = x_flat[0]
                x_flat = nn.functional.elu(x_flat)

            gnn_out = x_flat.view(batch_size, time_steps, n_nodes, -1)
            gnn_out = self.dropout(gnn_out)

            rnn_input = gnn_out.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, time_steps, -1)

            rnn_out, _ = self.rnn(rnn_input)

            rnn_out = rnn_out.view(batch_size, n_nodes, time_steps, -1)
            rnn_out = self.dropout(rnn_out)

            rnn_out = rnn_out[:, :, -1, :]

            predictions = self.fc(rnn_out).permute(0, 2, 1).unsqueeze(-1)

            return predictions


    class STGNNConcat(BaseModel):
        def __init__(
            self,
            input_size,
            covariate_size,
            gnn_type,
            rnn_type,
            n_nodes,
            horizon,
            hidden_size,
            encoding_size,
            num_gnn_layers,
            num_rnn_layers,
            num_layers,
            kernel_size,
            dropout_rate,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.encoding_size = encoding_size

            self.dropout = nn.Dropout(dropout_rate)

            self.spatial_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                if i == 0:
                    in_channels = input_size + covariate_size
                else:
                    in_channels = hidden_size
                if gnn_type == "GCN":
                    self.spatial_layers.append(
                        GraphConv(
                            input_size=in_channels,
                            output_size=hidden_size,
                            activation="leaky_relu",
                        )
                    )
                if gnn_type == "GDN":
                    self.spatial_layers.append(
                        DiffConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            activation="leaky_relu",
                            k=kernel_size,
                        )
                    )
                if gnn_type == "GAT":
                    self.spatial_layers.append(
                        GATConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            heads=kernel_size,
                        )
                    )

            self.rnn = None
            if rnn_type == "RNN":
                self.rnn = nn.RNN(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "LST":
                self.rnn = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "GRU":
                self.rnn = nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )

            self.fc = nn.Linear(hidden_size * 2, horizon)

        def forward(
            self, x,
            maximum_monthly_temperature, minimum_monthly_temperature,
            mean_monthly_temperature_extremity,
            mean_monthly_temperature, mean_annual_temperature,
            edge_index, edge_weight=None, **kwargs,
        ):
            batch_size, time_steps, n_nodes, input_size = x.shape

            covariates = torch.stack([
                maximum_monthly_temperature,
                minimum_monthly_temperature,
                mean_monthly_temperature_extremity,
                mean_monthly_temperature,
                mean_annual_temperature,
            ], dim=-1).unsqueeze(2).repeat(1, 1, n_nodes, 1)

            x_encoded = torch.cat([x, covariates], dim=-1)

            x_flat = x_encoded.view(batch_size * time_steps, n_nodes, -1)

            for i, spatial_layer in enumerate(self.spatial_layers):
                x_flat = spatial_layer(x_flat, edge_index)
                if isinstance(x_flat, tuple):
                    x_flat = x_flat[0]
                x_flat = nn.functional.elu(x_flat)

            gnn_out = x_flat.view(batch_size, time_steps, n_nodes, -1)
            gnn_out = self.dropout(gnn_out)

            rnn_input = gnn_out.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, time_steps, -1)

            rnn_out, _ = self.rnn(rnn_input)

            rnn_out = rnn_out.view(batch_size, n_nodes, time_steps, -1)
            rnn_out = self.dropout(rnn_out)

            rnn_out = rnn_out[:, :, -1, :]

            predictions = self.fc(rnn_out).permute(0, 2, 1).unsqueeze(-1)

            return predictions


    class STGNNPosencAdd(BaseModel):
        def __init__(
            self,
            input_size,
            covariate_size,
            gnn_type,
            rnn_type,
            n_nodes,
            horizon,
            hidden_size,
            encoding_size,
            num_gnn_layers,
            num_rnn_layers,
            num_layers,
            kernel_size,
            dropout_rate,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.encoding_size = encoding_size

            self.feature_encoder = nn.Linear(input_size, encoding_size)

            self.learnable_positional_encoding = nn.Embedding(
                num_embeddings=window_length,
                embedding_dim=encoding_size,
            )

            self.shortest_path_embedding = nn.Embedding(
                num_embeddings=max_path_distance + 1,
                embedding_dim=encoding_size,
            )

            self.centrality_embedding = nn.Embedding(
                num_embeddings=n_nodes + 1,
                embedding_dim=encoding_size,
            )

            self.dropout = nn.Dropout(dropout_rate)

            self.spatial_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                if i == 0:
                    in_channels = encoding_size + covariate_size
                else:
                    in_channels = hidden_size
                if gnn_type == "GCN":
                    self.spatial_layers.append(
                        GraphConv(
                            input_size=in_channels,
                            output_size=hidden_size,
                            activation="leaky_relu",
                        )
                    )
                if gnn_type == "GDN":
                    self.spatial_layers.append(
                        DiffConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            activation="leaky_relu",
                            k=kernel_size,
                        )
                    )
                if gnn_type == "GAT":
                    self.spatial_layers.append(
                        GATConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            heads=kernel_size,
                        )
                    )

            self.rnn = None
            if rnn_type == "RNN":
                self.rnn = nn.RNN(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "LST":
                self.rnn = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "GRU":
                self.rnn = nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )

            self.fc = nn.Linear(hidden_size * 2, horizon)

        def compute_shortest_path_encoding(
            self, edge_index, n_nodes, batch_size
        ):
            adj = to_dense_adj(edge_index, max_num_nodes=n_nodes)
            dist = torch.where(adj == 0, max_path_distance + 1, adj)
            dist = torch.clamp(dist, max=max_path_distance)
            shortest_path_encoding = self.shortest_path_embedding(dist.long())
            return shortest_path_encoding

        def compute_centrality_encoding(
            self, edge_index, n_nodes, batch_size
        ):
            centrality_scores = torch.bincount(edge_index[0], minlength=n_nodes)
            centrality_encoding = self.centrality_embedding(centrality_scores.long())
            centrality_encoding = centrality_encoding.unsqueeze(0).expand(batch_size, -1, -1)
            return centrality_encoding

        def forward(
            self, x,
            maximum_monthly_temperature, minimum_monthly_temperature,
            mean_monthly_temperature_extremity,
            mean_monthly_temperature, mean_annual_temperature,
            edge_index, edge_weight=None, **kwargs,
        ):
            batch_size, time_steps, n_nodes, input_size = x.shape

            x_encoded = self.feature_encoder(x)

            positions = torch.arange(0, time_steps, device=x.device).unsqueeze(0)
            pos_encoding = self.learnable_positional_encoding(positions)
            pos_encoding = pos_encoding.unsqueeze(2).expand(batch_size, time_steps, n_nodes, -1)

            shortest_path_encoding = self.compute_shortest_path_encoding(edge_index, n_nodes, batch_size)
            shortest_path_encoding = shortest_path_encoding.mean(dim=2).unsqueeze(1).expand(-1, time_steps, -1, -1)

            centrality_encoding = self.compute_centrality_encoding(edge_index, n_nodes, batch_size)
            centrality_encoding = centrality_encoding.unsqueeze(1).expand(-1, time_steps, -1, -1)

            x_encoded += pos_encoding
            x_encoded += shortest_path_encoding
            x_encoded += centrality_encoding

            covariates = torch.stack([
                maximum_monthly_temperature,
                minimum_monthly_temperature,
                mean_monthly_temperature_extremity,
                mean_monthly_temperature,
                mean_annual_temperature,
            ], dim=-1).unsqueeze(2).repeat(1, 1, n_nodes, 1)

            x_encoded = torch.cat([x_encoded, covariates], dim=-1)

            x_flat = x_encoded.view(batch_size * time_steps, n_nodes, -1)

            for i, spatial_layer in enumerate(self.spatial_layers):
                x_flat = spatial_layer(x_flat, edge_index)
                if isinstance(x_flat, tuple):
                    x_flat = x_flat[0]
                x_flat = nn.functional.elu(x_flat)

            gnn_out = x_flat.view(batch_size, time_steps, n_nodes, -1)
            gnn_out = self.dropout(gnn_out)

            rnn_input = gnn_out.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, time_steps, -1)

            rnn_out, _ = self.rnn(rnn_input)

            rnn_out = rnn_out.view(batch_size, n_nodes, time_steps, -1)
            rnn_out = self.dropout(rnn_out)

            rnn_out = rnn_out[:, :, -1, :]

            predictions = self.fc(rnn_out).permute(0, 2, 1).unsqueeze(-1)

            return predictions


    class STGNNPosencConcat(BaseModel):
        def __init__(
            self,
            input_size,
            covariate_size,
            gnn_type,
            rnn_type,
            n_nodes,
            horizon,
            hidden_size,
            encoding_size,
            num_gnn_layers,
            num_rnn_layers,
            num_layers,
            kernel_size,
            dropout_rate,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.encoding_size = encoding_size

            self.learnable_positional_encoding = nn.Embedding(
                num_embeddings=window_length,
                embedding_dim=encoding_size,
            )

            self.shortest_path_embedding = nn.Embedding(
                num_embeddings=max_path_distance + 1,
                embedding_dim=encoding_size,
            )

            self.centrality_embedding = nn.Embedding(
                num_embeddings=n_nodes + 1,
                embedding_dim=encoding_size,
            )

            self.dropout = nn.Dropout(dropout_rate)

            self.spatial_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                if i == 0:
                    in_channels = input_size + (encoding_size * 3) + covariate_size
                else:
                    in_channels = hidden_size
                if gnn_type == "GCN":
                    self.spatial_layers.append(
                        GraphConv(
                            input_size=in_channels,
                            output_size=hidden_size,
                            activation="leaky_relu",
                        )
                    )
                if gnn_type == "GDN":
                    self.spatial_layers.append(
                        DiffConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            activation="leaky_relu",
                            k=kernel_size
                        )
                    )
                if gnn_type == "GAT":
                    self.spatial_layers.append(
                        GATConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            heads=kernel_size
                        )
                    )

            self.rnn = None
            if rnn_type == "RNN":
                self.rnn = nn.RNN(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "LST":
                self.rnn = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "GRU":
                self.rnn = nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )

            self.fc = nn.Linear(hidden_size * 2, horizon)

        def compute_shortest_path_encoding(
            self, edge_index, n_nodes, batch_size
        ):
            adj = to_dense_adj(edge_index, max_num_nodes=n_nodes)
            dist = torch.where(adj == 0, max_path_distance + 1, adj)
            dist = torch.clamp(dist, max=max_path_distance)
            shortest_path_encoding = self.shortest_path_embedding(dist.long())
            return shortest_path_encoding

        def compute_centrality_encoding(
            self, edge_index, n_nodes, batch_size
        ):
            centrality_scores = torch.bincount(edge_index[0], minlength=n_nodes)
            centrality_encoding = self.centrality_embedding(centrality_scores.long())
            centrality_encoding = centrality_encoding.unsqueeze(0).expand(batch_size, -1, -1)
            return centrality_encoding

        def forward(
            self, x,
            maximum_monthly_temperature, minimum_monthly_temperature,
            mean_monthly_temperature_extremity,
            mean_monthly_temperature, mean_annual_temperature,
            edge_index, edge_weight=None, **kwargs,
        ):
            batch_size, time_steps, n_nodes, input_size = x.shape

            positions = torch.arange(0, time_steps, device=x.device).unsqueeze(0)
            pos_encoding = self.learnable_positional_encoding(positions)
            pos_encoding = pos_encoding.unsqueeze(2).expand(batch_size, time_steps, n_nodes, -1)

            shortest_path_encoding = self.compute_shortest_path_encoding(edge_index, n_nodes, batch_size)
            shortest_path_encoding = shortest_path_encoding.mean(dim=2).unsqueeze(1)
            shortest_path_encoding = shortest_path_encoding.expand(-1, time_steps, -1, -1).repeat(batch_size, 1, 1, 1)

            centrality_encoding = self.compute_centrality_encoding(edge_index, n_nodes, batch_size)
            centrality_encoding = centrality_encoding.unsqueeze(1).expand(-1, time_steps, -1, -1)

            x_encoded = torch.cat([x, pos_encoding, shortest_path_encoding, centrality_encoding], dim=-1)

            covariates = torch.stack([
                maximum_monthly_temperature,
                minimum_monthly_temperature,
                mean_monthly_temperature_extremity,
                mean_monthly_temperature,
                mean_annual_temperature,
            ], dim=-1).unsqueeze(2).repeat(1, 1, n_nodes, 1)

            x_encoded = torch.cat([x_encoded, covariates], dim=-1)

            x_flat = x_encoded.view(batch_size * time_steps, n_nodes, -1)

            for i, spatial_layer in enumerate(self.spatial_layers):
                x_flat = spatial_layer(x_flat, edge_index)
                if isinstance(x_flat, tuple):
                    x_flat = x_flat[0]
                x_flat = nn.functional.elu(x_flat)

            gnn_out = x_flat.view(batch_size, time_steps, n_nodes, -1)
            gnn_out = self.dropout(gnn_out)

            rnn_input = gnn_out.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, time_steps, -1)

            rnn_out, _ = self.rnn(rnn_input)

            rnn_out = rnn_out.view(batch_size, n_nodes, time_steps, -1)
            rnn_out = self.dropout(rnn_out)

            rnn_out = rnn_out[:, :, -1, :]

            predictions = self.fc(rnn_out).permute(0, 2, 1).unsqueeze(-1)

            return predictions


    class STGNNAttentionAdd(BaseModel):
        def __init__(
            self,
            input_size,
            covariate_size,
            gnn_type,
            rnn_type,
            n_nodes,
            horizon,
            hidden_size,
            encoding_size,
            num_gnn_layers,
            num_rnn_layers,
            num_layers,
            kernel_size,
            dropout_rate,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.encoding_size = encoding_size

            self.feature_encoder = nn.Linear(input_size, encoding_size)

            self.learnable_positional_encoding = nn.Embedding(
                num_embeddings=window_length,
                embedding_dim=encoding_size,
            )

            self.shortest_path_embedding = nn.Embedding(
                num_embeddings=max_path_distance + 1,
                embedding_dim=encoding_size,
            )

            self.centrality_embedding = nn.Embedding(
                num_embeddings=n_nodes + 1,
                embedding_dim=encoding_size,
            )

            self.dropout = nn.Dropout(dropout_rate)

            self.spatial_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                if i == 0:
                    in_channels = encoding_size + covariate_size
                else:
                    in_channels = hidden_size
                if gnn_type == "GCN":
                    self.spatial_layers.append(
                        GraphConv(
                            input_size=in_channels,
                            output_size=hidden_size,
                            activation="leaky_relu",
                        )
                    )
                if gnn_type == "GDN":
                    self.spatial_layers.append(
                        DiffConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            activation="leaky_relu",
                            k=kernel_size,
                        )
                    )
                if gnn_type == "GAT":
                    self.spatial_layers.append(
                        GATConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            heads=kernel_size,
                        )
                    )

            self.rnn = None
            if rnn_type == "RNN":
                self.rnn = nn.RNN(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "LST":
                self.rnn = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "GRU":
                self.rnn = nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )

            self.bidimensional_attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=num_attention_heads,
                dropout=dropout_rate,
                batch_first=True,
            )

            self.fc = nn.Linear(hidden_size * 2, horizon)

        def compute_shortest_path_encoding(
            self, edge_index, n_nodes, batch_size
        ):
            adj = to_dense_adj(edge_index, max_num_nodes=n_nodes)
            dist = torch.where(adj == 0, max_path_distance + 1, adj)
            dist = torch.clamp(dist, max=max_path_distance)
            shortest_path_encoding = self.shortest_path_embedding(dist.long())
            return shortest_path_encoding

        def compute_centrality_encoding(
            self, edge_index, n_nodes, batch_size
        ):
            centrality_scores = torch.bincount(edge_index[0], minlength=n_nodes)
            centrality_encoding = self.centrality_embedding(centrality_scores.long())
            centrality_encoding = centrality_encoding.unsqueeze(0).expand(batch_size, -1, -1)
            return centrality_encoding

        def forward(
            self, x,
            maximum_monthly_temperature, minimum_monthly_temperature,
            mean_monthly_temperature_extremity,
            mean_monthly_temperature, mean_annual_temperature,
            edge_index, edge_weight=None, **kwargs
        ):
            batch_size, time_steps, n_nodes, input_size = x.shape

            x_encoded = self.feature_encoder(x)

            positions = torch.arange(0, time_steps, device=x.device).unsqueeze(0)
            pos_encoding = self.learnable_positional_encoding(positions)
            pos_encoding = pos_encoding.unsqueeze(2).expand(batch_size, time_steps, n_nodes, -1)

            shortest_path_encoding = self.compute_shortest_path_encoding(edge_index, n_nodes, batch_size)
            shortest_path_encoding = shortest_path_encoding.mean(dim=2).unsqueeze(1).expand(-1, time_steps, -1, -1)

            centrality_encoding = self.compute_centrality_encoding(edge_index, n_nodes, batch_size)
            centrality_encoding = centrality_encoding.unsqueeze(1).expand(-1, time_steps, -1, -1)

            x_encoded += pos_encoding
            x_encoded += shortest_path_encoding
            x_encoded += centrality_encoding

            covariates = torch.stack([
                maximum_monthly_temperature,
                minimum_monthly_temperature,
                mean_monthly_temperature_extremity,
                mean_monthly_temperature,
                mean_annual_temperature,
            ], dim=-1).unsqueeze(2).repeat(1, 1, n_nodes, 1)

            x_encoded = torch.cat([x_encoded, covariates], dim=-1)

            x_flat = x_encoded.view(batch_size * time_steps, n_nodes, -1)

            for i, spatial_layer in enumerate(self.spatial_layers):
                x_flat = spatial_layer(x_flat, edge_index)
                if isinstance(x_flat, tuple):
                    x_flat = x_flat[0]
                x_flat = nn.functional.elu(x_flat)

            gnn_out = x_flat.view(batch_size, time_steps, n_nodes, -1)
            gnn_out = self.dropout(gnn_out)

            rnn_input = gnn_out.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, time_steps, -1)

            rnn_out, _ = self.rnn(rnn_input)

            rnn_out = rnn_out.view(batch_size, n_nodes, time_steps, -1)
            rnn_out = self.dropout(rnn_out)

            attn_input = rnn_out.permute(0, 2, 1, 3).reshape(batch_size, time_steps * n_nodes, -1)

            attn_output, _ = self.bidimensional_attention(attn_input, attn_input, attn_input)

            attn_output = attn_output.view(batch_size, time_steps, n_nodes, -1)

            attn_output = attn_output.permute(0, 2, 1, 3)

            attn_output = attn_output.mean(dim=2)

            predictions = self.fc(attn_output).unsqueeze(-1)
            predictions = predictions.permute(0, 2, 1, 3)

            return predictions


    class STGNNAttentionConcat(BaseModel):
        def __init__(
            self,
            input_size,
            covariate_size,
            gnn_type,
            rnn_type,
            n_nodes,
            horizon,
            hidden_size,
            encoding_size,
            num_gnn_layers,
            num_rnn_layers,
            num_layers,
            kernel_size,
            dropout_rate,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.encoding_size = encoding_size

            self.learnable_positional_encoding = nn.Embedding(
                num_embeddings=window_length,
                embedding_dim=encoding_size,
            )

            self.shortest_path_embedding = nn.Embedding(
                num_embeddings=max_path_distance + 1,
                embedding_dim=encoding_size,
            )

            self.centrality_embedding = nn.Embedding(
                num_embeddings=n_nodes + 1,
                embedding_dim=encoding_size,
            )

            self.dropout = nn.Dropout(dropout_rate)

            self.spatial_layers = nn.ModuleList()
            for i in range(num_gnn_layers):
                if i == 0:
                    in_channels = input_size + (encoding_size * 3) + covariate_size
                else:
                    in_channels = hidden_size
                if gnn_type == "GCN":
                    self.spatial_layers.append(
                        GraphConv(
                            input_size=in_channels,
                            output_size=hidden_size,
                            activation="leaky_relu",
                        )
                    )
                if gnn_type == "GDN":
                    self.spatial_layers.append(
                        DiffConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            activation="leaky_relu",
                            k=kernel_size,
                        )
                    )
                if gnn_type == "GAT":
                    self.spatial_layers.append(
                        GATConv(
                            in_channels=in_channels,
                            out_channels=hidden_size,
                            heads=kernel_size,
                        )
                    )

            self.rnn = None
            if rnn_type == "RNN":
                self.rnn = nn.RNN(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "LST":
                self.rnn = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )
            if rnn_type == "GRU":
                self.rnn = nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size * 2,
                    num_layers=num_rnn_layers,
                    batch_first=True,
                )

            self.bidimensional_attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=num_attention_heads,
                dropout=dropout_rate,
                batch_first=True,
            )

            self.fc = nn.Linear(hidden_size * 2, horizon)

        def compute_shortest_path_encoding(
            self, edge_index, n_nodes, batch_size
        ):
            adj = to_dense_adj(edge_index, max_num_nodes=n_nodes)
            dist = torch.where(adj == 0, max_path_distance + 1, adj)
            dist = torch.clamp(dist, max=max_path_distance)
            shortest_path_encoding = self.shortest_path_embedding(dist.long())
            return shortest_path_encoding

        def compute_centrality_encoding(
            self, edge_index, n_nodes, batch_size
        ):
            centrality_scores = torch.bincount(edge_index[0], minlength=n_nodes)
            centrality_encoding = self.centrality_embedding(centrality_scores.long())
            centrality_encoding = centrality_encoding.unsqueeze(0).expand(batch_size, -1, -1)
            return centrality_encoding

        def forward(
            self, x,
            maximum_monthly_temperature, minimum_monthly_temperature,
            mean_monthly_temperature_extremity,
            mean_monthly_temperature, mean_annual_temperature,
            edge_index, edge_weight=None, **kwargs,
        ):
            batch_size, time_steps, n_nodes, input_size = x.shape

            positions = torch.arange(0, time_steps, device=x.device).unsqueeze(0)
            pos_encoding = self.learnable_positional_encoding(positions)
            pos_encoding = pos_encoding.unsqueeze(2).expand(batch_size, time_steps, n_nodes, -1)

            shortest_path_encoding = self.compute_shortest_path_encoding(edge_index, n_nodes, batch_size)
            shortest_path_encoding = shortest_path_encoding.mean(dim=2).unsqueeze(1)
            shortest_path_encoding = shortest_path_encoding.expand(-1, time_steps, -1, -1).repeat(batch_size, 1, 1, 1)

            centrality_encoding = self.compute_centrality_encoding(edge_index, n_nodes, batch_size)
            centrality_encoding = centrality_encoding.unsqueeze(1).expand(-1, time_steps, -1, -1)

            x_encoded = torch.cat([x, pos_encoding, shortest_path_encoding, centrality_encoding], dim=-1)

            covariates = torch.stack([
                maximum_monthly_temperature,
                minimum_monthly_temperature,
                mean_monthly_temperature_extremity,
                mean_monthly_temperature,
                mean_annual_temperature,
            ], dim=-1).unsqueeze(2).repeat(1, 1, n_nodes, 1)

            x_encoded = torch.cat([x_encoded, covariates], dim=-1)

            x_flat = x_encoded.view(batch_size * time_steps, n_nodes, -1)

            for i, spatial_layer in enumerate(self.spatial_layers):
                x_flat = spatial_layer(x_flat, edge_index)
                if isinstance(x_flat, tuple):
                    x_flat = x_flat[0]
                x_flat = nn.functional.elu(x_flat)

            gnn_out = x_flat.view(batch_size, time_steps, n_nodes, -1)
            gnn_out = self.dropout(gnn_out)

            rnn_input = gnn_out.permute(0, 2, 1, 3).reshape(batch_size * n_nodes, time_steps, -1)

            rnn_out, _ = self.rnn(rnn_input)

            rnn_out = rnn_out.view(batch_size, n_nodes, time_steps, -1)
            rnn_out = self.dropout(rnn_out)

            attn_input = rnn_out.permute(0, 2, 1, 3).reshape(batch_size, time_steps * n_nodes, -1)

            attn_output, _ = self.bidimensional_attention(attn_input, attn_input, attn_input)

            attn_output = attn_output.view(batch_size, time_steps, n_nodes, -1)

            attn_output = attn_output.permute(0, 2, 1, 3)

            attn_output = attn_output.mean(dim=2)

            predictions = self.fc(attn_output).unsqueeze(-1)
            predictions = predictions.permute(0, 2, 1, 3)

            return predictions


    if flag == "stgnn" and additive:
        stgnn_class = STGNNAdd

    if flag == "stgnn" and concatenative:
        stgnn_class = STGNNConcat

    if flag == "stgnn_posenc" and additive:
        stgnn_class = STGNNPosencAdd

    if flag == "stgnn_posenc" and concatenative:
        stgnn_class = STGNNPosencConcat

    if flag == "stgnn_attention" and additive:
        stgnn_class = STGNNAttentionAdd

    if flag == "stgnn_attention" and concatenative:
        stgnn_class = STGNNAttentionConcat

    return (stgnn_class, spatial_type, temporal_type, None)


def train(sc):

    print("\n\n\n\n")
    t0 = perf_counter()

    dataset = AirQuality(root="./data", small=beijing_subset)
    print(dataset)

    connectivity = dataset.get_connectivity(
        threshold=connectivity_threshold,
        include_self=False,
        normalize_axis=1,
        layout="edge_index"
        )

    edge_index, edge_weight = connectivity

    torch_dataset = SpatioTemporalDataset(
        target=dataset.dataframe(),
        connectivity=connectivity,
        mask=dataset.mask,
        horizon=horizon_length,
        window=window_length,
        stride=1,
    )

    df = read_csv(os.path.join(os.getcwd(),
        "weather/monthly-climatology.csv"))
    annual = read_csv(os.path.join(os.getcwd(), "weather/annual-average.csv"))
    annual_avg_temp_2014 = annual.loc[annual.Category == 2014, "Annual Mean"].iloc[0]
    annual_avg_temp_2015 = annual.loc[annual.Category == 2015, "Annual Mean"].iloc[0]

    mean_monthly_temperatures = []
    minimum_monthly_temperatures = []
    maximum_monthly_temperatures = []
    mean_monthly_precipitations = []
    for row in df.iterrows():
        if (row[0] + 1) in [9, 11, 1, 3, 4, 6, 8]:
            mean_monthly_temperatures += [row[1]["Average Mean Surface Air Temperature"]] * 31 * 24
            minimum_monthly_temperatures += [row[1]["Average Minimum Surface Air Temperature"]] * 31 * 24
            maximum_monthly_temperatures += [row[1]["Average Maximum Surface Air Temperature"]] * 31 * 24
            mean_monthly_precipitations += [row[1]["Precipitation"] / 31] * 31 * 24
        if (row[0] + 1) in [12, 2, 5, 7]:
            mean_monthly_temperatures += [row[1]["Average Mean Surface Air Temperature"]] * 30 * 24
            minimum_monthly_temperatures += [row[1]["Average Minimum Surface Air Temperature"]] * 30 * 24
            maximum_monthly_temperatures += [row[1]["Average Maximum Surface Air Temperature"]] * 30 * 24
            mean_monthly_precipitations += [row[1]["Precipitation"] / 30] * 30 * 24
        if (row[0] + 1) in [10]:
            mean_monthly_temperatures += [row[1]["Average Mean Surface Air Temperature"]] * 28 * 24
            minimum_monthly_temperatures += [row[1]["Average Minimum Surface Air Temperature"]] * 28 * 24
            maximum_monthly_temperatures += [row[1]["Average Maximum Surface Air Temperature"]] * 28 * 24
            mean_monthly_precipitations += [row[1]["Precipitation"] / 28] * 28 * 24

    if beijing_subset:
        mean_monthly_temperatures = mean_monthly_temperatures[:-1]
        minimum_monthly_temperatures = minimum_monthly_temperatures[:-1]
        maximum_monthly_temperatures = maximum_monthly_temperatures[:-1]
        mean_monthly_precipitations = mean_monthly_precipitations[:-1]

    torch_dataset.add_covariate(
        name="mean_annual_temperature",
        value=np.array(
            ([annual_avg_temp_2014] * 5880) +
            ([annual_avg_temp_2015] * (2880-1)) +
        []),
        pattern="t",
    )
    torch_dataset.add_covariate(
        name="mean_monthly_temperature",
        value=np.array(mean_monthly_temperatures),
        pattern="t",
    )
    torch_dataset.add_covariate(
        name="mean_monthly_temperature_extremity",
        value=np.array([np.abs(t - annual_avg_temp_2014)
            for t in mean_monthly_temperatures[:5880]] +
            [np.abs(t - annual_avg_temp_2015)
            for t in mean_monthly_temperatures[5880:]]),
        pattern="t",
    )
    torch_dataset.add_covariate(
        name="minimum_monthly_temperature",
        value=np.array(minimum_monthly_temperatures),
        pattern="t",
    )
    torch_dataset.add_covariate(
        name="maximum_monthly_temperature",
        value=np.array(maximum_monthly_temperatures),
        pattern="t",
    )

    input_size = torch_dataset.n_channels
    covariate_size = 5
    n_nodes = torch_dataset.n_nodes
    horizon = torch_dataset.horizon
    torch_dataset.save(os.path.join(dataset_dir, "aqi36.pt"))
    print(torch_dataset)

    scalers = {"target": StandardScaler(axis=(0, 1))}
    splitter = TemporalSplitter(test_len=0.15, val_len=0.05)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=batch_size,
        workers=num_workers,
        pin_memory=True,
    )

    if not sc[0]:
        return np.float32(0.0)

    stgnn = sc[0](
                input_size,
                covariate_size,
                sc[1],
                sc[2],
                n_nodes,
                horizon,
                hidden_units,
                encoding_dim,
                gnn_layers,
                rnn_layers,
                1,
                gnn_kernel,
                dropout,
            )
    stgnn = stgnn.to("cuda")
    print(stgnn)

    tot = sum([p.numel() for p in stgnn.parameters() if p.requires_grad])
    out = f"Number of model ({stgnn.__class__.__name__}) parameters:{tot:10d}"
    print("\n", "\n", "=" * (len(out) + 3), "\n", out, "\n", "=" * (len(out) + 3), "\n", "\n")

    loss_fn = MaskedMAE()

    metrics = {
        "mae": MaskedMAE(),
        "mape": MaskedMAPE(),
        "mae_lag_01": MaskedMAE(at=1),
        "mae_lag_02": MaskedMAE(at=2),
        "mae_lag_03": MaskedMAE(at=3),
    }

    predictor = Predictor(
        model=stgnn,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": learning_rate, "weight_decay": 0.0001},
        loss_fn=loss_fn,
        metrics=metrics,
    )

    logger = TensorBoardLogger(save_dir="logs", name="praxis", version=0)

    checkpoint_callback = ModelCheckpoint(
        dirpath="logs",
        save_top_k=1,
        monitor=None,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        precision="32-true",
        accelerator="gpu",
        devices=1,
        limit_train_batches=1000,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )

    trainer.fit(predictor, datamodule=dm)

    predictor.load_model(checkpoint_callback.best_model_path)
    predictor.freeze()

    tt = trainer.test(predictor, datamodule=dm)

    t1 = perf_counter()

    print(f"\n\nLoss value is {np.float32(tt[0]['test_mae'])}\n\n")
    print(f"\n\nRan for {round(t1-t0, 3)} seconds.\n\n")

    return np.float32(tt[0]["test_mae"])


def make_csv(filename):

    with open(filename, "r") as f:
        lines = f.readlines()
        splits = []
        prsd = []
        once = []
        one = []
        for l in lines:
            if "AQI" in l or "Dataset" in l:
                one.append(l.strip("\n"))
            if "â" not in l and "===" not in l:
                if "Dataset" not in l and "AQI" not in l and "DataLoader" not in l:
                    if l != " \n" and l != "\n" and l != "\n ":
                        prsd.append(l.strip("\n").strip(" "))
            if "Ran for" in l:
                splits.append(prsd)
                once.append(one)
                prsd = []
                one = []

    ds = []
    for sl in splits:
        d = {}
        d["tmp0"] = "nnnnn"
        d["tmp1"] = "nnn"
        d["model_name"] = sl[0][:-1]
        d["model_id"] = "0"
        for s in sl:
            for k in ["test_mae ", "test_mae_lag_01", "test_mae_lag_02", "test_mae_lag_03", "test_mape "]:
                if k in s:
                    d[k] = np.round(np.float64(s[::-1][:s[::-1].find(" ")][::-1]), 3)
            if "parameters" in s:
                d["number_of_parameters"] = np.int64(s[::-1][:s[::-1].find(":")][::-1].strip(" "))
            if "seconds" in s:
                d["training_time_seconds"] = np.round(np.float64(s[len("Ran for "):len(s)-len(" seconds.")]), 1)
            for gk in ["GraphConv", "DiffConv", "GATConv"]:
                if gk in s:
                    d["spatial_component_name"] = name_mappings[gk]
                    d["spatial_component_id"] = id_mappings[gk]
            for rk in ["RNN", "LSTM", "GRU"]:
                if rk in s:
                    d["temporal_component_name"] = name_mappings[rk]
                    d["temporal_component_id"] = id_mappings[rk]
        if "Posenc" in sl[0]:
            d["positional_encoding"] = True
            d["attention"] = False
        elif "Attention" in sl[0]:
            d["positional_encoding"] = True
            d["attention"] = True
        else:
            d["positional_encoding"] = False
            d["attention"] = False
        if "Posenc" in sl[0] or "Attention" in sl[0]:
            if "Add" in sl[0]:
                d["additive"] = True
                d["concatenative"] = False
            if "Concat" in sl[0]:
                d["additive"] = False
                d["concatenative"] = True
        else:
            d["additive"] = False
            d["concatenative"] = False
        if sl[0][:-1] == "STGNNAdd":  d["model_id"] = "1"
        elif sl[0][:-1] == "STGNNPosencAdd":  d["model_id"] = "2"
        elif sl[0][:-1] == "STGNNPosencConcat":  d["model_id"] = "3"
        elif sl[0][:-1] == "STGNNAttentionAdd":  d["model_id"] = "4"
        elif sl[0][:-1] == "STGNNAttentionConcat":  d["model_id"] = "5"
        else:  d["model_id"] = "6"
        d["spatiotemporal_id"] = f"{d['spatial_component_id']}{d['temporal_component_id']}"
        d["model_type"] = f"{d['spatiotemporal_id']}-{d['model_id']}"
        ds.append(d)

    dataset_info = "\n".join((["\n"] + once[0] + ["\n"]))

    df = DataFrame(ds)
    df = df.rename(columns={"test_mae ": "test_mae", "test_mape ": "test_mape"})
    df = df[["model_type"]+cols]
    df = df.loc[df.model_name != "STGNNConcat",]
    df.loc[df.model_name == "STGNNAdd", "model_name"] = "STGNNDefault"
    df = df.reset_index(drop=True)
    df.index = list(range(1, (9 * 5) + 1))
    df["trial"] = int(filename[-5:-4])
    df = df.copy()

    return df


def plot_mae(df, title, xcol, ycol, xlbl, ylbl, agg):

    if not agg:
        tag = int(df.trial.iloc[0])
    else:
        tag = "final"

    plt.figure(figsize=(8, 5), facecolor="none")
    if xcol == "index":
        bars = plt.bar(df.index, df[ycol], color="gray", edgecolor="black")
    else:
        bars = plt.bar(df[xcol], df[ycol], color="gray", edgecolor="black")
    for bar in bars:
        if "for All Models" not in title:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{height:.2f}",
                    ha="center", va="bottom", fontsize=8)
    plt.yticks([])
    plt.xticks(rotation=45)
    plt.xticks(fontsize=8)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    # plt.title(title, pad=10)
    plt.grid(axis="y", linestyle="-", color="black", alpha=0.2)
    if ycol == "test_mae":
        if "for All Models" in title:
            plt.yticks(np.linspace(0, 40.00, num=11))
        else:
            plt.yticks(np.linspace(0, 36.00, num=13))
    elif ycol == "number_of_parameters":
        if "for All Models" in title:
            plt.yticks(np.linspace(0, 2.5e7, num=11))
        else:
            plt.yticks(np.linspace(0, 2.4e7, num=13))
    elif ycol == "training_time_seconds":
        if "for All Models" in title:
            plt.yticks(np.linspace(0, 12000.00, num=11))
        else:
            plt.yticks(np.linspace(0, 11400.00, num=13))
    plt.tight_layout(pad=1.0)

    plt.savefig(
        os.path.join(plots_dir, f"{title.replace(' ', '-')}_{tag}.png"), dpi=300, transparent=True,
    )
    plt.close()

    return None


def make_plots(df, metric, variance, all_plots, agg):

    if metric == "test_mae":
        title_tag = "Model Error"
    if metric == "number_of_parameters":
        title_tag = "Model Size"
    if metric == "training_time_seconds":
        title_tag = "Model Training Time"

    if not variance:
        top_5 = df.sort_values(by=metric).head(5)
        bottom_5 = df.sort_values(by=metric).tail(5)

    if not variance:
        mv_tag = "Mean"

        mv_type_grp = df.groupby("model_id")[metric].mean()
        mv_type = DataFrame(mv_type_grp)

        mv_c_type_grp = df.loc[df.spatial_component_id == "C",].groupby("model_id")[metric].mean()
        mv_c_type = DataFrame(mv_c_type_grp)
        mv_d_type_grp = df.loc[df.spatial_component_id == "D",].groupby("model_id")[metric].mean()
        mv_d_type = DataFrame(mv_d_type_grp)
        mv_a_type_grp = df.loc[df.spatial_component_id == "A",].groupby("model_id")[metric].mean()
        mv_a_type = DataFrame(mv_a_type_grp)

        mv_r_type_grp = df.loc[df.temporal_component_id == "R",].groupby("model_id")[metric].mean()
        mv_r_type = DataFrame(mv_r_type_grp)
        mv_l_type_grp = df.loc[df.temporal_component_id == "L",].groupby("model_id")[metric].mean()
        mv_l_type = DataFrame(mv_l_type_grp)
        mv_g_type_grp = df.loc[df.temporal_component_id == "G",].groupby("model_id")[metric].mean()
        mv_g_type = DataFrame(mv_g_type_grp)

        mv_default_type_grp = df.loc[
            (df.model_id == "1"),
        ].groupby(["spatiotemporal_id"])[metric].mean()
        mv_default_type = DataFrame(mv_default_type_grp)

        mv_posenc_add_type_grp = df.loc[
            (df.model_id == "2"),
        ].groupby(["spatiotemporal_id"])[metric].mean()
        mv_posenc_add_type = DataFrame(mv_posenc_add_type_grp)
        mv_posenc_concat_type_grp = df.loc[
            (df.model_id == "3"),
        ].groupby(["spatiotemporal_id"])[metric].mean()
        mv_posenc_concat_type = DataFrame(mv_posenc_concat_type_grp)

        mv_attention_add_type_grp = df.loc[
            (df.model_id == "4"),
        ].groupby(["spatiotemporal_id"])[metric].mean()
        mv_attention_add_type = DataFrame(mv_attention_add_type_grp)
        mv_attention_concat_type_grp = df.loc[
            (df.model_id == "5"),
        ].groupby(["spatiotemporal_id"])[metric].mean()
        mv_attention_concat_type = DataFrame(mv_attention_concat_type_grp)

    else:
        mv_tag = "Variance"

        mv_type_grp = df.groupby("model_id")[metric].var()
        mv_type = DataFrame(mv_type_grp)

        mv_c_type_grp = df.loc[df.spatial_component_id == "C",].groupby("model_id")[metric].var()
        mv_c_type = DataFrame(mv_c_type_grp)
        mv_d_type_grp = df.loc[df.spatial_component_id == "D",].groupby("model_id")[metric].var()
        mv_d_type = DataFrame(mv_d_type_grp)
        mv_a_type_grp = df.loc[df.spatial_component_id == "A",].groupby("model_id")[metric].var()
        mv_a_type = DataFrame(mv_a_type_grp)

        mv_r_type_grp = df.loc[df.temporal_component_id == "R",].groupby("model_id")[metric].var()
        mv_r_type = DataFrame(mv_r_type_grp)
        mv_l_type_grp = df.loc[df.temporal_component_id == "L",].groupby("model_id")[metric].var()
        mv_l_type = DataFrame(mv_l_type_grp)
        mv_g_type_grp = df.loc[df.temporal_component_id == "G",].groupby("model_id")[metric].var()
        mv_g_type = DataFrame(mv_g_type_grp)

        mv_default_type_grp = df.loc[(df.model_id == "1"),].groupby(["spatiotemporal_id"])[metric].var()
        mv_default_type = DataFrame(mv_default_type_grp)

        mv_posenc_add_type_grp = df.loc[(df.model_id == "2"),].groupby(["spatiotemporal_id"])[metric].var()
        mv_posenc_add_type = DataFrame(mv_posenc_add_type_grp)
        mv_posenc_concat_type_grp = df.loc[(df.model_id == "3"),].groupby(["spatiotemporal_id"])[metric].var()
        mv_posenc_concat_type = DataFrame(mv_posenc_concat_type_grp)

        mv_attention_add_type_grp = df.loc[(df.model_id == "4"),].groupby(["spatiotemporal_id"])[metric].var()
        mv_attention_add_type = DataFrame(mv_attention_add_type_grp)
        mv_attention_concat_type_grp = df.loc[(df.model_id == "5"),].groupby(["spatiotemporal_id"])[metric].var()
        mv_attention_concat_type = DataFrame(mv_attention_concat_type_grp)

    if all_plots:
        if not variance:
            plot_mae(
                top_5,
                f"Models with Lowest {title_tag}",
                "model_type", metric, "Model Type", "Test Mean Absolute Error", agg
            )
            plot_mae(
                bottom_5,
                f"Models with Highest {title_tag}",
                "model_type", metric, "Model Type", "Test Mean Absolute Error", agg
            )

        plot_mae(
            mv_type,
            f"{title_tag} {mv_tag} for All Model Classes",
            "index", metric, "Model Class", "Test Mean Absolute Error", agg
        )

        plot_mae(
            mv_c_type,
            f"{title_tag} {mv_tag} for Graph Convolutional Networks",
            "index", metric, "Model Class", "Test Mean Absolute Error", agg
        )
        plot_mae(
            mv_d_type,
            f"{title_tag} {mv_tag} for Diffusion Convolutional Networks",
            "index", metric, "Model Class", "Test Mean Absolute Error", agg
        )
        plot_mae(
            mv_a_type,
            f"{title_tag} {mv_tag} for Attention Convolutional Networks",
            "index", metric, "Model Class", "Test Mean Absolute Error", agg
        )

        plot_mae(
            mv_r_type,
            f"{title_tag} {mv_tag} for Recurrent Neural Networks",
            "index", metric, "Model Class", "Test Mean Absolute Error", agg
        )
        plot_mae(
            mv_l_type,
            f"{title_tag} {mv_tag} for Long Short-term Memory Networks",
            "index", metric, "Model Class", "Test Mean Absolute Error", agg
        )
        plot_mae(
            mv_g_type,
            f"{title_tag} {mv_tag} for Gated Recurrent Unit Networks",
            "index", metric, "Model Class", "Test Mean Absolute Error", agg
        )

        if not variance:
            plot_mae(
                mv_default_type,
                f"{title_tag} {mv_tag} for Models without Positional Encoding and without Attention",
                "index", metric, "Model Type", "Test Mean Absolute Error", agg
            )
            plot_mae(
                mv_posenc_add_type,
                f"{title_tag} {mv_tag} for Models with Additive Positional Encoding and without Attention",
                "index", metric, "Model Type", "Test Mean Absolute Error", agg
            )
            plot_mae(
                mv_posenc_concat_type,
                f"{title_tag} {mv_tag} for Models with Concatenative Positional Encoding and without Attention",
                "index", metric, "Model Type", "Test Mean Absolute Error", agg
            )
            plot_mae(
                mv_attention_add_type,
                f"{title_tag} {mv_tag} for Models with Additive Positional Encoding and with Attention",
                "index", metric, "Model Type", "Test Mean Absolute Error", agg
            )
            plot_mae(
                mv_attention_concat_type,
                f"{title_tag} {mv_tag} for Models with Concatenative Positional Encoding and with Attention",
                "index", metric, "Model Type", "Test Mean Absolute Error", agg
            )

    if not variance:
        all_models = df.sort_values(by=metric)
        plot_mae(
            all_models,
            f"{title_tag} for All Models",
            "model_type", metric, "Model Type", "Test Mean Absolute Error", agg
        )

    return None


def full_run(filenames):

    dfs = [make_csv(filename) for filename in filenames]
    l = []
    for mt in model_types:
        rs = [sdf.loc[sdf.model_type == mt,].iloc[0] for sdf in dfs]
        cdf = DataFrame(rs, columns=["model_type"]+cols)
        num_cdf = DataFrame([cdf[num_cols].mean()]).astype(np.float32)
        num_cdf[num_cols[:-2]] = num_cdf[num_cols[:-2]].round(3)
        num_cdf[num_cols[-2:]] = num_cdf[num_cols[-2:]].round(1)
        cat_cdf = cdf[cat_cols].mode()
        new_cdf = concat([num_cdf, cat_cdf], axis=1)
        new_cdf["model_type"] = mt
        l.append(new_cdf)
    df = concat(l, axis=0)
    df = df.reset_index(drop=True)
    df.index = list(range(1, (9 * 5) + 1))
    df = df[["model_type"]+cols]
    df = df.copy()
    df.to_csv(os.path.join(results_dir, f"results_final.csv"))
    make_plots(df, "test_mae", False, True, True)
    make_plots(df, "number_of_parameters", False, True, True)
    make_plots(df, "training_time_seconds", False, True, True)
    make_plots(df, "test_mae", True, True, True)

    return None


if __name__ == "__main__":

    spatial_typs = ["GCN", "GDN", "GAT"]
    temporal_typs = ["RNN", "LST", "GRU"]


    # # TODO UNCOMMENT this to run the experiment for each of all 9 spatiotemporal combinations and 5 architectures
    # # in total, training 45 models
    # for st0 in spatial_typs[:]:
    #     for tt0 in temporal_typs[:]:
    #         # 3 x 3

    #         sleep(30)
    #         train(get_model("stgnn", st0, tt0, True, False)) # 1
    #         sleep(30)

    #         sleep(30)
    #         train(get_model("stgnn_posenc", st0, tt0, True, False)) # 2
    #         sleep(30)
    #         train(get_model("stgnn_posenc", st0, tt0, False, True)) # 3
    #         sleep(30)

    #         sleep(30)
    #         train(get_model("stgnn_attention", st0, tt0, True, False)) # 4
    #         sleep(30)
    #         train(get_model("stgnn_attention", st0, tt0, False, True)) # 5
    #         sleep(30)


    # # TODO UNCOMMENT this to produce RESULTS.CSV and PLOTS.PNGs
    # filename_0 = os.path.join(os.getcwd(), "experiment.txt")
    # full_run([filename_0])


    pass


