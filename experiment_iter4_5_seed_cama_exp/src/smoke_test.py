#!/usr/bin/env python3
"""Smoke test: 1 task (rel-trial/study-outcome), 1 seed, 2 epochs, both methods."""

import os, sys, math, copy, json, time, gc, resource
from pathlib import Path
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

# Hardware
def _container_ram_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError): pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
RAM_BUDGET = int(TOTAL_RAM_GB * 0.80 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(min(int(_total * 0.90) / _total, 0.95))

logger.info(f"Device: {DEVICE}, RAM: {TOTAL_RAM_GB:.1f}GB")

from torch_geometric.seed import seed_everything
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv, MLP
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.norm import LayerNorm
from torch_frame.config.text_embedder import TextEmbedderConfig
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.expanduser("~/.cache/relbench_cama_exp")

# CAMAAggregation
class CAMAAggregation(Aggregation):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gate_net = nn.Linear(1, channels, bias=True)
        self.var_transform = nn.Linear(channels, channels, bias=False)
    def reset_parameters(self):
        nn.init.zeros_(self.gate_net.weight)
        nn.init.zeros_(self.gate_net.bias)
        nn.init.eye_(self.var_transform.weight)
    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2, max_num_elements=None):
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        mean_of_sq = self.reduce(x * x, index, ptr, dim_size, dim, reduce='mean')
        var = (mean_of_sq - mean * mean).clamp(min=1e-8)
        ones = x.new_ones(x.size(0), 1)
        cardinality = self.reduce(ones, index, ptr, dim_size, dim, reduce='sum')
        log_card = torch.log1p(cardinality)
        gate = torch.sigmoid(self.gate_net(log_card))
        return mean + gate * self.var_transform(var)

class CAMAHeteroGraphSAGE(nn.Module):
    def __init__(self, node_types, edge_types, channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            edge_conv_dict = {}
            for et in edge_types:
                cama = CAMAAggregation(channels)
                edge_conv_dict[et] = SAGEConv((channels, channels), channels, aggr=cama)
            conv = HeteroConv(edge_conv_dict, aggr="sum")
            self.convs.append(conv)
            norm_dict = nn.ModuleDict()
            for nt in node_types:
                norm_dict[nt] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)
    def reset_parameters(self):
        for conv in self.convs: conv.reset_parameters()
        for nd in self.norms:
            for n in nd.values(): n.reset_parameters()
    def forward(self, x_dict, edge_index_dict, num_sampled_nodes_dict=None, num_sampled_edges_dict=None):
        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: norm_dict[k](v) for k, v in x_dict.items()}
            x_dict = {k: v.relu() for k, v in x_dict.items()}
        return x_dict

class Model(nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels, out_channels, use_cama=False):
        super().__init__()
        self.encoder = HeteroEncoder(channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict)
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels)
        if use_cama:
            self.gnn = CAMAHeteroGraphSAGE(data.node_types, data.edge_types, channels, num_layers)
        else:
            self.gnn = HeteroGraphSAGE(data.node_types, data.edge_types, channels, aggr="mean", num_layers=num_layers)
        self.head = MLP(channels, out_channels=out_channels, norm="batch_norm", num_layers=1)
    def reset_parameters(self):
        self.encoder.reset_parameters(); self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters(); self.head.reset_parameters()
    def forward(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rt in rel_time_dict.items(): x_dict[nt] = x_dict[nt] + rt
        x_dict = self.gnn(x_dict, batch.edge_index_dict, batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
        return self.head(x_dict[entity_table][:seed_time.size(0)])

class GloveTextEmbedding:
    def __init__(self, device=None):
        self.model = SentenceTransformer("sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)
    def __call__(self, sentences):
        return torch.from_numpy(self.model.encode(sentences, show_progress_bar=False))

def main():
    os.makedirs(ROOT_DIR, exist_ok=True)

    # Load rel-trial dataset
    logger.info("Loading rel-trial...")
    t0 = time.time()
    dataset = get_dataset("rel-trial", download=True)
    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    text_embedder = GloveTextEmbedding(device=DEVICE)
    text_embedder_cfg = TextEmbedderConfig(text_embedder=text_embedder, batch_size=256)
    data, col_stats_dict = make_pkey_fkey_graph(
        db, col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=os.path.join(ROOT_DIR, "rel-trial_cache"))
    logger.info(f"  Loaded in {time.time()-t0:.1f}s")
    logger.info(f"  Node types: {data.node_types}")
    logger.info(f"  Edge types: {data.edge_types}")

    # Get task
    task = get_task("rel-trial", "study-outcome", download=True)
    entity_table = task.entity_table
    logger.info(f"  Entity table: {entity_table}")

    # Loaders
    loader_dict = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(data, num_neighbors=[64, 64], time_attr="time",
            input_nodes=table_input.nodes, input_time=table_input.time,
            transform=table_input.transform, batch_size=512,
            temporal_strategy="uniform", shuffle=(split == "train"),
            num_workers=0, persistent_workers=False)
    logger.info(f"  Loaders ready: train={len(loader_dict['train'])}, val={len(loader_dict['val'])}")

    for method in ['baseline', 'cama']:
        logger.info(f"\n--- Method: {method} ---")
        seed_everything(42)
        use_cama = (method == 'cama')
        model = Model(data, col_stats_dict, 2, 128, 1, use_cama).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = BCEWithLogitsLoss()

        for epoch in range(1, 3):
            model.train()
            losses, count = 0.0, 0
            for step, batch in enumerate(loader_dict["train"]):
                if step >= 50: break
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                pred = model(batch, entity_table).view(-1)
                loss = loss_fn(pred.float(), batch[entity_table].y.float())
                loss.backward()
                optimizer.step()
                losses += loss.item() * pred.size(0)
                count += pred.size(0)
            logger.info(f"  Epoch {epoch}: loss={losses/count:.4f}")

        # Test
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader_dict["test"]:
                batch = batch.to(DEVICE)
                pred = torch.sigmoid(model(batch, entity_table).view(-1))
                preds.append(pred.cpu())
        test_pred = torch.cat(preds).numpy()
        test_metrics = task.evaluate(test_pred)
        logger.info(f"  Test metrics: {test_metrics}")
        logger.info(f"  Pred stats: mean={test_pred.mean():.4f}, std={test_pred.std():.4f}")

        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("\nSMOKE TEST PASSED!")

if __name__ == "__main__":
    main()
