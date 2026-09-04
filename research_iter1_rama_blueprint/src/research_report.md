# RAMA Blueprint

## Summary

Complete code-level specification for RAMA (Rank-Aware Moment Aggregation) as a PyG Aggregation subclass. Covers: exact forward() API matching PyG's Aggregation base class, verified integration hooks in RelBench HeteroSAGE and RelGNN architectures (no source code changes needed thanks to PyG's isinstance(self.aggr, str) fuse check), hyperparameter grid derived from RelBench/RelGNN defaults, evaluation protocol matching PRMP first draft (Cohen's d, DerSimonian-Laird meta-analysis, 5 seeds, 21 entity-level tasks), complete risk register with mitigations, and per-dataset RelGNN configs for all 27 task combinations.

## Research Findings

## SECTION A: PyG Aggregation API Surface

RAMA must subclass `torch_geometric.nn.aggr.Aggregation`, PyG's abstract base for all aggregation operations [1]. The required `forward()` signature is:

```python
def forward(self, x: Tensor, index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
            dim: int = -2, max_num_elements: Optional[int] = None) -> Tensor
```

Where `x` is [num_edges, d] source features, `index` is [num_edges] mapping edges to target nodes, and the output is [dim_size, d] [1, 28].

The base class provides a `reduce()` helper for scatter/segment reductions accepting a `reduce` string ('sum', 'mean', etc.) [1]. PyG's built-in `VarAggregation` computes variance as `E[X^2] - E[X]^2` via two `reduce()` calls, with an optional `semi_grad` flag for memory savings [2].

`MultiAggregation` dispatches to child aggregators and combines outputs via configurable modes (cat, proj, attention, sum, mean) [3]. `DegreeScalerAggregation` (PNA) wraps base aggregators with degree-based scaling using 5 handcrafted scalers (identity, amplification, attenuation, linear, inverse_linear) [4]. This is the closest existing mechanism to RAMA's cardinality conditioning, but PNA uses fixed functions of degree only, while RAMA learns a gating function of both cardinality AND variance entropy.

## SECTION B: HeteroSAGE Integration Points

`HeteroGraphSAGE` creates per-edge-type `SAGEConv((channels, channels), channels, aggr=aggr)` wrapped in `HeteroConv(aggr="sum")` [5]. The within-edge-type `aggr` (default "mean") is where RAMA replaces mean aggregation; the cross-edge-type `aggr="sum"` is NOT touched [5].

SAGEConv accepts `Union[str, List[str], Aggregation]` [6], so a custom Aggregation instance can be passed directly.

**CRITICAL FINDING — message_and_aggregate bypass risk is RESOLVED**: SAGEConv implements `message_and_aggregate()` calling `spmm()`, but PyG's MessagePassing base sets `self.fuse &= isinstance(self.aggr, str) and self.aggr in FUSE_AGGRS` [7]. Since a custom Aggregation object fails the `isinstance` check, `self.fuse = False`, and PyG falls back to separate `message()` → `aggregate()` path where `aggregate()` delegates to the custom Aggregation's `forward()` [7, 8]. **No SAGEConv source code changes are needed.**

Parameter flow: `args.aggr` → `Model(aggr=)` → `HeteroGraphSAGE(aggr=)` → `SAGEConv(aggr=)` [9, 10].

## SECTION C: RelGNN Integration Points

RelGNN has three aggregation points [11, 12, 14]:

1. **Within-route SAGEConv (PRIMARY TARGET)**: `self.aggr_conv = SAGEConv(in_channels, out_channels, aggr=aggr)` in RelGNNConv for dim-fact-dim FK-join aggregation [11].
2. **Cross-route summation (NOT targeted)**: `group()` function in RelGNN_HeteroConv sums outputs from atomic routes [12].
3. **Attention (NOT targeted)**: TransformerConv for query-to-destination [14].

Per-dataset configs: most use aggr="sum", channels=128; rel-trial study-adverse/site-success use aggr="mean" with num_neighbors=64 [16].

RAMA integration: replace `SAGEConv(in_ch, out_ch, aggr=aggr)` with `SAGEConv(in_ch, out_ch, aggr=RAMAAggregation(out_ch))` in relgnn_conv.py [11].

## SECTION D: RAMA Module Specification

The complete RAMA class subclasses `Aggregation` with constructor parameters `channels` (embedding dim d), `gate_hidden` (default 16), and `eps` (default 1e-8). Learnable parameters: `W_sigma` (Linear(d,d)), `W_g` (Linear(2, gate_hidden)), `gate_out` (Linear(gate_hidden, d)). All zero-initialized so RAMA starts as pure mean aggregation [2].

Forward computation at each step with tensor shapes:
1. `mu = reduce(x, 'mean')` → [N_parents, d]
2. `var = reduce(x*x, 'mean') - mu*mu` → [N_parents, d] (matching VarAggregation pattern [2])
3. `N = reduce(ones, 'sum')` → [N_parents, 1] (cardinality)
4. `rho = H(var_normalized) / log(d)` → [N_parents, 1] (effective rank proxy)
5. `g = sigmoid(gate_out(relu(W_g([log1p(N), rho]))))` → [N_parents, d]
6. `out = mu + g * W_sigma(var)` → [N_parents, d]

The effective rank proxy uses per-dimension variance entropy instead of SVD [18, 19]: O(Nd) vs O(Nd^2). Validated by RankMe's use of spectral entropy for representation quality measurement [20].

Unlike PRMP's `.detach()` approach, RAMA has full gradient flow through all components [17]. The sigmoid gate prevents gradient explosion; zero initialization ensures safe warm-start.

## SECTION E: Hyperparameter Grid & Evaluation

RelBench defaults [9, 21]: lr=0.005, epochs=10, batch_size=512, channels=128, num_layers=2, num_neighbors=128, aggr="sum".

RAMA-specific grid: gate_hidden {8,16,32}, variance_transform {"none","log1p"}, rank_proxy {"variance_entropy","none"}, init {"zero","small_random"}, lr {0.001,0.005}, channels {64,128}.

Evaluation: 21 entity-level tasks (12 classification ROC-AUC + 9 regression MAE) across 7 datasets [21]. Minimum 5 seeds. Statistical analysis: Cohen's d with pooled SD, DerSimonian-Laird random-effects meta-analysis, Cochran's Q, Egger's test — matching PRMP first draft protocol [22].

Seven baselines: (A) mean, (B) sum, (C) PRMP, (D) mean+LayerNorm, (E) ungated mu+W_sigma(var), (F) PNA MultiAggregation [23], (G) VPA 1/sqrt(N) scaling.

## SECTION F: Risk Analysis

**Risk 1 (Bypass) — RESOLVED**: PyG's `isinstance(self.aggr, str)` check prevents fused mode for custom Aggregation objects [7, 8]. No code changes needed.

**Risk 2 (Two-pass variance)**: Use VarAggregation's `E[X^2]-E[X]^2` pattern with `.clamp(min=0)` [2, 24].

**Risk 3 (Numerical stability)**: eps=1e-8 floor before log; N=1 yields zero variance (gate irrelevant); N=0 handled by PyG [1, 24].

**Risk 4 (Memory)**: ~2x mean-only aggregation, but less than PNA's 4x [4, 23].

**Risk 5 (GPU non-determinism)**: Inherent to scatter ops; mitigated with ≥5 seeds [24].

**Risk 6 (Type annotation)**: Model class types aggr as str; update to Union[str, Aggregation] [10, 27].

## Sources

[1] [PyG Aggregation Base Class Source Code](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/torch_geometric/nn/aggr/base.py) — Provided exact forward() signature RAMA must implement and documented the reduce() helper method.

[2] [PyG VarAggregation, StdAggregation Source Code](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/torch_geometric/nn/aggr/basic.py) — Revealed VarAggregation computes variance as E[X^2]-E[X]^2 with semi_grad option. Established pattern for RAMA variance computation.

[3] [PyG MultiAggregation Source Code](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/torch_geometric/nn/aggr/multi.py) — Documented how multiple aggregators are dispatched and combined via cat/attention/proj modes.

[4] [PyG DegreeScalerAggregation (PNA) Source Code](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/torch_geometric/nn/aggr/scaler.py) — Documented PNA's 5 degree-based scalers. Established PNA as closest existing mechanism to RAMA's cardinality conditioning.

[5] [RelBench HeteroGraphSAGE Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/relbench/modeling/nn.py) — Confirmed SAGEConv instantiation with aggr parameter and HeteroConv wrapping with aggr='sum'.

[6] [PyG SAGEConv Source Code](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/torch_geometric/nn/conv/sage_conv.py) — Confirmed aggr accepts Union[str, List[str], Aggregation]. Revealed message_and_aggregate() calls spmm directly.

[7] [PyG MessagePassing Base Class](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/torch_geometric/nn/conv/message_passing.py) — CRITICAL: Found isinstance(self.aggr, str) check preventing fused execution for custom Aggregation objects.

[8] [PyG LSTM Aggregator PR Discussion](https://github.com/pyg-team/pytorch_geometric/pull/4379) — Confirmed self.fuse=False mechanism for non-standard aggregations from PyG maintainer.

[9] [RelBench gnn_entity.py Training Script](https://raw.githubusercontent.com/snap-stanford/relbench/main/examples/gnn_entity.py) — Documented default hyperparameters: lr=0.005, epochs=10, batch_size=512, channels=128, aggr='sum'.

[10] [RelBench Model Class Source Code](https://raw.githubusercontent.com/snap-stanford/relbench/main/examples/model.py) — Traced aggr parameter flow from Model to HeteroGraphSAGE. Documented complete model architecture.

[11] [RelGNN RelGNNConv Source Code](https://raw.githubusercontent.com/snap-stanford/RelGNN/main/examples/relgnn_conv.py) — Confirmed self.aggr_conv = SAGEConv(in_channels, out_channels, aggr=aggr) for dim-fact-dim routes.

[12] [RelGNN HeteroConv Source Code](https://raw.githubusercontent.com/snap-stanford/RelGNN/main/examples/relgnn_hetero_conv.py) — Documented cross-route aggregation via group() function. RAMA targets only within-route SAGEConv.

[13] [RelGNN Neural Network Architecture](https://raw.githubusercontent.com/snap-stanford/RelGNN/main/examples/relgnn_nn.py) — Traced aggr propagation through RelGNN layers with LayerNorm and ReLU.

[14] [RelGNN Paper (ICML 2025)](https://arxiv.org/html/2502.06784v2) — Extracted all equations: FUSE (Eq.4), AGGR (Eq.5), Composite MP (Eq.3). Confirmed SOTA on 27/30 tasks.

[15] [RelGNN Model Class](https://raw.githubusercontent.com/snap-stanford/RelGNN/main/examples/relgnn_model.py) — Documented model architecture and aggr parameter flow to RelGNN core.

[16] [RelGNN Per-Dataset Configs (get_configs)](https://raw.githubusercontent.com/snap-stanford/RelGNN/main/examples/utils.py) — Extracted all 27 dataset-task configs. Most use aggr='sum', channels=128; rel-trial exceptions use aggr='mean'.

[17] [PRMP First Draft Repository](https://github.com/ai-inventor-papers/ai-invention-b2d5b0-predictive-residual-message-passing-filt) — Documented PRMP evaluation: 5 variants x 5 seeds x 3 tasks, gradient-detached ablations, 8-10 tasks across 5 datasets.

[18] [Roy & Vetterli 2007: Effective Rank Definition](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf) — Original erank(A) = exp(-sum p_i log p_i) formula with normalized singular values. Bounds: 1 <= erank <= rank.

[19] [Effective Rank Overview](https://www.emergentmind.com/topics/effective-rank-erank) — Comprehensive definition: erank = exp(H(M)), alternative form r_e = trace(S)/||S||_2, properties and bounds.

[20] [RankMe: Matrix Information Theory for SSL](https://arxiv.org/pdf/2305.17326) — Validated spectral entropy as proxy for representation quality in neural networks.

[21] [RelBench Paper (NeurIPS 2024)](https://arxiv.org/html/2407.20060v1) — Complete list of 30 tasks across 7 datasets with Table 9 hyperparameters. Two exceptions from defaults documented.

[22] [PRMP Definitive Evaluation Script](https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-b2d5b0-predictive-residual-message-passing-filt/main/evaluation_iter7_definitive_prmp/src/eval.py) — Cohen's d computation, DerSimonian-Laird meta-analysis, Cochran's Q, Egger's test. 5 seeds, 8-10 tasks.

[23] [PNA: Principal Neighbourhood Aggregation (NeurIPS 2020)](https://arxiv.org/abs/2004.05718) — PNA combines mean/std/min/max aggregators with degree scalers. Strongest multi-aggregator baseline.

[24] [torch_scatter Documentation](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html) — Documented scatter operations, GPU non-determinism, and two-pass scatter_std implementation.

[25] [Oversquashing/Oversmoothing Mitigation Survey](https://arxiv.org/html/2411.17429v1) — Context for how variance injection may interact with oversmoothing in GNNs.

[26] [Entropy Aware Message Passing in GNNs](https://arxiv.org/html/2403.04636) — Precedent for entropy-based gating in GNN message passing.

[27] [PyG Aggregation Resolver](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/torch_geometric/nn/resolver.py) — Confirmed custom Aggregation objects are passed through correctly by aggr_resolver.

[28] [PyG Aggregation Class Documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.Aggregation.html) — Official documentation for forward() parameters and usage patterns.

## Follow-up Questions

- How does RAMA's variance entropy proxy correlate with the true SVD-based effective rank across different RelBench datasets, and is the O(Nd) approximation sufficiently accurate to drive meaningful gating decisions?
- What is the interaction between RAMA's per-dimension gating and RelGNN's TransformerConv attention mechanism -- does gated variance injection before attention help or conflict with attention-based aggregation?
- How does the RAMA gate behave on rel-trial tasks where the baseline already requires switching from sum to mean aggregation -- does RAMA automatically learn to close the gate (approaching mean) on these tasks?

---
*Generated by AI Inventor Pipeline*
