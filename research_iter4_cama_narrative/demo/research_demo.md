# CAMA Narrative

## Summary

Comprehensive research across four threads for the CAMA paper: (1) Six precedent papers identified at NeurIPS/ICML/ICLR for scope framing, with reusable framing patterns (underestimated potential, plug-in module, aggregation-is-bottleneck). (2) Mechanism narrative drafted with CRITICAL revision: "rank collapse" must be replaced with "information compression" since Roth & Liebig's rank collapse is depth-wise across layers, not within-step at FK aggregation. (3) CAMA recommended as best name (low collision, domain-specific, mechanistically accurate); AMA eliminated (Adam collision); RAMA suboptimal. (4) Complete NeurIPS-style 9-page outline with claims-to-evidence mapping and 5-6 figures. KEY FINDING: RelBench uses sum aggregation by default (overriding SAGEConv's mean), which strengthens the narrative -- RelBench's designers independently recognized mean's limitations at FK joins.

## Research Findings

## Thread 1: Scope Framing — Precedent Papers

Six precedent papers were identified that successfully contributed improvements to basic/commodity GNN architectures while acknowledging SOTA solves problems differently:

1. **"Classic GNNs are Strong Baselines"** (NeurIPS 2024) demonstrated that with proper hyperparameter tuning, classic GNNs match or exceed Graph Transformers on 17/18 datasets, framing it as "the potential of message-passing GNNs may have been previously underestimated" [1].

2. **"Can Classic GNNs Be Strong Baselines"** (ICML 2025) extended this to graph-level tasks, emphasizing "greater efficiency, running several times faster than GTs" and "challenging the notion that complex mechanisms in GTs are essential" [2].

3. **SGC** (ICML 2019) framed simplification as contribution: "GCNs derive inspiration primarily from recent deep learning approaches, and as a result, may inherit unnecessary complexity and redundant computation" [3].

4. **GenAgg** (NeurIPS 2023) provided the CRITICAL quote for CAMA: "GNNs struggle to 'make up for' the lack of representational complexity in their constituent aggregators, even when using state-of-the-art parametrised aggregators" [4]. This directly supports CAMA's thesis that the aggregation primitive is the bottleneck.

5. **PNA** (NeurIPS 2020) demonstrated "the requirement for multiple aggregation functions" in continuous feature spaces, combining mean, std, min, max with degree scalers [5]. Critically, PNA applies all aggregators UNIFORMLY with no gating — CAMA's differentiator is GATED variance injection conditioned on cardinality and effective rank.

6. **DropEdge** (ICLR 2020) provided the ideal framing template: "a simple, general-purpose, plug-in technique that can be equipped with many other backbone models for enhanced performance" [13].

Four reusable framing patterns were distilled: (A) Practical impact of improving widely-deployed defaults, (B) Scope as feature not limitation, (C) Complementary contribution, (D) DropEdge-style plug-in framing.

## Thread 2: Mechanism Narrative

### CRITICAL REVISION: Rank Collapse vs Information Compression

Roth & Liebig (LoG 2024) define rank collapse as a DEPTH-wise phenomenon: "the rank of node representations collapses" as layers increase, bounded by "the joint algebraic multiplicity of all eigenvalues" of the aggregation matrix [7]. Their analysis explicitly operates across layers, not within a single step. CAMA addresses a DIFFERENT phenomenon: within-step information compression at FK aggregation. The paper must use "information compression" or "distributional compression" terminology, reserving "rank collapse" for depth-wise citations.

### RelGNN Aggregation (Sum-Based)

RelGNN uses attention-weighted aggregation with softmax normalization per FK join, then simple summation across atomic routes with NO degree normalization [6]. This preserves distributional information because: (a) attention weights vary per-neighbor, and (b) no 1/N division occurs.

### Sum vs Mean: Theoretical Foundation

Xu et al. (ICLR 2019) proved that sum aggregation is injective over multisets (Lemma 5) while mean is NOT — mean cannot distinguish multisets with different cardinalities but identical proportions [9]. This is the theoretical foundation for why sum-based architectures don't need CAMA.

### VPA vs CAMA: Different Goals

GNN-VPA (ICLR 2024) normalizes by 1/sqrt(N) for TRAINING STABILITY: "the random variable y = (1/sqrt(N)) sum(z_n) has the same mean and variance as z" (Lemma 1) [8]. CAMA has a fundamentally different goal: FEATURE ENRICHMENT — injecting variance as informative signal, not normalizing it away.

### CRITICAL FINDING: RelBench Uses Sum by Default

RelBench uses "the heterogeneous version of the GraphSAGE model with sum-based neighbor aggregation" [11], OVERRIDING SAGEConv's default of mean [15]. This is powerful evidence: RelBench's designers independently recognized that mean aggregation is problematic at FK joins. CAMA provides an alternative — enrich mean rather than abandon it. This is particularly valuable for the vast existing deployment base of GraphSAGE with default mean aggregation.

### Effective Rank Definition

Roy & Vetterli (EUSIPCO 2007): erank(A) = exp(-sum(rho_i * ln(rho_i))) where rho_i are normalized eigenvalues [10, 16]. CAMA uses a variance-entropy proxy that is O(Nd) — no SVD needed.

## Thread 3: CAMA vs RAMA vs AMA Naming

**AMA is ELIMINATED** due to fatal collision with Adam (Adaptive Moment Estimation). While Adam is technically "not an acronym" per Cornell's reference [17], it is universally associated with "Adaptive Moment." An "Adaptive Moment Aggregation" would confuse every reviewer.

**RAMA has MEDIUM collision risk** — "Rama" exists as a bioinformatics ML tool (Nature Scientific Reports, 2017) and "RAMA" as a meta-algorithmic framework (MDPI 2025). Additionally, "rank" is heavily overloaded in ML (tensor rank, matrix rank, learning-to-rank).

**CAMA (Cardinality-Aware Moment Aggregation) is RECOMMENDED** because: (a) "Cardinality" is THE distinguishing property of FK joins in RDL; (b) the gate IS conditioned on log(cardinality); (c) no known ML collisions (CAM = Class Activation Mapping is different enough); (d) 4-letter pronounceable acronym. "Cardinality Preserved Attention" (CPA) exists [14] but uses a different acronym and validates the "cardinality" concept.

## Thread 4: Paper Outline

NeurIPS 2025 confirmed: 9 content pages + unlimited appendix [12].

The outline maps every claim to specific experiments:
- **Introduction** (1.5pp): FK aggregation bottleneck, mean loses information (spectral analysis evidence), CAMA mechanism
- **Related Work** (1pp): PNA/GenAgg/VPA (homogeneous), Roth depth-wise rank collapse (different phenomenon), RelBench/RelGNN (sum/attention avoid problem)
- **Method** (2pp): Information compression proposition, CAMA formula (mu + g * W_sigma * sigma^2), O(Nd) cost, 2d+2 parameters
- **Experiments** (3pp): Forest plot (Cohen's d), spectral analysis, gate heatmap, ablations (gated vs ungated, gate inputs, sum comparison)
- **Discussion** (0.5pp): Scope delineation, PRMP paradox resolution, safety (gate closes when inappropriate)

**Key experimental design note**: Base architecture must use mean aggregation (not RelBench's default sum). Sum is included as a BASELINE to show CAMA brings mean closer to sum's performance [11, 19].

**5 main figures**: (1) Motivating compression visualization, (2) Architecture diagram, (3) Forest plot, (4) Compression-vs-improvement scatter, (5) Gate heatmap.

## Sources

[1] [Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification (NeurIPS 2024)](https://arxiv.org/html/2406.08993v1) — Precedent paper demonstrating classic GNNs match Graph Transformers with proper tuning; key framing: 'potential previously underestimated'

[2] [Can Classic GNNs Be Strong Baselines for Graph-Level Tasks (ICML 2025)](https://arxiv.org/html/2502.09263v2) — Extended classic-GNN-are-strong-baselines argument to graph-level tasks; efficiency + capability framing

[3] [Simplifying Graph Convolutional Networks - SGC (ICML 2019)](https://proceedings.mlr.press/v97/wu19e.html) — Precedent for simplification-as-contribution: GCNs 'inherit unnecessary complexity'

[4] [Generalised f-Mean Aggregation for Graph Neural Networks - GenAgg (NeurIPS 2023)](https://arxiv.org/html/2306.13826) — Critical quote: 'GNNs struggle to make up for the lack of representational complexity in their constituent aggregators'

[5] [Principal Neighbourhood Aggregation for Graph Nets - PNA (NeurIPS 2020)](https://arxiv.org/abs/2004.05718) — Multi-aggregator with degree scalers; uses ALL aggregators uniformly with NO gating (CAMA differentiator)

[6] [RelGNN: Relational Graph Neural Network (2025)](https://arxiv.org/html/2502.06784v2) — Uses attention-weighted aggregation + sum across atomic routes; no degree normalization

[7] [Rank Collapse Causes Over-Smoothing and Over-Correlation in GNNs (LoG 2024)](https://arxiv.org/html/2308.16800v3) — Rank collapse is DEPTH-wise across layers, not within-step; rank bounded by eigenvalue multiplicity

[8] [GNN-VPA: Variance-Preserving Aggregation (ICLR 2024)](https://arxiv.org/html/2403.04747v1) — VPA normalizes by 1/sqrt(N) for training stability; fundamentally different from CAMA's feature enrichment

[9] [How Powerful are Graph Neural Networks - GIN (ICLR 2019)](https://arxiv.org/abs/1810.00826) — Sum aggregation is injective over multisets (Lemma 5); mean is NOT injective

[10] [The Effective Rank: A Measure of Effective Dimensionality (Roy & Vetterli, EUSIPCO 2007)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf) — erank(A) = exp(Shannon entropy of normalized singular values); formal definition

[11] [RelBench: A Benchmark for Deep Learning on Relational Databases (NeurIPS 2024)](https://arxiv.org/html/2407.20060v1) — Uses SUM aggregation by default, overriding SAGEConv's mean; evidence mean is problematic at FK joins

[12] [NeurIPS 2025 Call for Papers](https://neurips.cc/Conferences/2025/CallForPapers) — 9 content pages + unlimited appendix; LaTeX required; double-blind review

[13] [DropEdge: Towards Deep Graph Convolutional Networks (ICLR 2020)](https://arxiv.org/abs/1907.10903) — Precedent for 'simple, general-purpose, plug-in technique' framing pattern

[14] [Improving Attention Mechanism in GNNs via Cardinality Preservation (2019)](https://arxiv.org/abs/1907.02204) — CPA exists with different acronym; validates 'cardinality' concept in GNN aggregation

[15] [PyTorch Geometric SAGEConv Documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html) — SAGEConv defaults to aggr='mean'; most widely used GNN layer

[16] [Effective Rank Mathematical Definition (Portfolio Optimizer)](https://portfoliooptimizer.io/blog/the-matrix-effective-rank-measuring-the-dimensionality-of-a-universe-of-assets/) — Readable formulation: erank(A) = exp(-sum(rho_i ln(rho_i))) with normalized eigenvalues

[17] [Adam Optimizer (Cornell Optimization Wiki)](https://optimization.cbe.cornell.edu/index.php?title=Adam) — Adam 'is not an acronym' but universally associated with Adaptive Moment Estimation

[18] [Relational Deep Learning: Challenges and Next-Gen Architectures (2025)](https://arxiv.org/html/2506.16654v1) — Bridge/hub nodes cause information loss; RelGNN addresses via atomic routes

[19] [RDB2G-Bench: Comprehensive Benchmark for Automatic Graph Modeling of RDBs](https://arxiv.org/html/2506.01360v1) — Benchmarks both sum and mean GraphSAGE; high Spearman correlation (>0.8) between them

## Follow-up Questions

- What specific RelBench tasks from the first draft should be prioritized for CAMA experiments, given that the base architecture must use mean aggregation (not RelBench's default sum)?
- Should the paper include a formal proposition on information compression bounds at FK aggregation (e.g., variance decreases as O(1/N)), or keep it empirical with spectral analysis?
- How should the experimental comparison between mean+CAMA vs plain sum aggregation be framed — as demonstrating CAMA 'recovers' sum-level performance, or as providing complementary benefits beyond what sum offers?

---
*Generated by AI Inventor Pipeline*
