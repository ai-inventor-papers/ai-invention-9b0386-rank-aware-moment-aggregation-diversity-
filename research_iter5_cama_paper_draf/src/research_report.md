# CAMA Paper Draft

## Summary

Complete NeurIPS 2025 paper draft for Cardinality-Aware Moment Aggregation (CAMA), a lightweight drop-in module enriching mean aggregation at FK joins in relational deep learning with gated per-dimension variance injection. Contains all sections in full prose (~5,850 words): Abstract, Introduction, Related Work (4 subsections), Method (4 subsections with Algorithm 1), Experiments (7 subsections with 3 table and 5 figure placeholders), Discussion and Limitations, Conclusion. All 18 bibliography entries verified with correct authors/venues/arXiv IDs. 17-item claim-evidence map distinguishing 10 established claims from 7 pending experimental validations. 11 RelBench tasks across 5 databases verified. Terminology constraints enforced throughout (information compression, not rank collapse; CAMA, not RAMA/AMA). Novelty confirmed via negative search: no prior work combines gated variance injection with cardinality conditioning at FK joins.

## Research Findings

The complete CAMA NeurIPS 2025 paper draft has been produced across two files: research_report.md (full prose, ~315 lines, ~5,850 content words) and research_out.json (structured metadata with all specifications).

## Key Research Findings

### 1. Novelty Confirmed
No prior work combines per-dimension variance as informative signal (not normalization) with cardinality-gated injection specific to FK joins [4, 5, 11]. The closest methods are:
- **PNA** [5]: applies all 12 aggregator-scaler combinations uniformly with no gating; degree scalers conflate cardinality with diversity
- **GNN-VPA** [4]: uses 1/sqrt(N) normalization for training stability -- treats variance as nuisance, not information
- **GenAgg** [11]: learns parametric f-mean globally, no per-node adaptation or diversity sensing

### 2. Theoretical Anchor Verified
Xu et al. (ICLR 2019) Lemma 5 proves sum aggregation can represent injective functions over multisets of bounded size: there exists f such that h(X) = sum f(x) is unique for each multiset X [2, 22]. Mean and max cannot distinguish certain non-isomorphic multisets. This directly motivates why mean aggregation loses information at FK joins.

### 3. RelBench Task Selection Verified
All 11 proposed experimental tasks confirmed as real RelBench tasks with correct metrics [17, 18, 19, 20, 21]:
- rel-amazon: user-churn (AUROC), user-ltv (MAE), item-churn (AUROC)
- rel-f1: driver-dnf (AUROC), driver-position (MAE)
- rel-stack: user-engagement (AUROC), post-votes (MAE)
- rel-hm: user-churn (AUROC), item-sales (MAE)
- rel-avito: user-visits (AUROC), ad-ctr (MAE)

### 4. Terminology Distinction Validated
'Information compression' is correctly distinguished from 'rank collapse' [8]. Roth & Liebig's rank collapse is depth-wise across GNN layers (Proposition 5.6: amplification depends only on aggregation matrix) [8]. CAMA targets within-step compression at a single FK aggregation. Nait Saada et al.'s width-wise collapse in attention [10] is the closest conceptual relative but proposes rank-one subtraction, not applicable to FK joins with variable cardinality.

### 5. Citation Details Verified
All 18 references confirmed with correct authors, venues, years, and arXiv IDs [1-23]:
- RelGNN: Chen, Kanatsoulis, Leskovec (ICML 2025) [6]
- Griffin: Wang, Wang, Gan, Wang, Yang, Wipf, Zhang (ICML 2025) -- **corrected** from dependency which listed 'Pelevin' [7]
- RelBench: Robinson et al., 12 authors (NeurIPS 2024 D&B) [1]
- GNN-VPA: Schneckenreiter et al. (ICLR 2024 Tiny Paper) [4]

### 6. NeurIPS 2025 Format Confirmed
9 content pages + unlimited appendix, LaTeX required, submission deadline May 15, 2025 AOE [14]. The draft targets approximately 5,850 words fitting within: Abstract (0.25p), Intro (1.75p), Related Work (1p), Method (1.75p), Experiments (3p), Discussion (0.75p), Conclusion (0.25p).

### 7. Paper Structure Completeness
The draft contains:
- **5 figure specifications** (motivating diagram, architecture, compression measurement, forest plot, gate heatmap)
- **3 table specifications** (main results 11 tasks x 4 methods, ablation 6 variants, RelGNN scope validation)
- **17-item claim-evidence map** (10 established, 7 pending experiment)
- **18 bibliography entries** organized by category
- **6 terminology constraints** enforced throughout
- **Appendix outline** with 6 sections (A-F)

### 8. Scope Honesty Maintained
Every section acknowledges CAMA targets mean-aggregation architectures only. Table 3 (RelGNN integration) explicitly designed as a negative-result scope validation. The paper never claims CAMA is "state-of-the-art" -- it "bridges the gap between mean and sum aggregation."

### Confidence Assessment
**HIGH confidence** in: novelty claim, citation accuracy, task selection, terminology, paper structure, theoretical grounding.
**MEDIUM confidence** in: exact page fit (may need minor adjustments during LaTeX compilation), experimental design choices (hyperparameter selection pending pilot runs).
**PENDING**: All numerical results in Tables 1-3 and Figures 3-5 require experimental validation before submission.

## Sources

[1] [RelBench: A Benchmark for Deep Learning on Relational Databases (NeurIPS 2024)](https://arxiv.org/abs/2407.20060) — Primary benchmark paper. Confirmed sum aggregation default, 11 databases, 30+ tasks. Authors verified.

[2] [How Powerful are Graph Neural Networks? (Xu et al., ICLR 2019)](https://arxiv.org/abs/1810.00826) — Theoretical anchor: Lemma 5 proves sum injective over multisets, mean/max are not.

[3] [Inductive Representation Learning on Large Graphs (Hamilton et al., NeurIPS 2017)](https://arxiv.org/abs/1706.02216) — GraphSAGE paper establishing mean aggregation as default.

[4] [GNN-VPA: Variance-Preserving Aggregation (Schneckenreiter et al., ICLR 2024)](https://arxiv.org/abs/2403.04747) — VPA normalizes variance for stability; fundamentally different from CAMA's enrichment.

[5] [PNA: Principal Neighbourhood Aggregation (Corso et al., NeurIPS 2020)](https://arxiv.org/abs/2004.05718) — Multi-aggregator baseline with uniform application and degree scalers.

[6] [RelGNN: Composite Message Passing (Chen et al., ICML 2025)](https://arxiv.org/abs/2502.06784) — SOTA on RelBench via atomic routes and attention aggregation.

[7] [Griffin: Graph-Centric RDB Foundation Model (Wang et al., ICML 2025)](https://arxiv.org/abs/2505.05568) — Foundation model with mean intra-relation aggregation. Author correction: no 'Pelevin'.

[8] [Rank Collapse in GNNs (Roth & Liebig, LoG 2024)](https://arxiv.org/abs/2308.16800) — Depth-wise rank collapse theory. Distinct from CAMA's within-step target.

[9] [Attention Loses Rank Doubly Exponentially (Dong et al., ICML 2021)](https://arxiv.org/abs/2103.03404) — Depth-wise collapse in transformers mitigated by skip connections.

[10] [Mind the Gap: Rank Collapse in Attention (Nait Saada et al., 2024)](https://arxiv.org/abs/2410.07799) — Width-wise collapse as context grows; closest conceptual relative to CAMA.

[11] [GenAgg: Generalised f-Mean Aggregation (Kortvelesy et al., NeurIPS 2023)](https://arxiv.org/abs/2306.13826) — Learned parametric aggregator, globally shared without per-node adaptation.

[12] [GNN Bottleneck and Over-squashing (Alon & Yahav, ICLR 2021)](https://arxiv.org/abs/2006.05205) — Over-squashing phenomenon distinct from information compression.

[13] [Effective Rank (Roy & Vetterli, EUSIPCO 2007)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf) — Effective rank definition used for CAMA extension variant.

[14] [NeurIPS 2025 Call for Papers](https://neurips.cc/Conferences/2025/CallForPapers) — Confirmed 9 content pages, LaTeX, May 15 deadline.

[15] [Graph Attention Networks (Velickovic et al., ICLR 2018)](https://arxiv.org/abs/1710.10903) — GAT reference for attention-weighted aggregation.

[16] [GCN (Kipf & Welling, ICLR 2017)](https://arxiv.org/abs/1609.02907) — Foundational GCN reference.

[17] [RelBench rel-amazon tasks](https://relbench.stanford.edu/datasets/rel-amazon/) — Verified: user-churn, item-churn (AUROC), user-ltv, item-ltv (MAE).

[18] [RelBench rel-f1 tasks](https://relbench.stanford.edu/datasets/rel-f1/) — Verified: driver-dnf, driver-top3 (AUROC), driver-position (MAE).

[19] [RelBench rel-stack tasks](https://relbench.stanford.edu/datasets/rel-stack/) — Verified: user-engagement (AUROC), post-votes (MAE).

[20] [RelBench rel-hm tasks](https://relbench.stanford.edu/datasets/rel-hm/) — Verified: user-churn (AUROC), item-sales (MAE).

[21] [RelBench rel-avito tasks](https://relbench.stanford.edu/datasets/rel-avito/) — Verified: user-visits (AUROC), ad-ctr (MAE).

[22] [Review of How Powerful are GNNs (Jaehyeong Jo)](https://harryjo97.github.io/paper%20review/How-Powerful-are-Graph-Neural-Networks/) — Confirmed Lemma 5 exact statement and mean/max non-injectivity.

[23] [DropEdge (Rong et al., ICLR 2020)](https://arxiv.org/abs/1907.10903) — Precedent for plug-in module framing pattern.

## Follow-up Questions

- What hidden dimension and number of GNN layers should be used for fair comparison across all baselines, and should these match RelBench's default training configuration or be independently tuned?
- Should the effective-rank gate extension (Appendix B) be included in the main experiments as an additional row, or kept strictly in the appendix to maintain focus on the simpler log-N gate?
- How should the paper handle the case where CAMA's improvements on regression tasks are not statistically significant -- should these tasks be dropped from the main table or retained for transparency?

---
*Generated by AI Inventor Pipeline*
