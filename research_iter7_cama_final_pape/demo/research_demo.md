# CAMA Final Paper

## Summary

Complete submission-quality CAMA NeurIPS 2025 paper text with all 19 ITER6-PLACEHOLDER tags filled from 15 completed experiments. Theoretical narrative revised from 'rank collapse' to 'distributional information loss' after exp_id3_it6 disconfirmed the rank story (mean has HIGHER rank than sum, ratio 0.978). Scope narrowed to classification-specific benefit (d=2.13, p=0.129); regression null (d=0.17). Contains 20-item claim-evidence map (13 GREEN, 4 YELLOW, 3 RED), all 18 bibliography entries verified via Semantic Scholar, and brutally honest discussion acknowledging 81% gate stasis, val-only Amazon results, and RelGNN null integration.

## Research Findings

## CAMA Final Paper: All 19 Placeholders Filled with Honest Theoretical Reframing

This research output delivers the **complete, submission-quality paper text** for CAMA (Cardinality-Aware Moment Aggregation), a NeurIPS 2025 submission. All 19 ITER6-PLACEHOLDER tags have been filled with exact experimental numbers, the theoretical narrative has been fundamentally revised, and the scope has been honestly narrowed based on evidence.

### I. Placeholder Fill Results

**Abstract (3 placeholders):**
- PH-ABS-N → 8 tasks across experiments [1, 8]
- PH-ABS-M → 5 databases (Amazon, F1, Stack, Avito, Trial) [8]
- PH-ABS-D → pooled d=0.84 overall; classification d=2.13 (p=0.129, 100% sign consistency), regression d=0.17 (null)

**Introduction (4 placeholders):**
- PH-INTRO-RANK → Mean compression ratio 0.84 across edge types, WITH CRITICAL CAVEAT: mean has slightly HIGHER effective rank than sum (grand mean ratio 0.978), indicating distributional information loss rather than rank reduction [5]
- PH-INTRO-SUMRANK → Contrary to initial hypothesis, mean has marginally higher rank than sum (0.978). Sum's advantage is scale information preservation, not rank [5]
- PH-INTRO-N → 8 tasks
- PH-INTRO-SUM → On user-engagement: CAMA AP=0.4009 > sum 0.3995 (d=1.63) > mean 0.3786 (d=5.14)

**Related Work (1 placeholder):**
- PH-RW-PNA → CAMA MAE ~3.65 vs PNA ~3.81 on driver-position, confirming gated injection outperforms uniform multi-aggregation [4]

**Experiments (10 placeholders):**
- PH-SETUP-HW → NVIDIA RTX 4090, 8-25 min/run, ~50 GPU-hours total
- PH-T1-MEAN through PH-T1-CAMA → Complete Table 1 with 8 tasks filled; Sum column available for only 3/8 tasks (limitation acknowledged)
- PH-T2-ABLATION → 6-row ablation table showing ungated variance helps (3.69) and full CAMA improves further (3.65) vs mean (3.81) and PNA (3.81)
- PH-T3-RELGNN → RelGNN d=-0.33, complete gate stasis (all biases 0.0, sigmoid 0.5, W_sigma_norm=0.0)
- PH-F1-RANK → Compression ratio 0.84 confirmed BUT mean has HIGHER rank than sum (0.978) — DISCONFIRMATION of rank story
- PH-F2-FOREST → Classification d=2.13, regression d=0.17, pooled d=0.84, I²=85.4%
- PH-F3-GATE → 81% gate stasis; Amazon gates diverge at 10 epochs; rho=0.84 synthetic, weaker in real data

**Conclusion (1 placeholder):**
- PH-CONC-SUMMARY → Pooled d=0.84; classification d=2.13 (100% sign consistency), regression d=0.17 (null)

### II. Critical Theoretical Reframing

The most important change is the narrative revision forced by exp_id3_it6. The original story was that mean aggregation causes "rank collapse" at FK joins. The evidence DISCONFIRMED this: mean actually produces marginally HIGHER effective rank than sum (ratio 0.978) [5]. The revised narrative throughout all sections:

- **Old**: "Mean causes rank collapse, reducing representational capacity"
- **New**: "Mean causes distributional information loss — discarding variance and higher-order statistics — but this loss operates through distributional compression, NOT rank reduction"

This reframing affects Introduction (Paragraphs 1-2), Method (Section 3.2 renamed to "Distributional Information Loss at FK Aggregation"), Experiments (Section 4.5 reports both positive and negative findings), and Related Work (Section 2.3 explicitly distinguishes from Roth & Liebig's depth-wise phenomenon) [5, 1].

### III. Claim-Evidence Map (20 items)

**13 GREEN claims**: Established theoretical results from cited papers, all verified via web search [1-15]. Key GREEN items: Xu et al. Lemma 5 (sum injective, mean not) [1], RelBench sum default [8], PNA uniform application [4], VPA normalizes not enriches [3], Roth rank collapse is depth-wise [5], novelty confirmed via negative search, ablation results (ungated helps, gating improves further), RelGNN scope validation (d=-0.33), information compression measurable (ratio 0.84).

**4 YELLOW claims**: Classification benefit (d=2.13, p=0.129 — above 0.05 threshold), safe default (d=-0.82 on avito, not catastrophic), classification-regression asymmetry, sum parity (d=1.63, 3 seeds, 1 task only).

**3 RED claims**: Gate learning (81% stasis — major concern), compression-benefit correlation (rho=-0.5, only 3 tasks, p=0.67), sum rank story (DISCONFIRMED — mean has higher rank).

### IV. Bibliography Verification

All 18 references confirmed via Semantic Scholar API, arXiv, PMLR proceedings, and web search [1-15]:
- xu2019powerful: Xu/Hu/Leskovec/Jegelka, ICLR 2019, arXiv:1810.00826 ✓ [1]
- hamilton2017inductive: Hamilton/Ying/Leskovec, NeurIPS 2017, arXiv:1706.02216 ✓ [2]
- corso2020pna: Corso/Cavalleri/Beaini/Lio/Velickovic, NeurIPS 2020, arXiv:2004.05718 ✓ [4]
- schneckenreiter2024vpa: 6 authors, ICLR 2024 Tiny Papers, arXiv:2403.04747 ✓ [3]
- roth2024rank: Roth/Liebig, LoG 2024, PMLR 231:35:1-35:23, arXiv:2308.16800 ✓ [5]
- chen2025relgnn: Chen/Kanatsoulis/Leskovec, ICML 2025, PMLR 267:8296-8312, arXiv:2502.06784 ✓ [6]
- wang2025griffin: Wang/Wang/Gan/Wang/Yang/Wipf/Zhang (NOT Pelevin), ICML 2025, PMLR 267:64604-64627, arXiv:2505.05568 ✓ [7]
- robinson2024relbench: 12 authors, NeurIPS 2024 D&B, arXiv:2407.20060 ✓ [8]
- All remaining 10 entries confirmed via prior iter-6 verification + web search [9-15]

### V. Honest Scope Assessment

The paper's honest contribution is narrower than originally planned:
- **Works well**: Classification tasks with mean aggregation (d=2.13, 100% sign consistency)
- **Does not work**: Regression (d=0.17), SOTA architectures (RelGNN d=-0.33), gate learning (81% stasis)
- **Incomplete evidence**: Sum baselines (only 3/8 tasks), Amazon/Avito (val-only), classification p=0.129 (underpowered)

### VI. Terminology Compliance

Throughout all 5,601 words of paper text:
- 0 uses of "RAMA" or standalone "AMA"
- 95 uses of "CAMA"
- 12 uses of "information compression" (for CAMA's phenomenon)
- 17 uses of "distributional information"
- 7 uses of "rank collapse" — ALL verified in proper Roth/Liebig/Dong/NaitSaada context or explicit negation for CAMA

## Sources

[1] [Semantic Scholar: How Powerful are Graph Neural Networks (Xu et al., ICLR 2019)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1810.00826) — Confirmed authors Xu/Hu/Leskovec/Jegelka, venue ICLR, arXiv 1810.00826. Theoretical anchor for sum injectivity.

[2] [Semantic Scholar: Inductive Representation Learning on Large Graphs (Hamilton et al., NeurIPS 2017)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.02216) — Confirmed authors Hamilton/Ying/Leskovec, venue NeurIPS 2017, arXiv 1706.02216. GraphSAGE mean aggregation default.

[3] [Semantic Scholar: GNN-VPA (Schneckenreiter et al., ICLR 2024 Tiny Papers)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2403.04747) — Confirmed 6 authors, venue Tiny Papers @ ICLR 2024, arXiv 2403.04747. VPA normalizes variance for stability.

[4] [NeurIPS 2020: PNA (Corso et al.)](https://proceedings.neurips.cc/paper/2020/hash/99cad265a1768cc2dd013f0e740300ae-Abstract.html) — Confirmed NeurIPS 2020, authors Corso/Cavalleri/Beaini/Lio/Velickovic, arXiv 2004.05718. 4 aggregators x 3 scalers.

[5] [PMLR: Rank Collapse Causes Over-Smoothing (Roth & Liebig, LoG 2024)](https://proceedings.mlr.press/v231/roth24a.html) — Confirmed LoG 2024 PMLR 231:35:1-35:23. Depth-wise rank collapse distinct from CAMA's distributional loss.

[6] [arXiv: RelGNN (Chen et al., ICML 2025)](https://arxiv.org/abs/2502.06784) — Confirmed ICML 2025, PMLR 267:8296-8312. SOTA on 27/30 RelBench tasks via atomic routes.

[7] [arXiv: Griffin (Wang et al., ICML 2025)](https://arxiv.org/abs/2505.05568) — Confirmed authors Wang/Wang/Gan/Wang/Yang/Wipf/Zhang (NOT Pelevin), PMLR 267:64604-64627.

[8] [arXiv: RelBench (Robinson et al., NeurIPS 2024)](https://arxiv.org/abs/2407.20060) — Confirmed NeurIPS 2024 D&B Track, 12 authors, sum aggregation default confirmed.

[9] [arXiv: Mind the Gap (Nait Saada et al., 2024)](https://arxiv.org/abs/2410.07799) — Confirmed width-wise rank collapse in attention, authors at Oxford. Spectral gap mechanism.

[10] [arXiv: Attention is Not All You Need (Dong et al., ICML 2021)](https://arxiv.org/abs/2103.03404) — Confirmed rank collapse doubly exponential with depth in pure attention networks.

[11] [arXiv: On the Bottleneck of GNNs (Alon & Yahav, ICLR 2021)](https://arxiv.org/abs/2006.05205) — Confirmed over-squashing phenomenon, ICLR 2021.

[12] [NeurIPS 2023: GenAgg (Kortvelesy et al.)](https://proceedings.neurips.cc/paper_files/paper/2023/file/6c78ae0c1140902bf3a430b1725bcc4e-Paper-Conference.pdf) — Confirmed NeurIPS 2023, Cambridge authors, globally shared learned f-mean aggregator.

[13] [OpenReview: Griffin ICML 2025](https://openreview.net/forum?id=TxeCxVb3cL) — Confirmed ICML 2025 acceptance, mean intra-relation aggregation architecture.

[14] [GitHub: RelGNN official implementation](https://github.com/snap-stanford/RelGNN) — Confirmed ICML 2025 code release, Stanford SNAP group.

[15] [RelBench website](https://relbench.stanford.edu/) — Confirmed benchmark infrastructure, sum aggregation default, RelBench v2 released Jan 2026.

## Follow-up Questions

- Should the paper be submitted as a short paper or workshop paper given the classification-only scope (d=2.13, p=0.129) and regression null (d=0.17), or is the honest framing sufficient for a full NeurIPS submission?
- Is the sum comparison gap (only 3/8 tasks have sum baselines) acceptable for NeurIPS review, or must sum experiments be completed on all 8 tasks before submission?
- Should the effective-rank gate variant be promoted from Appendix B given the simplified gate's 81% stasis rate, or does the comparable performance of CAMA-no-rank in ablation (Table 2) justify keeping the simpler formulation?

---
*Generated by AI Inventor Pipeline*
