# HDM-Traj

**A Task-Driven Dual-Mode Framework for Leakage-Free Language-Controllable Continuous Trajectory Generation**

HDM-Traj generates continuous GPS trajectories for two types of application: group-level mobility simulation and individual-level intelligent navigation. It combines global route synthesis with language-controllable generation and local editing in a unified framework.

## Overview

Mobility simulation requires diverse route sets reflecting group-level variation, whereas intelligent navigation requires a specific route satisfying individual-level preference. HDM-Traj addresses these different task requirements through two modes:

| Mode | Task | Approach |
| --- | --- | --- |
| **Mode A: Global route synthesis** | Generate route sets for group-level mobility simulation without language instructions. | Retrieve OD-similar training trajectories, align them to the requested endpoints, and select routes using a trained route selector. |
| **Mode B: Language-controllable generation and local editing** | Generate a continuous GPS route reflecting individual preference and revise it under fixed endpoints. | Parse natural-language preferences into five mobility controls, generate candidate routes with a diffusion-Transformer decoder, and select a route according to the instruction. |

A deterministic task router selects the appropriate mode according to the requested application task.

## Method

### Endpoint-derived reference condition

Both modes construct their reference condition solely from the origin–destination (OD) chord. Target intermediate-route information is excluded at inference, including information that could otherwise enter through target-derived reference routes.

### Mode A: Global route synthesis

Because the same OD pair can correspond to multiple plausible routes, endpoints alone do not determine the intermediate route. Mode A uses train-split route memory to supply route-structure evidence:

1. Retrieve trajectories with OD pairs similar to the requested endpoints.
2. Align the retrieved trajectories to those endpoints.
3. Score the aligned candidates using a trained route selector and training-derived priors.
4. Output the selected continuous GPS trajectory.

The reported Mode-A configuration uses candidate descriptors and training-derived priors without language instructions or semantic scoring.

### Mode B: Language-controllable generation and local editing

Mode B uses frozen Qwen representations with rule-assisted correction to translate natural-language preferences into five mobility controls:

| Control | Meaning |
| --- | --- |
| **Distance** | Desired route length. |
| **Direction** | Desired turning pattern. |
| **Curvature** | Desired degree of route smoothness or winding. |
| **Progress** | Desired trip pace. |
| **Reference adherence** | Desired closeness to the OD-chord reference. |

These controls condition a diffusion-Transformer decoder to generate continuous GPS candidate routes. Candidate selection identifies a route that matches the requested preference.

For local editing, Mode B updates the mobility controls according to the revised instruction and generates edited candidates while keeping the origin and destination fixed. Selection considers the requested change and its effects on other route attributes.

## Experiments

HDM-Traj is evaluated on three city-scale trajectory datasets: **Porto, Beijing, and Rome**. The experiments assess OD-conditioned continuous trajectory generation, with language controllability and fixed-OD local editing evaluated on Porto.

Across the three cities, Mode A achieves the lowest density error among the compared methods while exactly preserving endpoints. On Porto, the reported results are:

| Evaluation | Metric | Result |
| --- | --- | --- |
| Language-to-control planning | Five-field exact-match rate | **0.900** |
| Language-controllable generation | Instruction-following F1 | **0.540** |
| Fixed-OD local editing | Target-improvement success rate | **0.712** |

Additional experiments examine downstream utility, module ablations, hyperparameter sensitivity, spatial-scale robustness, and information leakage during inference.
