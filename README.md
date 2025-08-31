# Leveraging Dynamic Few-Shot Prompting and Ensemble Method for Task-Oriented Dialogue With Subjective Knowledge

Published in IPM Findings 2025

# DESIGN Framework

This repository contains the main codebase for the **DESIGN** framework. Below is a description of the key directories and their contents:

---

## 1. Overview
This is the core implementation of the **DESIGN** framework, which provides solutions for various tasks in the domain of knowledge-aware dialogue systems.

---

## 2. Directory Structure

### `baseline`
- Contains the official code provided by the competition organizers.
- Includes the **classification-based method** implementation for the **DESIGN KS** task.

### `DynamicGeneration`
- Houses the main code for the **DESIGN RG** task.
- Specifically tailored for the **SK-TOD dataset**, this module focuses on dynamic response generation.

### `DynamicKS`
- Implements the **generation-based method** for the **DESIGN KS** task.
- Provides an alternative approach to handling knowledge selection dynamically.

### `SELECT`
- Serves as an extension to the `DynamicGeneration` module.
- Contains additional code for the **DESIGN** framework specifically adapted to the **ReDial** dataset.

---

For further details about the framework or specific tasks, please refer to the respective directories or contact the maintainers.

### Citation

If you find DESIGN useful for your research and applications, please cite using this BibTeX:
```
@article{RAO2026104317,
title = {Leveraging dynamic few-shot prompting and ensemble method for task-oriented dialogue with subjective knowledge},
journal = {Information Processing & Management},
volume = {63},
number = {2, Part A},
pages = {104317},
year = {2026},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2025.104317},
url = {https://www.sciencedirect.com/science/article/pii/S0306457325002584},
author = {Dongning Rao and Jietao Zhuang and Zhihua Jiang},
keywords = {Natural language processing, Task-oriented dialogue, Subjective knowledge, Aspect-based sentiment analysis, Dynamic few-shot prompt},
abstract = {Subjective knowledge is key to meeting customer needs. Thus, the Subjective Knowledge-grounded Task-oriented Dialogue (SK-TOD) task tries to accommodate subjective user requests like “Does the restaurant have a good atmosphere?” by choosing relevant subjective knowledge snippets and generating appropriate responses. However, unlike existing methods like retrieval-augmented generation using external objective knowledge, selecting subjective knowledge and summarizing opinions from reviews in a specified scope pose new challenges. Therefore, this paper proposes the DESIGN (Dynamic fEw-Shot promptInG and eNsemble) method for SK-TOD. Specifically, DESIGN first adopts Aspect-Based Sentiment Analysis (ABSA) to enhance subjective knowledge snippets and then builds an ensemble composed of diverse base models for knowledge selection (KS). Here, the base models include both classification models and generative models. At last, for response generation (RG), DESIGN employs generative models conditioned on dialogue context and ABSA-enhanced knowledge. Particularly, we devise the sample selection via the similarity-alignment algorithm to choose similar samples dynamically for the few-shot prompting of KS and RG. We experiment on the 11th Dialog System Technology Challenge (DSTC11) SK-TOD benchmark and an extended dataset, ReDial, with 6147 instances. For KS, we beat the winner of DSTC11 and boosted the F1 for 7% regarding the baseline and achieved 86.16%. For RG, DESIGN outperforms baselines and the DSTC11 winner across eight metrics.E.g., DESIGN improves entailment performance by 5% over the DSTC11 winner and 10% over the baseline.11Our source code can be visited via GitHub: https://github.com/fip-lab/DESIGN.}
}
```