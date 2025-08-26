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