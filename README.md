# EnergyOM

Ontology matching framework for the energy domain.

EnergyOM is a research framework for ontology matching in the energy domain.
It provides end to end support for reading heterogeneous energy ontologies, generating candidate alignments, applying matching strategies, and evaluating results using standard metrics.

This project is intended for experiments, benchmarking, and reproducible evaluation of ontology matching pipelines on real world energy data models.

---

## Overview

EnergyOM provides

- utilities to read and normalize ontology and schema files
- matching modules that combine lexical, structural, embedding based and domain specific signals
- experiment scripts for both general benchmarks and energy specific cases
- evaluation scripts that compute standard matching metrics and export results

The primary goal is to support research and reproducible experiments on ontology matching in the energy domain.

---

## Repository structure

```text
EnergyOM/
├── Data/ # Ontologies, ground truth alignments, result files
├── Data_Reading/ # Readers for ontology and schema sources
├── Data_Processing/ # Cleaning, normalization, feature preparation
├── Data_Matching/ # Matching strategies and candidate generation
├── Model/ # Model wrappers and helper classes
├── Evaluation/ # Evaluation and reporting utilities
├── Experiment/
│ └── Onto2Onto/ # Additional experiment configuration and scripts
├── energy_domain_experiment/ # Energy domain specific experiments
└── conference_experiment.py # Example script for OAEI Conference benchmark
```

You can extend any of these modules to plug in new ontologies, models, or matching strategies.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zhiyupan/EnergyOM.git
cd EnergyOM
```

### 2. Set up a Python environment

Create and activate a virtual environment with your preferred tool, for example

```bash
python -m venv .venv
source .venv/bin/activate # Linux or macOS
# .venv\Scripts\activate # Windows PowerShell
```

Then install the Python packages used by the project, for example with

```bash
pip install -r requirements.txt
```

If a requirements file is not available yet, please inspect the import statements in the modules and install the listed libraries manually with pip.

---

## Quick start

### 1. Prepare data

Place your ontology files, mappings, and answer files under the `Data` directory or follow the same layout as the existing experiments.

Typical subfolders contain

- source ontologies
- target ontologies
- reference alignments for evaluation
- folders for storing experiment outputs


### 2. Run energy domain experiments

Energy specific experiments are organized in

- `energy_domain_experiment`
- `Experiment/Onto2Onto`

A typical workflow could look like

```bash
cd energy_domain_experiment
python run_energy_experiments.py # example name, adapt to the actual script
```

These experiments are intended to

- use energy domain ontologies stored under `Data`
- test different matching strategies and models in the energy setting
- export detailed results for further analysis with the tools in `Evaluation`

Adapt the script and configuration to your own energy ontologies or schemas.

---

## Supported matching dimensions

EnergyOM is designed to combine multiple signals

- lexical similarity between labels, comments, and identifiers
- hierarchy and structural context from the ontology graph
- domain and range information of properties
- embedding based representations for candidate generation
- hybrid scoring and binary decision filtering

---

## Evaluation

The `Evaluation` module contains helpers to compute and save

- precision
- recall
- F1 score
- ranking based measures such as mean reciprocal rank
- top k selection statistics

A typical usage pattern inside an experiment script

1. point the evaluator to a folder with produced alignments
2. point it to the corresponding answer files
3. call an evaluation function to write metrics into Excel or CSV files under `Data`  

Please see the functions and docstrings in `Evaluation` for concrete examples.

---



## License

```text
MIT License
```



---

## Contact and citation

For questions or feedback, please open an issue in the GitHub repository.
