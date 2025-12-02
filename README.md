# EnergyOM

Ontology matching framework for the energy domain.

This repository contains code and experiments for matching heterogeneous energy schemas and ontologies to a common target model. It focuses on practical pipelines for real energy data where schemas, device models and KPIs differ across systems.

---

## Overview

EnergyOM provides

* utilities to read and normalize ontology and schema files  
* matching modules that combine lexical, structural and embedding based signals  
* experiment scripts for both general benchmarks and energy specific cases  
* evaluation scripts that compute standard matching metrics and export results  

The primary goal is to support research and reproducible experiments on ontology matching in the energy domain.

---

## Repository structure

```text
EnergyOM/
├── Data/                     # Ontologies, ground truth alignments, result files
├── Data_Reading/             # Readers for ontology and schema sources
├── Data_Processing/          # Cleaning, normalization, feature preparation
├── Data_Matching/            # Matching strategies and candidate generation
├── Model/                    # Model wrappers and helper classes
├── Evaluation/               # Evaluation and reporting utilities
├── Experiment/
│   └── Onto2Onto/           # Additional experiment configuration and scripts
├── energy_domain_experiment/ # Energy domain specific experiments
└── conference_experiment.py  # Example script for OAEI Conference benchmark
