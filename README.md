# Adaptive Federated Learning System with Layered Differential Privacy

**Inventor:** Ayush  
**Date of First Reduction to Practice:** May 21, 2026  
**Status:** Patent Pending (Provisional Filing in Progress)

---

## 🏆 Results

| Round | Accuracy | Loss |
|-------|----------|------|
| 1 | 97.14% | 0.1348 |
| 2 | 98.43% | 0.0456 |
| 3 | 98.84% | 0.0338 |
| 4 | 98.83% | 0.0339 |
| 5 | **99.07%** | **0.0278** |

**Privacy Guarantee:** (10.0, 1e-5)-DP  
**Hardware:** Standard laptop, no GPU  
**Training Time:** ~20 minutes

---

## 🔬 Three Novel Technical Components

### 1. Dynamic Compression Ratio Selector
Measures device compute capacity per round and dynamically selects
the optimal gradient compression ratio per client device.

### 2. Heterogeneity-Aware Reliability-Weighted Aggregation
Unlike FedAvg (weights by sample count only), this system computes
a **reliability score** per client from accuracy consistency and
improvement trend, then aggregates by both sample count AND reliability.

### 3. Layered Differential Privacy Budget Allocator
Classifies neural network layers into sensitivity tiers (high/medium/low)
and applies differentiated Gaussian noise per tier — more noise to
sensitive early layers, less to final classification layers.

---

## 📁 Project Structure
fedlearn_project/
├── model.py       # CNN architecture
├── client.py      # Federated client logic
├── server.py      # Aggregation server
├── privacy.py     # Layered DP module
├── utils.py       # Utilities
├── simulate.py    # Main simulation
├── analyze.py     # Patent evidence generator
└── results/       # Output charts and reports
---

## ⚖️ Patent Notice

This repository constitutes a public invention disclosure.
All novel methods described herein are the intellectual property
of the inventor. Provisional patent application filing in progress
at the Indian Patent Office under Patents Act 1970.