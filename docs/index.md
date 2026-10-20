# hiepy

**hiepy** is a Python library for working with Hierarchical Tucker (HT) tensors in PyTorch.

It provides data structures and operations for representing and manipulating high-dimensional tensors in HT format.

---

## Hierarchical Tucker Format

The HT format represents tensors using a hierarchical low-rank decomposition.

It reduces memory usage and computational cost by avoiding explicit dense tensor representations.

---

## Core Components

hiepy is built around three main concepts:

- **HTTensor**: hierarchical tucker tensor representation  
- **DimTree**: tree structure defining mode grouping  
- **Operations**: algebra, truncation, and tensor transformations  

---

## Functionality

The library supports:

- construction of HT tensors
- basic algebraic operations
- rank truncation
- tensor reshaping and indexing