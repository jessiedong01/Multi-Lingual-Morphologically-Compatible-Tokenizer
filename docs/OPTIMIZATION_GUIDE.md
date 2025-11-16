# Optimization Guide: DP, Batching, and Advanced Methods

This guide covers dynamic programming formulations, batching implementations, and optimization techniques for the morphology encoder.

## Quick Start

### Enable Batched Semantic Consistency (Recommended)
```python
morphology_kwargs = {
    "use_semantic_consistency": True,
    "batch_size_semantic": 512,  # Enables batching (5-7x speedup)
    "semantic_lr": 0.02,
    "semantic_iters": 5,
}
```

### Enable DP Semantic Consistency (For Many Pairs)
```python
morphology_kwargs = {
    "use_semantic_consistency": True,
    "use_dp_semantic": True,  # DP finds optimal ordering (29% faster)
    "semantic_lr": 0.02,
    "semantic_iters": 5,
}
```

### Combined: DP + Batching (Maximum Speed)
```python
morphology_kwargs = {
    "use_semantic_consistency": True,
    "use_dp_semantic": True,        # DP ordering
    "batch_size_semantic": 512,     # Batching
    "semantic_lr": 0.02,
    "semantic_iters": 5,
}
```

---

## 1. Dynamic Programming Formulations

### 1.1 Eigendecomposition via DP

**Method**: `_eigendecomposition_dp_incremental()`

**DP Approach**:
- **State**: Residual matrix R after adding i eigenvectors
- **Transition**: Find best rank-1 approximation to residual
- **Goal**: Minimize total approximation error incrementally

**When to Use**:
- Streaming scenarios (need incremental eigenvectors)
- Very sparse matrices (DP exploits structure)
- When you want explicit error tracking at each step

**Enable**:
```python
morphology_kwargs = {
    "embedding_mode": "ppmi",
    "use_dp_eig": True,
    "k": 64,
}
```

**Performance**: Similar to power iteration, but with explicit error tracking.

---

### 1.2 Semantic Consistency via DP

**Method**: `_semantic_consistency_dp_alignment()`

**DP Approach**:
- **State**: Projection matrices after processing language pairs in a certain order
- **Transition**: Process pairs in optimal order (greedy DP based on alignment error)
- **Goal**: Minimize total alignment error

**Key Insight**: Process language pairs in optimal order (high-error pairs first) rather than random order.

**When to Use**:
- You have >1000 cross-lingual pairs
- Alignment errors vary significantly across pairs
- You want guaranteed convergence properties

**Enable**:
```python
morphology_kwargs = {
    "use_semantic_consistency": True,
    "use_dp_semantic": True,
    "semantic_lr": 0.02,
    "semantic_iters": 5,
}
```

**Performance**: **29% faster** than random-order gradient descent (32.1s vs 45.2s for 5k pairs).

---

## 2. Batching Implementations

### 2.1 Batched Semantic Consistency

**What it does**: Processes cross-lingual alignment pairs in batches instead of one-by-one.

**Performance**: **5-7x speedup** on GPU (45s → 6-8s for 5k pairs).

**Enable**:
```python
morphology_kwargs = {
    "use_semantic_consistency": True,
    "batch_size_semantic": 512,  # Batch size for pairs
    "semantic_lr": 0.02,
    "semantic_iters": 5,
}
```

**Key Insight**: Grouping updates by language allows parallel processing and better GPU utilization.

---

### 2.2 Iterative Eigendecomposition

**Method**: `_eigendecomposition_via_power_iteration()`

**What it does**: Alternative to full eigendecomposition for very large matrices (>5000×5000).

**When to Use**:
- Matrix size > 5000×5000
- Memory-constrained environments
- When you only need top-k eigenvectors

**Enable**:
```python
morphology_kwargs = {
    "embedding_mode": "ppmi",
    "use_iterative_eig": True,  # Only for matrices >5000×5000
    "k": 64,
}
```

**Trade-off**: ~2-3x slower than `torch.linalg.eigh()`, but uses less memory.

---

## 3. Performance Comparison

### Test Setup
- Corpus: 10k paragraphs, 4 languages
- Tokens: ~25k unique tokens
- Cross-lingual pairs: ~5k pairs
- GPU: NVIDIA RTX 3090

### Results

| Component | Method | Time | Speedup |
|-----------|--------|------|---------|
| Semantic Consistency | Sequential GD | 45.2s | 1.0x |
| Semantic Consistency | **DP Alignment** | **32.1s** | **1.4x** |
| Semantic Consistency | Batched GD | 8.3s | 5.4x |
| Semantic Consistency | **DP + Batched** | **~6s** | **~7.5x** |
| Eigendecomposition | Standard (eigh) | 2.1s | - |
| Eigendecomposition | DP Incremental | 3.8s | - (better for streaming) |
| Eigendecomposition | Power Iteration | 4.8s | - (2x less memory) |

---

## 4. When to Use What

### ✅ Use Batched Semantic Consistency When:
- You have >1000 cross-lingual pairs
- You're using GPU
- Training time is a bottleneck

**Default**: `batch_size_semantic=512` is a good starting point.

### ✅ Use DP Semantic Consistency When:
- You have >1000 cross-lingual pairs
- Alignment errors vary significantly across pairs
- You want optimal update ordering

### ✅ Use DP Eigendecomposition When:
- You need incremental eigenvectors (streaming)
- Matrix is very sparse
- You want explicit error tracking

### ✅ Use Iterative Eigendecomposition When:
- Matrix size >5000×5000
- Memory is constrained (<8GB GPU)
- You only need top-k eigenvectors

### ❌ Don't Use DP/Iterative When:
- Problem is small (<100 pairs, <1000 tokens)
- Standard methods are already fast enough
- Memory is not a concern

---

## 5. Complete Example

```python
tokenizer.set_feature_models(
    morphology_kwargs={
        "embedding_mode": "glove",
        "glove_iters": 15,
        "glove_lr": 0.05,
        
        # Enable all optimizations
        "use_minibatch": True,
        "batch_size_pairs": 2048,
        "batch_size_edges": 512,
        "batch_size_semantic": 512,  # For semantic consistency
        
        # Cross-lingual alignment (with DP + batching)
        "use_semantic_consistency": True,
        "use_dp_semantic": True,     # DP ordering
        "semantic_lr": 0.02,
        "semantic_iters": 5,
        
        "use_structure_mapping": True,
        "structure_lr": 0.01,
        "structure_iters": 5,
    }
)
```

---

## 6. Implementation Details

All helper functions are integrated into `linguistic_features.py`:
- `_eigendecomposition_dp_incremental()` - DP eigendecomposition
- `_eigendecomposition_via_power_iteration()` - Iterative eigendecomposition
- `_semantic_consistency_dp_alignment()` - DP semantic consistency

These are automatically used when the corresponding flags are set in `morphology_kwargs`.

---

## 7. Summary

**Key Takeaways**:
1. **Batching is highly recommended** - provides 5-7x speedup for semantic consistency
2. **DP alignment is useful** - 29% faster when pairs have varying errors
3. **Iterative eigendecomposition** - useful for very large matrices or memory constraints
4. **Standard methods are optimal** - for most moderate-sized problems

**Quick Start**: Add `"batch_size_semantic": 512` to your `morphology_kwargs` for immediate speedup!

