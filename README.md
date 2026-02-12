# UNet Path Finder

**Shortest road path finding algorithm on residual attention U-Net models with cluster union-find connectivity**

A deep learning system for road segmentation and pathfinding in aerial imagery. The project combines attention-augmented U-Net architectures with graph-based pathfinding algorithms, featuring a novel connectivity recovery mechanism using union-find clustering to handle fragmented segmentations.

---

## Overview

This project addresses road network extraction and optimal path computation from aerial imagery through a two-stage algorithmic pipeline:

1. **Semantic Segmentation**: A modified U-Net with ResNet50 encoder and attention gates produces binary road masks from RGB images
2. **Graph-Based Pathfinding**: Morphological skeletonization, connectivity recovery via union-find clustering, and BFS-based shortest path computation

The key contribution lies in the connectivity recovery algorithm, which automatically reconnects broken road segments using spatial proximity clustering and union-find merging, addressing a common failure mode in segmentation-based navigation systems.

## Demo

![Demo](docs/assets/unet_demo.gif)
![Demo](docs/assets/ss2.png)

---

## Network Architecture

### ResNet50-UNet with Attention Gates

The segmentation model extends the standard U-Net architecture with two key algorithmic modifications: a pretrained ResNet50 encoder for hierarchical feature extraction and attention gates for adaptive feature selection during decoding.

**Encoder: Hierarchical Feature Extraction**

The encoder employs ResNet50's bottleneck architecture to extract multi-scale features across five progressive stages. Each stage doubles the receptive field while increasing semantic abstraction:

- Initial convolution block captures low-level edges and textures (64 channels, H/2)
- ResNet Layer1-4 process features through 3, 4, 6, and 3 bottleneck blocks respectively
- Progressive downsampling to H/4, H/8, H/16, H/32 with channel expansion to 256, 512, 1024, 2048
- Bottleneck layer compresses features to 1024 channels at lowest resolution

The bottleneck blocks use residual connections (identity shortcuts) to enable gradient flow through deep layers, addressing the vanishing gradient problem that limits plain convolutional networks.

**Attention Gates: Adaptive Feature Selection**

Standard U-Net concatenates all encoder features to the decoder regardless of relevance. This implementation introduces a gating mechanism that dynamically determines which features should influence reconstruction at each scale.

The attention algorithm operates at each decoder stage:

1. **Feature Transformation**: Both the upsampled decoder features (gating signal) and encoder skip connection are projected to a common intermediate dimensionality through 1×1 convolutions
2. **Compatibility Scoring**: The transformed features are added element-wise and passed through ReLU activation, computing compatibility between decoder context and encoder features
3. **Attention Weight Generation**: A 1×1 convolution followed by sigmoid produces spatial attention coefficients α ∈ [0,1] for each spatial location
4. **Feature Reweighting**: The original skip connection is element-wise multiplied by attention coefficients, suppressing irrelevant regions (α→0) while preserving relevant features (α→1)

This mechanism enables the network to focus on road-specific patterns while ignoring background clutter, particularly improving performance at object boundaries where context disambiguation is critical.

**Decoder: Progressive Spatial Reconstruction**

The decoder reconstructs spatial resolution through symmetric expansion. At each of five stages:

1. Transposed convolution upsamples features by 2× spatially
2. Attention gate processes the corresponding encoder skip connection based on current decoder context
3. Attended skip features concatenate with upsampled features
4. Double convolution block (two 3×3 conv-BN-ReLU sequences) refines the merged features

Channel dimensions progressively decrease (1024→512→256→64→32) as spatial resolution expands, culminating in a 1×1 convolution to single-channel segmentation logits.

**Alternative Architectures**

The repository includes two architectural variants for comparison:

- **ResNet50-UNet (Standard)**: Identical encoder-decoder structure without attention gates, providing an ablation baseline to isolate attention mechanism contribution
- **VGG16-UNet**: Replaces ResNet50 encoder with VGG16 batch-normalized layers, offering a simpler sequential convolution architecture without residual connections

---

## Pathfinding Algorithms

The pathfinding system transforms pixel-wise segmentation into navigable graph structures through a series of algorithmic operations designed to handle real-world imperfections in the segmentation output.

### Pipeline Architecture

The pathfinding pipeline consists of four sequential stages:

1. **Binarization**: Threshold continuous probability maps to binary classification
2. **Morphological Skeletonization**: Reduce road regions to 1-pixel centerlines via Zhang-Suen thinning
3. **Connectivity Recovery**: Reconnect fragmented components via union-find clustering
4. **Shortest Path Computation**: BFS traversal on the connected skeleton graph

Each stage addresses specific challenges in converting dense segmentation to discrete navigation graphs.

### Morphological Skeletonization

The Zhang-Suen thinning algorithm iteratively removes boundary pixels while preserving topological connectivity. The algorithm alternates between two sub-iterations, each examining pixels in the skeleton and removing those that satisfy specific structural conditions:

- The pixel must be a boundary pixel (has at least one non-road neighbor)
- Removing the pixel must not disconnect the skeleton (topological constraint)
- The pixel must not be an endpoint of a line segment

This produces a medial axis transform that represents roads as 1-pixel-wide curves following centerlines, converting 2D regions into 1D graph structures suitable for pathfinding. The resulting skeleton preserves junction topology and connectivity while dramatically reducing computational requirements for subsequent graph algorithms.

---

### Union-Find Connectivity Recovery Algorithm

**Problem Formulation**

Road segmentation models frequently produce fragmented outputs due to occlusions (trees, shadows, vehicles), difficult imaging conditions, or model uncertainty. These fragmentations manifest as disconnected components in the skeleton, preventing pathfinding across gaps even when the underlying road structure is continuous.

Traditional morphological operations (dilation, closing) can bridge gaps but suffer from two limitations:
1. They distort road topology and width, creating artificial connections
2. They lack selectivity, potentially connecting unrelated road segments

The connectivity recovery algorithm addresses these limitations through a graph-theoretic approach: identify disconnected components, measure inter-component distances, and selectively merge nearby clusters while maintaining topological validity.

**Algorithm Design**

The `connect_broken_roads` function implements a three-phase clustering algorithm:

**Phase 1: Connected Component Decomposition**

The skeleton is decomposed into maximal connected components using a standard two-pass labeling algorithm. Each component represents a set of transitively connected road pixels forming a subgraph of the overall road network.

Let C = {C₁, C₂, ..., Cₖ} denote the set of k disconnected components, where each Cᵢ ⊆ S is a set of pixel coordinates belonging to component i.

**Phase 2: Proximity-Based Connection Candidate Generation**

For each unordered pair of components (Cᵢ, Cⱼ), compute the minimum Euclidean distance:

d(Cᵢ, Cⱼ) = min{‖p - q‖₂ : p ∈ Cᵢ, q ∈ Cⱼ}

This requires comparing all pixel pairs between components. The implementation vectorizes this computation: extract coordinate arrays for each component and compute a distance matrix using broadcasting, then identify the minimum distance and corresponding pixel pair.

If d(Cᵢ, Cⱼ) ≤ δ (threshold distance), create a connection candidate (p*, q*, d) where p* ∈ Cᵢ and q* ∈ Cⱼ are the closest pixels.

The threshold δ controls connection selectivity: small values (20-40 pixels) bridge only obvious gaps, while larger values (80-150 pixels) aggressively connect distant components at risk of spurious connections.

**Phase 3: Union-Find Greedy Merging**

Connection candidates are sorted by distance in ascending order, prioritizing short-range connections over speculative long-range bridges. The union-find data structure manages component connectivity while preventing redundant connections.

Union-find maintains a forest where each tree represents a connected component. Two operations manage the structure:

**Find Operation with Path Compression**:
```
find(x):
    if parent[x] ≠ x:
        parent[x] = find(parent[x])  // Recursively find root and compress path
    return parent[x]
```

Path compression flattens tree structures, ensuring subsequent find operations approach O(1) amortized complexity.

**Union Operation**:
```
union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x ≠ root_y:
        parent[root_x] = root_y  // Merge trees
        return True
    return False  // Already connected
```

The merging algorithm processes candidates in distance order:

```
for each (p, q, d) in sorted_candidates:
    i = component_of(p)
    j = component_of(q)
    
    if union(i, j):  // If components not already connected
        draw_line(skeleton, p, q)  // Physically connect the gap
```

The union-find structure provides two critical properties:

1. **Transitivity Detection**: If A connects to B and B connects to C, attempting to connect A and C returns false, avoiding redundant edges
2. **Cycle Prevention**: Components already in the same union-find tree are not reconnected, maintaining graph sparsity

**Iterative Multi-Hop Connectivity**

A single pass of the algorithm connects components whose closest points are within threshold distance. However, some components may be unreachable in one iteration but become reachable after intermediate connections are established.

The iterative refinement wrapper applies the base algorithm repeatedly:

```
for iteration in 1..max_iterations:
    skeleton_prev = skeleton.copy()
    skeleton = connect_broken_roads(skeleton, δ)
    
    if skeleton == skeleton_prev:  // No new connections
        break
```

This enables multi-hop connectivity: if components A and B connect in iteration 1, and B and C connect in iteration 2, then A and C become transitively connected even if their direct distance exceeds δ.

Convergence typically occurs in 2-3 iterations for realistic road networks, as most fragmentation gaps are local rather than requiring long connection chains.

**Algorithmic Complexity**

- Component labeling: O(HW) using two-pass algorithm
- Pairwise distance computation: O(k²N²) where k is number of components, N is average component size
- Sorting candidates: O(C log C) where C is number of candidate connections
- Union-find operations: O(α(k)) amortized per operation, where α is the inverse Ackermann function (effectively constant)

The dominant cost is pairwise distance computation, quadratic in the number of components and component sizes.

---

### Shortest Path Computation via BFS

Once connectivity is established, the system computes optimal routes through the road network. The skeleton is interpreted as an implicit graph where road pixels are vertices and 8-connected adjacencies are edges.

**Algorithm Selection Rationale**

Breadth-First Search (BFS) is chosen over alternative shortest path algorithms:

- **vs. Dijkstra's Algorithm**: BFS is optimal for unweighted graphs (or uniformly weighted graphs), which applies here since all skeleton edges represent unit spatial steps. Dijkstra introduces unnecessary priority queue overhead without benefit.
- **vs. A* Search**: A* requires an admissible heuristic function. Euclidean distance to the goal is inadmissible on constrained road networks (straight-line distance underestimates true road distance due to mandatory routing). Without a valid heuristic, A* degrades to Dijkstra's algorithm.
- **vs. Bidirectional Search**: Offers potential speedup but increases implementation complexity. The road networks are sparse enough that standard BFS suffices.

**BFS Implementation**

The algorithm maintains a queue of vertices to explore and a visited map tracking the parent of each discovered vertex:

```
BFS(skeleton, start, end):
    queue ← [start]
    visited ← {start: null}
    
    while queue not empty:
        current ← queue.dequeue()
        
        if current == end:
            return reconstruct_path(visited, end)
        
        for each neighbor in 8_connected_neighbors(current):
            if skeleton[neighbor] == 255 and neighbor not in visited:
                visited[neighbor] ��� current
                queue.enqueue(neighbor)
    
    return null  // No path exists

reconstruct_path(visited, end):
    path ← []
    node ← end
    while node ≠ null:
        path.prepend(node)
        node ← visited[node]
    return path
```

**8-Connected Adjacency**

The neighborhood definition includes all eight adjacent pixels (cardinal and diagonal directions). This represents the graph as a grid where diagonal movements are permitted, allowing more natural path shapes compared to 4-connected grids.

Alternatively, diagonal edges could be weighted √2 vs. 1 for cardinal edges to reflect true Euclidean distance, converting to a weighted graph requiring Dijkstra's algorithm. The current implementation treats all edges uniformly for simplicity.

**Optimality Guarantee**

BFS explores vertices in order of increasing distance from the source. When the destination vertex is first encountered (dequeued from the queue), its distance equals the minimum possible path length in the graph. This follows from BFS's level-order traversal property: all vertices at distance d are discovered before any vertex at distance d+1.

**Practical Considerations**

For typical road networks, the skeleton graph is sparse (average degree ~3 at pixels, higher at junctions). BFS explores only the connected component containing the start vertex, typically a small fraction of the full skeleton for localized queries. Worst-case complexity is O(V + E) where V is the number of road pixels, but practical performance is much better due to early termination upon reaching the goal.

---

## Training Algorithm

### Multi-Objective Loss Function

The training algorithm optimizes a weighted combination of three loss functions, each addressing different aspects of segmentation quality:

**Composite Loss**: L_total = 0.3·L_Jaccard + 0.2·L_Dice + 0.5·L_Focal

**Jaccard Loss (IoU Loss)**:
Directly optimizes intersection-over-union metric by minimizing:
L_Jaccard = 1 - (|Y ∩ Ŷ|)/(|Y ∪ Ŷ|)

This emphasizes regional overlap and is particularly sensitive to large-scale topological correctness.

**Dice Loss (F1 Loss)**:
Optimizes the F1 score via:
L_Dice = 1 - (2|Y ∩ Ŷ|)/(|Y| + |Ŷ|)

Compared to Jaccard, Dice is more sensitive to small objects and thin structures (narrow roads), as the denominator doesn't double-penalize the intersection term.

**Focal Loss**:
Addresses extreme class imbalance (background pixels vastly outnumber road pixels) through:
L_Focal = -α(1 - p_t)^γ · log(p_t)

The modulating factor (1-p_t)^γ down-weights loss contribution from easy examples (high confidence correct predictions) and focuses learning on hard examples (misclassifications or low-confidence predictions). This prevents the overwhelming number of easy background pixels from dominating gradient updates.

The loss weighting (30% Jaccard, 20% Dice, 50% Focal) balances regional accuracy, boundary precision, and hard example learning based on empirical validation performance.

### Optimization Strategy

**Gradient Descent with Adaptive Learning**:
AdamW optimizer (Adam with decoupled weight decay) adapts learning rates per parameter based on first and second moment estimates of gradients, enabling stable convergence despite varying gradient magnitudes across network depth.

**Learning Rate Scheduling**:
ReduceLROnPlateau monitors validation loss and reduces learning rate by 50% when progress stagnates for 5 consecutive epochs. This allows fine-grained optimization in later training stages while maintaining aggressive initial learning.

**Gradient Clipping**:
Gradients are clipped to maximum L2 norm of 1.0 before parameter updates. This prevents instability from occasional large gradients (particularly in attention gates) that can destabilize training.

**Early Stopping**:
Training terminates if validation loss fails to improve by at least 0.0001 for 15 consecutive epochs, preventing overfitting to the training set.

**Mixed Precision Training**:
Automatic Mixed Precision (AMP) performs forward passes and loss computation in float16 precision while maintaining float32 precision for critical operations (gradient accumulation, parameter updates). This accelerates training by ~2× on modern GPUs while maintaining numerical stability through gradient scaling.

### Data Augmentation Strategy

The augmentation pipeline increases training data diversity through geometric and photometric transformations:

**Geometric Augmentations**:
- Random rotation (±180°) with reflection padding
- Horizontal and vertical flips
- Perspective transforms (scale 5-10%) simulating viewpoint changes
- Elastic deformation (α=120, σ=6) simulating terrain variations
- Grid distortion simulating lens aberrations

**Photometric Augmentations**:
- Random brightness/contrast adjustment
- Additive Gaussian noise
- Motion blur (kernel size 3)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

These augmentations improve model robustness to imaging variations (lighting conditions, camera angles, atmospheric effects) commonly encountered in aerial imagery.

---

## Key Algorithmic Contributions

### Attention-Guided Feature Selection

The attention mechanism implements a learned gating function that adapts feature propagation based on decoder context. This differs from fixed skip connections by allowing the network to suppress background features dynamically, particularly valuable when roads occupy small portions of the image.

### Union-Find Connectivity Recovery

The connectivity algorithm provides several advantages over morphological alternatives:

1. **Selectivity**: Only connects components within distance threshold, avoiding spurious connections
2. **Topology Preservation**: Draws minimal connecting segments rather than dilating entire regions
3. **Cycle Prevention**: Union-find structure prevents redundant connections
4. **Iterative Refinement**: Multi-hop connectivity emerges through repeated application

This approach specifically addresses the fragmented segmentation problem while maintaining graph sparsity for efficient pathfinding.

### Multi-Objective Optimization

The composite loss function balances competing objectives:
- Jaccard emphasizes regional overlap
- Dice prioritizes thin structures
- Focal handles class imbalance

This multi-objective approach outperforms single-loss training by simultaneously optimizing different aspects of segmentation quality.

---

## Results

**Segmentation Performance** (Massachusetts Roads Dataset):
- Dice Coefficient: 0.85-0.90
- IoU: 0.75-0.82

**Connectivity Recovery**:
- Successfully bridges gaps up to 100 pixels
- Typical convergence in 2-3 iterations
- Enables pathfinding across previously disconnected regions

**Pathfinding Performance**:
- BFS computation: <50ms for typical networks
- Guarantees shortest path on connected graphs

---

## References

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015
2. Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," arXiv:1804.03999
3. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016
4. Zhang & Suen, "A Fast Parallel Algorithm for Thinning Digital Patterns," CACM 1984
5. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017

---

## Installation and Usage

```bash
git clone https://github.com/Miker2808/unet_path_finder.git
cd unet_path_finder
pip install -r requirements.txt

# Run pathfinding demo
cd pathfinder
python pathfinder.py

# Train model
python train.py
```

## License

MIT License

## Contact

**Repository**: [github.com/Miker2808/unet_path_finder](https://github.com/Miker2808/unet_path_finder)
