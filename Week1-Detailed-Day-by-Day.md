# Week 1 — Linear Algebra, Practically (Day-by-Day)

**Goal by end of week:** You can look at a matrix operation and know what it's *doing* geometrically — not just recite the formula. This underlies everything from camera calibration to CNN filters to PCA.

---

## Day 1 (1.5-2 hrs) — Vectors & Setup
**Watch:** 3Blue1Brown "Essence of Linear Algebra" — Ep 1 (Vectors) and Ep 2 (Linear combinations, span, basis vectors) — ~25 min total

**Set up your environment:**
```bash
mkdir industrial-ai-journey && cd industrial-ai-journey
python -m venv ml-env
source ml-env/bin/activate    # or ml-env\Scripts\activate on Windows
pip install numpy matplotlib jupyter
git init
mkdir week1_linalg
jupyter notebook
```

**Do in notebook (`week1_linalg/week1_linalg.ipynb`):**
```python
import numpy as np
import matplotlib.pyplot as plt

# Represent two vectors
v1 = np.array([2, 3])
v2 = np.array([-1, 2])

# Plot them from origin
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
plt.xlim(-5, 5); plt.ylim(-5, 5)
plt.grid(); plt.legend()
plt.title("Two vectors")
plt.show()

# Vector addition — plot v1+v2 too, see it geometrically
```
**Task:** Add a third vector `v3 = v1 + v2`, plot it, confirm visually it's the diagonal of the parallelogram formed by v1 and v2.

**Checkbox:** ⬜ Notebook runs, plot shows 3 vectors, you can explain in one sentence what vector addition looks like geometrically.

---

## Day 2 (1.5 hrs) — Dot Products
**Watch:** 3Blue1Brown Ep 9 (Dot products and duality) — ~14 min

**Do:**
```python
def dot_product(a, b):
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

v1 = np.array([2, 3])
v2 = np.array([-1, 2])

manual = dot_product(v1, v2)
numpy_result = np.dot(v1, v2)
assert manual == numpy_result
print(f"Dot product: {manual}")

# Cosine similarity — used constantly in ML (feature similarity, embeddings)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Cosine similarity: {cosine_similarity(v1, v2):.3f}")
```
**Task:** Compute cosine similarity for 3 pairs of vectors — two pointing the same direction, two perpendicular, two opposite. Confirm the values make sense (should be ~1, ~0, ~-1).

**Checkbox:** ⬜ You can explain why cosine similarity of perpendicular vectors is 0, and why this matters for comparing feature vectors in ML.

---

## Day 3 (2 hrs) — Matrices as Transformations
**Watch:** 3Blue1Brown Ep 3 (Linear transformations) and Ep 4 (Matrix multiplication as composition) — ~25 min

**Do:**
```python
# A rotation matrix — you've likely used this exact concept in camera calibration
theta = np.radians(45)
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

point = np.array([1, 0])
rotated_point = rotation_matrix @ point
print(f"Original: {point}, Rotated 45°: {rotated_point}")

# Plot before/after
fig, ax = plt.subplots()
ax.quiver(0, 0, point[0], point[1], angles='xy', scale_units='xy', scale=1, color='gray', label='original')
ax.quiver(0, 0, rotated_point[0], rotated_point[1], angles='xy', scale_units='xy', scale=1, color='red', label='rotated')
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
ax.legend(); ax.grid()
plt.show()
```
**Task:** Rotate a set of 5-6 points (e.g., forming a small square) by 90°, 180°, and 45°, plot each result. This is exactly the math behind image rotation and camera extrinsics.

**Checkbox:** ⬜ You can build a rotation matrix for any angle from memory and explain why matrix multiplication order matters (rotation ∘ scale ≠ scale ∘ rotation).

---

## Day 4 (2 hrs) — Matrix Multiplication From Scratch
**Watch:** 3Blue1Brown Ep 4 recap if needed, or Ep 5 (3D transformations) for intuition

**Do — implement matrix multiplication without `@` or `np.dot`:**
```python
def matmul_scratch(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    assert cols_A == rows_B, "Incompatible shapes"
    result = np.zeros((rows_A, cols_B))
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

manual = matmul_scratch(A, B)
numpy_result = A @ B
assert np.allclose(manual, numpy_result)
print("Matches NumPy!")
```
**Task:** Time your scratch implementation vs NumPy's `@` on a 100x100 random matrix multiplication. Note the speed difference — this is your first real taste of *why* vectorized operations matter for ML performance.

**Checkbox:** ⬜ You understand matrix multiplication well enough to implement it blind, and you've seen firsthand why NumPy/optimized libraries exist.

---

## Day 5 (1.5-2 hrs) — Eigenvalues/Eigenvectors + Tie-In to Your Background
**Watch:** 3Blue1Brown Ep 14 (Eigenvectors and eigenvalues) — ~17 min

**Do:**
```python
A = np.array([[2, 0], [0, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Try a non-diagonal matrix — see that eigenvectors are the directions
# that DON'T rotate under the transformation, only scale
A2 = np.array([[3, 1], [0, 2]])
eigenvalues2, eigenvectors2 = np.linalg.eig(A2)
print(f"Eigenvalues: {eigenvalues2}")
```
**Task — write a short paragraph (in the notebook, as markdown) connecting this to what you already know:**
- PCA (used in Week 6) finds eigenvectors of the covariance matrix — the "directions of maximum variance" in your sensor data
- If you've done camera calibration, the intrinsic/extrinsic matrices and homography estimation all lean on these same linear algebra building blocks
- CNNs later in the course learn filters that are conceptually similar to eigenvector-style "important directions" in image data, just learned rather than computed analytically

**Checkbox:** ⬜ You can explain in plain language (no formulas) what an eigenvector represents, and you've written the connecting paragraph.

---

## Day 6-7 (2-3 hrs) — Consolidation + Small Project
**Task:** Build one small self-contained thing that uses everything from this week:

**"Rotate and analyze a point cloud"**
```python
# Generate a synthetic 2D "part outline" - e.g., points forming a rough rectangle
points = np.array([
    [0, 0], [2, 0], [2, 1], [0, 1],
    [0.5, 0.5], [1.5, 0.5]  # a couple extra "feature" points
])

# 1. Rotate the whole point set by 30 degrees (Day 3 skill)
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
rotated_points = points @ R.T

# 2. Compute the "centroid" and center the points
centroid = points.mean(axis=0)
centered_points = points - centroid

# 3. Find the covariance matrix and its eigenvectors — the principal axes of the shape
cov_matrix = np.cov(centered_points.T)
eigvals, eigvecs = np.linalg.eig(cov_matrix)
print(f"Principal axis directions:\n{eigvecs}")

# 4. Plot original points, rotated points, and the principal axes as arrows from centroid
```
This is a scaled-down preview of exactly what PCA does in Week 6, and geometrically similar to how orientation/pose estimation works in vision — very relevant to your background.

**Checkbox:** ⬜ Notebook runs end to end, plot shows original shape, rotated shape, and principal axis arrows
**Checkbox:** ⬜ Commit everything to Git: `git add . && git commit -m "Week 1: Linear algebra fundamentals"`
**Checkbox:** ⬜ Write 3-4 sentences in the repo README under a "Week 1" heading summarizing what you learned and one thing that clicked

---

## End of Week 1 — Self-Check
You should be able to answer these without looking anything up:
1. What does a dot product tell you about two vectors?
2. Why does matrix multiplication order matter?
3. What is an eigenvector, in plain language?
4. How does this connect to something you've already done in machine vision/optics?

If any of these feel shaky, that's fine — spend an extra day on 3Blue1Brown's relevant episode before moving to Week 2. Don't rush past math you're unsure of; it compounds in Weeks 5-9.
