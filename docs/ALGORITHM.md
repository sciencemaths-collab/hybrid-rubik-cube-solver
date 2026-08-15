# Algorithm Design

## 1. Discrete cube-search layer

The cube module represents a 3×3 cube as 54 facelet slots embedded in a 3D coordinate system. Legal moves are generated as exact quarter-turn permutations for the six faces \(U,D,L,R,F,B\), including inverse and half turns.

Two transparent baseline searches are included:

- **BFS**: uninformed shortest-path baseline for shallow scrambles.
- **A\***: uses a conservative misplaced-facelet heuristic to prioritize states closer to the goal.

The discrete benchmark is intentionally shallow and interpretable. It exists to study state encoding, search expansion, pruning, and candidate ranking rather than to compete with highly optimized specialist cube solvers.

## 2. Hybrid Adaptive Topological Energy Minimization (HATEM)

The continuous prototype represents \(n\) heterogeneous mini-units at positions \(X\in\mathbb{R}^{n\times3}\), with target configuration \(X^*\).

Each mini-unit receives synthetic heterogeneous parameters:

- frequency-like parameter \(f_i\),
- vibration-like parameter \(v_i\),
- base energy/stiffness parameter \(e_i\).

A representative objective is

\[
E(X)=\alpha E_{stretch}+\beta E_{bend}+\gamma E_{rep}+E_{target}+\eta E_{field}.
\]

The target term uses rigid Kabsch alignment:

\[
E_{target}=\|XR+t-X^*\|_F^2.
\]

### Continuous track

SPSA estimates a gradient from two objective evaluations:

\[
\widehat g_t = \frac{E(X_t+a_t\Delta_t)-E(X_t-a_t\Delta_t)}{2a_t}\,\Delta_t.
\]

A candidate update is

\[
X' = X_t-\eta_t\widehat g_t+\xi_t,
\]

with Metropolis acceptance under a cooling schedule.

### Discrete / macro-move track

The current macro library includes contiguous block swaps and rotations. Candidate moves are described by local features such as block length, frequency mismatch, minimum separation, perturbation state, and curvature proxy.

A small learned policy network estimates expected improvement. A UCB controller balances exploitation with exploration across macro-move families.

The key architectural idea is therefore:

> **Use continuous relaxation for local improvement, but permit structural macro-moves when the current landscape makes local descent inefficient.**

## 3. Perturbation model

The synthetic benchmark includes six controlled scenarios:

- baseline,
- bloat,
- shrink,
- electric forcing,
- added heat,
- removed heat.

These alter preferred geometry, stochasticity, or field terms and are used to test whether the adaptive hybrid remains robust when the landscape changes.

## 4. Comparable progress metric

Within the HATEM benchmark family, all solvers receive the same objective-evaluation budget. If \(b_k\) is best-so-far objective value, normalized progress is

\[
P_k=\frac{b_0-b_k}{b_0-\min(b)+\varepsilon}.
\]

The reported score is

\[
Score=0.7P_{last}+0.3\,AUC(P).
\]

This metric compares convergence behavior under a shared evaluation budget. It does **not** make unrelated application domains intrinsically equivalent.
