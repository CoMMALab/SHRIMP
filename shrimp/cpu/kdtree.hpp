// SHRIMP: batched incremental kd-tree for nearest-neighbour queries over robot configurations.
//
// Adapted from VAMP (https://github.com/KavrakiLab/vamp, Apache License 2.0, see LICENSE-VAMP in this directory),
// src/impl/vamp/planning/nn/kdtree.hh as found in the ik-stuff fork. The structure and algorithms are VAMP's:
// leaves of up to LEAF configurations in dimension-major blocks of W lanes, an axis-aligned bounding box on every
// node (grown along the insertion path, used to prune), splits on leaf overflow at the median of the widest box
// axis, nearest-first traversal, a leaf-scan early exit every CHECK dimensions, and (1 + epsilon)-approximate
// pruning. Changes for SHRIMP: VAMP's vector layer is replaced by plain W-lane loops (the compiler vectorises
// them with -march=native), and the quaternion blocks, the curve-parameter coordinate, k-NN, prefetching and
// huge-page hints are removed. New: per-joint metric weights (see set_weights).
//
// Metric: sqrt(sum_j (w_j (a_j - b_j))^2). Weights are applied to configurations on insert and to queries, so
// the tree itself is plain L2 on weighted coordinates; set them before inserting. Distances returned are squared
// and weighted.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

template <int D, int LEAF = 128, int W = 8>
struct KDTree {
    static_assert(LEAF % W == 0, "LEAF must be a multiple of W");
    static constexpr int NB = LEAF / W;
    static constexpr int CHECK = 8;                       // leaf-scan early-exit test every CHECK dimensions
    static constexpr std::uint32_t nil = std::numeric_limits<std::uint32_t>::max();
    static constexpr float inf = std::numeric_limits<float>::infinity();

    struct Leaf {
        alignas(32) std::array<float, NB * D * W> data = {};     // zeroed: partial blocks never read garbage
        std::array<std::uint32_t, LEAF> indices = {};
        std::uint32_t count = 0;

        float coord(int i, int j) const { return data[((i / W) * D + j) * W + (i % W)]; }
        void append(const float* q, std::uint32_t index)
        {
            for (int j = 0; j < D; ++j) data[((count / W) * D + j) * W + (count % W)] = q[j];
            indices[count++] = index;
        }
    };
    struct Node {
        std::array<float, D> lo, hi;
        float split = 0.f;
        std::array<std::uint32_t, 2> children = {nil, nil};
        std::uint32_t leaf = nil;
        std::uint32_t axis = 0;
    };

    KDTree()
    {
        weights_.fill(1.f);
        init_root();
    }

    int size() const { return (int)size_; }

    // (1 + epsilon)-approximate queries: subtrees and leaf blocks that cannot beat the current best by more than
    // the (1 + epsilon) factor are skipped, so the neighbour returned is within (1 + epsilon) of the true nearest
    // distance. epsilon = 0 is exact.
    void set_epsilon(float epsilon)
    {
        const float s = 1.f + std::max(epsilon, 0.f);
        prune_scale_ = 1.f / (s * s);
    }
    // Per-joint metric weights (D of them); before inserting.
    void set_weights(const float* w) { std::copy_n(w, D, weights_.begin()); }

    void reserve(int n)
    {
        leaves_.reserve(2 * n / LEAF + 2);
        nodes_.reserve(4 * n / LEAF + 3);
    }

    void insert(int index, const float* q)
    {
        std::array<float, D> qa;
        for (int j = 0; j < D; ++j) qa[j] = q[j] * weights_[j];
        std::uint32_t ni = 0;
        for (;;) {
            grow(nodes_[ni], qa);
            if (nodes_[ni].leaf == nil) {
                const Node& n = nodes_[ni];
                ni = n.children[qa[n.axis] < n.split ? 0 : 1];
                continue;
            }
            if (leaves_[nodes_[ni].leaf].count == LEAF) {
                split_leaf(ni);
                const Node& n = nodes_[ni];
                ni = n.children[qa[n.axis] < n.split ? 0 : 1];
                continue;
            }
            leaves_[nodes_[ni].leaf].append(qa.data(), (std::uint32_t)index);
            ++size_;
            return;
        }
    }

    // Index of the nearest node (squared weighted distance in d2); the tree must not be empty.
    int nearest(const float* q, float& d2) const
    {
        std::array<float, D> qw;
        for (int j = 0; j < D; ++j) qw[j] = q[j] * weights_[j];
        d2 = inf;
        std::uint32_t best = 0;
        search(0, qw.data(), d2, best);
        return (int)best;
    }

private:
    void init_root()
    {
        Node root;
        root.lo.fill(inf);
        root.hi.fill(-inf);
        root.leaf = 0;
        nodes_.push_back(root);
        leaves_.emplace_back();
    }

    static void grow(Node& n, const std::array<float, D>& q)
    {
        for (int j = 0; j < D; ++j) { n.lo[j] = std::min(n.lo[j], q[j]); n.hi[j] = std::max(n.hi[j], q[j]); }
    }

    // Squared distance from q to the node's box (a lower bound for every config below it).
    static float bound2(const Node& n, const float* q)
    {
        float b = 0.f;
        for (int j = 0; j < D; ++j) {
            const float d = std::max(std::max(n.lo[j] - q[j], q[j] - n.hi[j]), 0.f);
            b += d * d;
        }
        return b;
    }

    void scan_leaf(const Leaf& leaf, const float* q, float& best, std::uint32_t& best_i) const
    {
        const int active = (leaf.count + W - 1) / W;
        for (int b = 0; b < active; ++b) {
            const float cutoff = best * prune_scale_;
            float acc[W] = {};
            bool viable = true;
            for (int j = 0; j < D; ++j) {
                if (j != 0 && j % CHECK == 0) {                      // every lane already past the bound: skip
                    bool any = false;
                    for (int l = 0; l < W; ++l) any |= acc[l] <= cutoff;
                    if (!any) { viable = false; break; }
                }
                const float* r = leaf.data.data() + ((size_t)b * D + j) * W;
                for (int l = 0; l < W; ++l) { const float d = r[l] - q[j]; acc[l] += d * d; }
            }
            if (!viable) continue;
            const int lanes = std::min(W, (int)leaf.count - b * W);
            for (int l = 0; l < lanes; ++l) {
                const std::uint32_t idx = leaf.indices[b * W + l];
                if (acc[l] < best || (acc[l] == best && idx < best_i)) { best = acc[l]; best_i = idx; }
            }
        }
    }

    void search(std::uint32_t ni, const float* q, float& best, std::uint32_t& best_i) const
    {
        const Node& n = nodes_[ni];
        if (n.leaf != nil) { scan_leaf(leaves_[n.leaf], q, best, best_i); return; }
        const float b[2] = {bound2(nodes_[n.children[0]], q), bound2(nodes_[n.children[1]], q)};
        const int near = b[0] <= b[1] ? 0 : 1;
        if (b[near] <= best * prune_scale_) search(n.children[near], q, best, best_i);
        if (b[1 - near] <= best * prune_scale_) search(n.children[1 - near], q, best, best_i);
    }

    void split_leaf(std::uint32_t ni)
    {
        const std::uint32_t li = nodes_[ni].leaf;
        std::uint32_t axis = 0;
        float best_extent = -inf;
        for (int j = 0; j < D; ++j) {
            const float extent = nodes_[ni].hi[j] - nodes_[ni].lo[j];
            if (extent > best_extent) { best_extent = extent; axis = j; }
        }
        const Leaf snapshot = leaves_[li];
        std::array<std::uint32_t, LEAF> order;
        std::iota(order.begin(), order.end(), 0u);
        constexpr int half = LEAF / 2;
        std::nth_element(order.begin(), order.begin() + half, order.end(), [&](std::uint32_t a, std::uint32_t c) {
            const float ca = snapshot.coord(a, axis), cc = snapshot.coord(c, axis);
            return ca != cc ? ca < cc : snapshot.indices[a] < snapshot.indices[c];
        });
        const float split = snapshot.coord(order[half], axis);
        std::sort(order.begin(), order.begin() + half);                  // insertion order within each side
        std::sort(order.begin() + half, order.end());

        const std::uint32_t ri = (std::uint32_t)leaves_.size();
        leaves_.emplace_back();
        leaves_[li] = Leaf{};
        Node left, right;
        left.lo.fill(inf); left.hi.fill(-inf); left.leaf = li;
        right.lo.fill(inf); right.hi.fill(-inf); right.leaf = ri;
        std::array<float, D> tmp;
        for (int i = 0; i < LEAF; ++i) {
            const std::uint32_t pos = order[i];
            for (int j = 0; j < D; ++j) tmp[j] = snapshot.coord(pos, j);
            Node& child = i < half ? left : right;
            grow(child, tmp);
            leaves_[child.leaf].append(tmp.data(), snapshot.indices[pos]);
        }
        const std::uint32_t ci = (std::uint32_t)nodes_.size();
        nodes_.push_back(left);
        nodes_.push_back(right);
        Node& parent = nodes_[ni];
        parent.leaf = nil;
        parent.axis = axis;
        parent.split = split;
        parent.children = {ci, ci + 1};
    }

    float prune_scale_ = 1.f;
    std::array<float, D> weights_;
    std::vector<Node> nodes_;
    std::vector<Leaf> leaves_;
    std::size_t size_ = 0;
};
