// Host-program helpers shared by the CPU benchmark mains (rrtc_main.hpp, crrtc_main.hpp): raw-file IO and the
// per-problem obstacle sort. Concatenated before the main by src/rrtc.py.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

template <class T> static bool readf(const char* fn, std::vector<T>& v)
{
    FILE* f = fopen(fn, "rb");
    if (!f) return false;
    const size_t n = fread(v.data(), sizeof(T), v.size(), f);
    fclose(f);
    return n == v.size();
}
template <class T> static void writef(const char* fn, const std::vector<T>& v)
{
    FILE* f = fopen(fn, "wb");
    fwrite(v.data(), sizeof(T), v.size(), f);
    fclose(f);
}

// Obstacles (K floats each) sorted by a lower bound on their distance from the origin, nearest first, as VAMP's
// Environment::sort; env_hit stops at the first obstacle farther than |sphere centre| + r.
template <int K, class Dist>
static void sort_env(const float* src, int n, Dist dist, std::vector<float>& out, std::vector<float>& dmin)
{
    std::vector<std::pair<float, int>> key(n);
    for (int i = 0; i < n; ++i) key[i] = {dist(src + K * i) - 1e-5f, i};     // 1e-5: rounding safety
    std::sort(key.begin(), key.end());
    out.resize((size_t)n * K);
    dmin.resize(n);
    for (int i = 0; i < n; ++i) {
        std::copy_n(src + (size_t)K * key[i].second, K, out.begin() + (size_t)K * i);
        dmin[i] = key[i].first;
    }
}

// The three obstacle sorts of one problem (spheres, boxes, capsules; layouts as the World in the robot header), with the
// distance of each shape from the origin: sphere |c| - r, box exact origin -> box distance, capsule origin -> segment - r.
struct SortedEnv {
    std::vector<float> sp, bx, cp, smin, bmin, cmin;
    void build(const float* sph, int ns, const float* box, int nb, const float* cap, int nc)
    {
        sort_env<4>(sph, ns, [](const float* s) { return std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) - s[3]; },
                    sp, smin);
        sort_env<15>(box, nb, [](const float* b) {
            const float lx = -(b[0] * b[3] + b[1] * b[6] + b[2] * b[9]);          // R^T (0 - c), R row-major
            const float ly = -(b[0] * b[4] + b[1] * b[7] + b[2] * b[10]);
            const float lz = -(b[0] * b[5] + b[1] * b[8] + b[2] * b[11]);
            const float ox = std::max(std::fabs(lx) - b[12], 0.f), oy = std::max(std::fabs(ly) - b[13], 0.f),
                        oz = std::max(std::fabs(lz) - b[14], 0.f);
            return std::sqrt(ox * ox + oy * oy + oz * oz); }, bx, bmin);
        sort_env<7>(cap, nc, [](const float* c) {
            const float sx = c[3] - c[0], sy = c[4] - c[1], sz = c[5] - c[2];
            const float len2 = std::max(sx * sx + sy * sy + sz * sz, 1e-12f);
            const float s = std::min(std::max(-(c[0] * sx + c[1] * sy + c[2] * sz) / len2, 0.f), 1.f);
            const float dx = c[0] + s * sx, dy = c[1] + s * sy, dz = c[2] + s * sz;
            return std::sqrt(dx * dx + dy * dy + dz * dz) - c[6]; }, cp, cmin);
    }
};
