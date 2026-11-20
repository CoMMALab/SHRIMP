// Benchmark host program for rrtc.cuh (file-driven; the library entry point will replace this).
// argv: P B max_iters max_samples range seed MS MB MC lat_n starts goals env env_n
//       out_path out_len out_stat out_lat      (files are raw float32 / int32)
// stdout: median ms of one launch over all P problems; out_lat: ms per problem (lat_n problems spread over the
// set, one problem per launch); out_stat: int32 (solved, iters) per problem; out_len: int32 path length
// (-1 = too long).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// Sort one problem's K-float obstacles (first n of the padded list) by a lower bound on their distance from the
// origin, nearest first (VAMP's Environment::sort); dmin gets that bound per obstacle (env_hit's early exit).
template <int K, class Dist>
static void sort_env(float* arr, int n, Dist dist, float* dmin)
{
    std::vector<std::pair<float, int>> key(n);
    for (int i = 0; i < n; ++i) key[i] = {dist(arr + K * i) - 1e-5f, i};     // 1e-5: rounding safety
    std::sort(key.begin(), key.end());
    std::vector<float> tmp(arr, arr + (size_t)n * K);
    for (int i = 0; i < n; ++i) {
        std::copy_n(tmp.data() + (size_t)K * key[i].second, K, arr + (size_t)K * i);
        dmin[i] = key[i].first;
    }
}

int main(int argc, char** argv)
{
    int a = 1;
    const int P = atoi(argv[a++]);
    Params prm{};
    prm.B = atoi(argv[a++]);
    prm.max_iters = atoi(argv[a++]);
    prm.max_samples = atoi(argv[a++]);
    prm.range = (float)atof(argv[a++]);
    prm.seed = (unsigned)atoi(argv[a++]);
    prm.MS = atoi(argv[a++]);
    prm.MB = atoi(argv[a++]);
    prm.MC = atoi(argv[a++]);
    const int lat_n = std::min(P, atoi(argv[a++]));
    const char *fs = argv[a++], *fg = argv[a++], *fe = argv[a++], *fn = argv[a++];
    const char *fpath = argv[a++], *flen = argv[a++], *fstat = argv[a++], *flat = argv[a++];
    const size_t esz = (size_t)prm.MS * 4 + (size_t)prm.MB * 15 + (size_t)prm.MC * 7, ms = prm.max_samples;

    std::vector<float> starts((size_t)P * N), goals((size_t)P * N), env((size_t)P * esz);
    std::vector<int> env_n(3 * (size_t)P);
    if (!readf(fs, starts) || !readf(fg, goals) || !readf(fe, env) || !readf(fn, env_n)) return 1;

    const size_t dsz = (size_t)prm.MS + prm.MB + prm.MC;
    std::vector<float> dmin((size_t)P * dsz, 0.f);
    for (int p = 0; p < P; ++p) {
        float* sph = env.data() + (size_t)p * esz;
        float* box = sph + (size_t)prm.MS * 4;
        float* cap = box + (size_t)prm.MB * 15;
        float* dm = dmin.data() + (size_t)p * dsz;
        sort_env<4>(sph, env_n[3 * p], [](const float* s) { return std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) - s[3]; }, dm);
        sort_env<15>(box, env_n[3 * p + 1], [](const float* b) {      // distance origin -> box
            const float lx = -(b[0] * b[3] + b[1] * b[6] + b[2] * b[9]);          // R^T (0 - c), R row-major
            const float ly = -(b[0] * b[4] + b[1] * b[7] + b[2] * b[10]);
            const float lz = -(b[0] * b[5] + b[1] * b[8] + b[2] * b[11]);
            const float ox = std::max(std::fabs(lx) - b[12], 0.f), oy = std::max(std::fabs(ly) - b[13], 0.f),
                        oz = std::max(std::fabs(lz) - b[14], 0.f);
            return std::sqrt(ox * ox + oy * oy + oz * oz); }, dm + prm.MS);
        sort_env<7>(cap, env_n[3 * p + 2], [](const float* c) {       // distance origin -> segment, minus r
            const float sx = c[3] - c[0], sy = c[4] - c[1], sz = c[5] - c[2];
            const float len2 = std::max(sx * sx + sy * sy + sz * sz, 1e-12f);
            const float s = std::min(std::max(-(c[0] * sx + c[1] * sy + c[2] * sz) / len2, 0.f), 1.f);
            const float dx = c[0] + s * sx, dy = c[1] + s * sy, dz = c[2] + s * sz;
            return std::sqrt(dx * dx + dy * dy + dz * dz) - c[6]; }, dm + prm.MS + prm.MB);
    }

    float *ds, *dg, *denv, *ddmin, *dnodes, *dpath;
    int *denvn, *dpar, *drdy, *dlen;
    PState* dst;
    cudaMalloc(&ds, starts.size() * 4);
    cudaMalloc(&dg, goals.size() * 4);
    cudaMalloc(&denv, env.size() * 4);
    cudaMalloc(&denvn, env_n.size() * 4);
    cudaMalloc(&ddmin, dmin.size() * 4);
    cudaMemcpy(ddmin, dmin.data(), dmin.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dnodes, (size_t)P * 2 * ms * N * 4);
    cudaMalloc(&dpar, (size_t)P * 2 * ms * 4);
    cudaMalloc(&drdy, (size_t)P * 2 * ms * 4);
    cudaMemset(drdy, 0, (size_t)P * 2 * ms * 4);
    cudaMalloc(&dst, (size_t)P * sizeof(PState));
    cudaMalloc(&dpath, (size_t)P * MAXP * N * 4);
    cudaMalloc(&dlen, (size_t)P * 4);
    cudaMemcpy(ds, starts.data(), starts.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, goals.data(), goals.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(denv, env.data(), env.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(denvn, env_n.data(), env_n.size() * 4, cudaMemcpyHostToDevice);

    int epoch = 0;
    auto run = [&](int nP, int p0) {                 // plan problems p0 .. p0 + nP - 1 (one launch)
        prm.epoch = ++epoch;
        prm.p0 = p0;
        k_init<<<(nP + 63) / 64, 64>>>(ds + (size_t)p0 * N, dg + (size_t)p0 * N, dnodes + (size_t)p0 * 2 * ms * N,
                                       dpar + (size_t)p0 * 2 * ms, drdy + (size_t)p0 * 2 * ms, dst + p0, dlen + p0,
                                       prm, nP);
        k_rrtc<<<nP * prm.B, NT>>>(denv + (size_t)p0 * esz, ddmin + (size_t)p0 * dsz, denvn + 3 * (size_t)p0, dnodes + (size_t)p0 * 2 * ms * N,
                                   dpar + (size_t)p0 * 2 * ms, drdy + (size_t)p0 * 2 * ms, dst + p0,
                                   dpath + (size_t)p0 * MAXP * N, dlen + p0, prm);
    };

    for (int i = 0; i < 2; ++i) run(P, 0);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return 2; }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<float> batch, lat(lat_n);
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(e0);
        run(P, 0);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float t;
        cudaEventElapsedTime(&t, e0, e1);
        batch.push_back(t);
    }
    for (int i = 0; i < lat_n; ++i) {                // problems spread evenly over the set (it is sorted by family)
        cudaEventRecord(e0);
        run(1, (int)((size_t)i * P / lat_n));
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&lat[i], e0, e1);
    }
#ifdef SHRIMP_PROF
    { unsigned long long z[16] = {}; cudaMemcpyToSymbol(g_prof, z, sizeof z); }
#endif
    run(P, 0);                                       // results from the full batch
    cudaDeviceSynchronize();
#ifdef SHRIMP_PROF
    {
        unsigned long long h[16];
        cudaMemcpyFromSymbol(h, g_prof, sizeof h);
        const double calls = (double)h[3], fk = (double)(h[0] + h[1] + h[2]), bt = (double)h[7];
        fprintf(stderr, "profile (clock64 cycles of one warp; elapsed time, includes waiting for other warps)\n"
                "  fkcc per edge check (%.0f checks): FK %.0f (%.0f%%)  obstacles %.0f (%.0f%%)  self pairs %.0f (%.0f%%)  "
                "total %.0f\n", calls, h[0] / calls, 100 * h[0] / fk, h[1] / calls, 100 * h[1] / fk, h[2] / calls,
                100 * h[2] / fk, fk / calls);
        fprintf(stderr, "  per block (%llu blocks, %.0f cycles): nearest %.0f%%  edge_ok %.0f%%  add_node %.0f%%  "
                "other %.0f%%\n", h[8], bt / h[8], 100 * h[4] / bt, 100 * h[5] / bt, 100 * h[6] / bt,
                100 * (bt - h[4] - h[5] - h[6]) / bt);
    }
#endif
    if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaGetLastError())); return 2; }

    std::vector<PState> st(P);
    std::vector<int> len(P), stat(2 * (size_t)P);
    std::vector<float> path((size_t)P * MAXP * N);
    cudaMemcpy(st.data(), dst, (size_t)P * sizeof(PState), cudaMemcpyDeviceToHost);
    cudaMemcpy(len.data(), dlen, (size_t)P * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), dpath, path.size() * 4, cudaMemcpyDeviceToHost);
    for (int p = 0; p < P; ++p) { stat[2 * p] = st[p].solved == 1; stat[2 * p + 1] = st[p].iters; }
    writef(fpath, path);
    writef(flen, len);
    writef(fstat, stat);
    writef(flat, lat);
    std::nth_element(batch.begin(), batch.begin() + batch.size() / 2, batch.end());
    printf("%.6f\n", batch[batch.size() / 2]);
    return 0;
}
