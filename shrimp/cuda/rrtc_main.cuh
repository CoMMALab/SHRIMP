// Benchmark host program for rrtc.cuh (file-driven; the library entry point will replace this).
// argv: P B max_iters max_samples range seed MS MB MC lat_n starts goals env env_n
//       out_path out_len out_stat out_lat      (files are raw float32 / int32)
// stdout: median ms of one launch over all P problems; out_lat: ms per problem (lat_n problems spread over the
// set, one problem per launch); out_stat: int32 (solved, iters) per problem; out_len: int32 path length
// (-1 = too long).
#include <algorithm>
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

    float *ds, *dg, *denv, *dnodes, *dpath;
    int *denvn, *dpar, *drdy, *dlen;
    PState* dst;
    cudaMalloc(&ds, starts.size() * 4);
    cudaMalloc(&dg, goals.size() * 4);
    cudaMalloc(&denv, env.size() * 4);
    cudaMalloc(&denvn, env_n.size() * 4);
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
        k_rrtc<<<nP * prm.B, NT>>>(denv + (size_t)p0 * esz, denvn + 3 * (size_t)p0, dnodes + (size_t)p0 * 2 * ms * N,
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
    run(P, 0);                                       // results from the full batch
    cudaDeviceSynchronize();
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
