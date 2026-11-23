// Benchmark host program for crrtc.cuh (file-driven, like rrtc_main.cuh; host_util.hpp is concatenated before it).
// argv: P B max_iters max_samples range seed MS MB MC lat_n cparams starts goals env env_n tsr
//       out_path out_len out_stat out_lat      (files are raw float32 / int32)
// cparams: 18 float32, in this order: lanes perturb slack reached2 emit_all edge_mode smooth freeze closer closer2
//          smooth_rate spacing gap_slack proj_iters lambda rate stall2 hold   (see CParams / ProjParams)
// tsr: P x Tsr (float32; crab_codegen.constraint_header.pack_tsr per problem). env as rrtc_main.cuh (host-sorted here).
// stdout: median ms of one launch over all P problems; out_lat: ms per problem (lat_n problems spread over the set, one
// problem per launch); out_stat: int32 (solved, iters) per problem, solved: 1 yes, 0 / -1 out of iterations or a tree
// filled up (not told apart), -2 start or goal not projectable onto the region; out_len: path length (-1 = too long).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int CP_N = 18;

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
    const char *fcp = argv[a++], *fs = argv[a++], *fg = argv[a++], *fe = argv[a++], *fn = argv[a++], *ft = argv[a++];
    const char *fpath = argv[a++], *flen = argv[a++], *fstat = argv[a++], *flat = argv[a++];
    const size_t esz = (size_t)prm.MS * 4 + (size_t)prm.MB * 15 + (size_t)prm.MC * 7, ms = prm.max_samples;
    const size_t dsz = (size_t)prm.MS + prm.MB + prm.MC;

    std::vector<float> cpv(CP_N);
    std::vector<float> starts((size_t)P * N), goals((size_t)P * N), env((size_t)P * esz);
    std::vector<int> env_n(3 * (size_t)P);
    std::vector<Tsr> tsr(P);
    if (!readf(fcp, cpv) || !readf(fs, starts) || !readf(fg, goals) || !readf(fe, env) || !readf(fn, env_n) ||
        !readf(ft, tsr))
        return 1;
    CParams cp{};
    cp.lanes = (int)cpv[0];
    cp.perturb = cpv[1];
    cp.slack = cpv[2];
    cp.reached2 = cpv[3];
    cp.emit_all = (int)cpv[4];
    cp.edge_mode = (int)cpv[5];
    cp.smooth = (int)cpv[6];
    cp.freeze = (int)cpv[7];
    cp.closer = (int)cpv[8];
    cp.closer2 = cpv[9];
    cp.smooth_rate = cpv[10];
    cp.spacing = cpv[11];
    cp.gap_slack = cpv[12];
    cp.proj.max_iters = (int)cpv[13];
    cp.proj.lambda = cpv[14];
    cp.proj.rate = cpv[15];
    cp.proj.stall2 = cpv[16];
    cp.proj.drift2 = 0.f;
    cp.proj.hold = (int)cpv[17];

    std::vector<float> dmin((size_t)P * dsz, 0.f);       // obstacles sorted nearest-first per problem (env_hit's cutoff)
    for (int p = 0; p < P; ++p) {
        float* sph = env.data() + (size_t)p * esz;
        float* box = sph + (size_t)prm.MS * 4;
        float* cap = box + (size_t)prm.MB * 15;
        SortedEnv se;
        se.build(sph, env_n[3 * p], box, env_n[3 * p + 1], cap, env_n[3 * p + 2]);
        std::copy(se.sp.begin(), se.sp.end(), sph);
        std::copy(se.bx.begin(), se.bx.end(), box);
        std::copy(se.cp.begin(), se.cp.end(), cap);
        float* dm = dmin.data() + (size_t)p * dsz;
        std::copy(se.smin.begin(), se.smin.end(), dm);
        std::copy(se.bmin.begin(), se.bmin.end(), dm + prm.MS);
        std::copy(se.cmin.begin(), se.cmin.end(), dm + prm.MS + prm.MB);
    }

    float *ds, *dg, *denv, *ddmin, *dnodes, *dpath;
    int *denvn, *dpar, *drdy, *dlen;
    Tsr* dtsr;
    PState* dst;
    cudaMalloc(&ds, starts.size() * 4);
    cudaMalloc(&dg, goals.size() * 4);
    cudaMalloc(&denv, env.size() * 4);
    cudaMalloc(&ddmin, dmin.size() * 4);
    cudaMalloc(&denvn, env_n.size() * 4);
    cudaMalloc(&dtsr, (size_t)P * sizeof(Tsr));
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
    cudaMemcpy(ddmin, dmin.data(), dmin.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(denvn, env_n.data(), env_n.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtsr, tsr.data(), (size_t)P * sizeof(Tsr), cudaMemcpyHostToDevice);

    int epoch = 0;
    auto run = [&](int nP, int p0) {                 // plan problems p0 .. p0 + nP - 1 (one launch pair)
        prm.epoch = ++epoch;
        prm.p0 = p0;
        k_cinit<<<(nP + 63) / 64, 64>>>(ds + (size_t)p0 * N, dg + (size_t)p0 * N, dtsr + p0,
                                        dnodes + (size_t)p0 * 2 * ms * N, dpar + (size_t)p0 * 2 * ms,
                                        drdy + (size_t)p0 * 2 * ms, dst + p0, dlen + p0, prm, cp, nP);
        k_crrtc<<<nP * prm.B, NT>>>(denv + (size_t)p0 * esz, ddmin + (size_t)p0 * dsz, denvn + 3 * (size_t)p0, dtsr + p0,
                                    dnodes + (size_t)p0 * 2 * ms * N, dpar + (size_t)p0 * 2 * ms,
                                    drdy + (size_t)p0 * 2 * ms, dst + p0, dpath + (size_t)p0 * MAXP * N, dlen + p0, prm, cp);
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
    for (int i = 0; i < lat_n; ++i) {                // problems spread evenly over the set (it is sorted by obstacle count)
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
    for (int p = 0; p < P; ++p) { stat[2 * p] = st[p].solved; stat[2 * p + 1] = st[p].iters; }
    writef(fpath, path);
    writef(flen, len);
    writef(fstat, stat);
    writef(flat, lat);
    std::nth_element(batch.begin(), batch.begin() + batch.size() / 2, batch.end());
    printf("%.6f\n", batch[batch.size() / 2]);
    return 0;
}
