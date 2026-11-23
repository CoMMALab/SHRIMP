// Benchmark host program for cpu/crrtc.hpp (file-driven, like rrtc_main.hpp).
// argv: P threads max_iters max_samples range seed MS MB MC cparams starts goals env env_n tsr
//       out_path out_len out_stat out_ms      (files are raw float32 / int32)
// cparams: 17 float32, in this order: lanes perturb slack reached2 emit_all edge_mode smooth freeze closer closer2
//          smooth_rate spacing proj_iters lambda rate stall2 hold   (see CParams / ProjParams)
// tsr: P x Tsr (float32; crab_codegen.constraint_header.pack_tsr per problem). env as rrtc_main.hpp.
// stdout: wall ms for all P problems on `threads` threads. out_ms: planning ms per problem (one core);
// out_stat: int32 (solved, iters, fkcc calls, full task-space evaluations, error-only evaluations) per problem; out_len: int32 path length (-1 = too
// long). solved: 1 yes, 0 out of iterations, -1 a tree filled up, -2 start or goal not projectable onto the region.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

constexpr int CP_N = 17;

#ifdef SHRIMP_PROF
static std::atomic<unsigned long long> g_ptot[4], g_ctot[3];     // summed thread cycle counters (crrtc.hpp, robot header)
#endif

int main(int argc, char** argv)
{
    int a = 1;
    const int P = atoi(argv[a++]), threads = std::max(1, atoi(argv[a++]));
    Params prm{};
    prm.max_iters = atoi(argv[a++]);
    prm.max_samples = atoi(argv[a++]);
    prm.range = (float)atof(argv[a++]);
    prm.seed = (unsigned)atoi(argv[a++]);
    prm.epsilon = 0.f;
    prm.brute = 0;
    for (int k = 0; k < N; ++k) prm.weights[k] = 1.f;
    const int MS = atoi(argv[a++]), MB = atoi(argv[a++]), MC = atoi(argv[a++]);
    const char *fcp = argv[a++], *fs = argv[a++], *fg = argv[a++], *fe = argv[a++], *fn = argv[a++], *ft = argv[a++];
    const char *fpath = argv[a++], *flen = argv[a++], *fstat = argv[a++], *fms = argv[a++];
    const size_t esz = (size_t)MS * 4 + (size_t)MB * 15 + (size_t)MC * 7;

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
    cp.proj.max_iters = (int)cpv[12];
    cp.proj.lambda = cpv[13];
    cp.proj.rate = cpv[14];
    cp.proj.stall2 = cpv[15];
    cp.proj.drift2 = 0.f;
    cp.proj.hold = (int)cpv[16];

    std::vector<float> path((size_t)P * MAXP * N), ms(P);
    std::vector<int> len(P), stat(5 * (size_t)P);
    std::atomic<int> next{0};
    auto worker = [&] {
        for (int p; (p = next++) < P;) {
            const float* sph = env.data() + (size_t)p * esz;
            const float* box = sph + (size_t)MS * 4;
            const float* cap = box + (size_t)MB * 15;
            SortedEnv se;
            se.build(sph, env_n[3 * p], box, env_n[3 * p + 1], cap, env_n[3 * p + 2]);
            World w;
            w.sph = se.sp.data(); w.box = se.bx.data(); w.cap = se.cp.data();
            w.smin = se.smin.data(); w.bmin = se.bmin.data(); w.cmin = se.cmin.data();
            w.ns = env_n[3 * p]; w.nb = env_n[3 * p + 1]; w.nc = env_n[3 * p + 2];
            const auto t0 = std::chrono::steady_clock::now();
            g_fk = 0;
            g_pe = 0;
            g_pc = 0;
            const Result r = plan_c(&starts[(size_t)p * N], &goals[(size_t)p * N], w, tsr[p], prm, cp,
                                    &path[(size_t)p * MAXP * N]);
            ms[p] = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
            stat[5 * p] = r.solved;
            stat[5 * p + 1] = r.iters;
            stat[5 * p + 2] = (int)g_fk;
            stat[5 * p + 3] = (int)g_pe;
            stat[5 * p + 4] = (int)g_pc;
            len[p] = r.len;
        }
#ifdef SHRIMP_PROF
        for (int i = 0; i < 4; ++i) g_ptot[i] += g_pcy[i];
        for (int i = 0; i < 3; ++i) g_ctot[i] += g_cyc[i];
#endif
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    const float wall = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();

#ifdef SHRIMP_PROF
    {   // cycles per call (rdtsc, its own ~30 cycles included) and the shares of the measured pieces
        double ne = 0, nf = 0, nc = 0;
        for (int p = 0; p < P; ++p) { nf += stat[5 * p + 3]; ne += stat[5 * p + 4]; nc += stat[5 * p + 2]; }
        const double tot = (double)g_ptot[0] + g_ptot[1] + g_ptot[2] + g_ptot[3];
        fprintf(stderr, "profile (cycles per call; share of the measured total)\n"
                "  tsr_err  %8.0f  (%.0f calls)  %4.1f%%\n  tsr_eval %8.0f  (%.0f calls)  %4.1f%%\n"
                "  tsr_step %8.0f  (%.0f calls)  %4.1f%%\n  fkcc     %8.0f  (%.0f calls)  %4.1f%%\n",
                g_ptot[0] / std::max(ne, 1.0), ne, 100 * g_ptot[0] / tot, g_ptot[1] / std::max(nf, 1.0), nf,
                100 * g_ptot[1] / tot, g_ptot[2] / std::max(nf, 1.0), nf, 100 * g_ptot[2] / tot,
                g_ptot[3] / std::max(nc, 1.0), nc, 100 * g_ptot[3] / tot);
        const double fk = (double)g_ctot[0] + g_ctot[1] + g_ctot[2];
        fprintf(stderr, "  inside fkcc: FK %.0f (%.0f%%)  obstacles %.0f (%.0f%%)  self pairs %.0f (%.0f%%)\n",
                g_ctot[0] / std::max(nc, 1.0), 100 * g_ctot[0] / fk, g_ctot[1] / std::max(nc, 1.0), 100 * g_ctot[1] / fk,
                g_ctot[2] / std::max(nc, 1.0), 100 * g_ctot[2] / fk);
    }
#endif
    writef(fpath, path);
    writef(flen, len);
    writef(fstat, stat);
    writef(fms, ms);
    printf("%.6f\n", wall);
    return 0;
}
