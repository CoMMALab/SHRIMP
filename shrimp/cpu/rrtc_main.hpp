// Benchmark host program for cpu/rrtc.hpp (file-driven, like cuda/rrtc_main.cuh).
// argv: P threads max_iters max_samples range seed epsilon weights MS MB MC starts goals env env_n
//       (epsilon: NN approximation, 0 = exact; weights: "w0,w1,..." per joint or "-" for ones)
//       out_path out_len out_stat out_ms      (files are raw float32 / int32)
// stdout: wall ms for all P problems on `threads` threads. out_ms: planning ms per problem (one core);
// out_stat: int32 (solved, iters, fkcc calls) per problem; out_len: int32 path length (-1 = too long).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

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

#ifdef SHRIMP_PROF
static std::atomic<unsigned long long> g_tot[3];     // summed thread cycle counters (see fkcc)
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
    prm.epsilon = (float)atof(argv[a++]);
    {   // weights: N comma-separated floats, or "-" for all ones
        const char* s = argv[a++];
        for (int k = 0; k < N; ++k) {
            prm.weights[k] = s[0] == '-' ? 1.f : (float)strtod(s, (char**)&s);
            if (*s == ',') ++s;
        }
    }
    const int MS = atoi(argv[a++]), MB = atoi(argv[a++]), MC = atoi(argv[a++]);
    const char *fs = argv[a++], *fg = argv[a++], *fe = argv[a++], *fn = argv[a++];
    const char *fpath = argv[a++], *flen = argv[a++], *fstat = argv[a++], *fms = argv[a++];
    const size_t esz = (size_t)MS * 4 + (size_t)MB * 15 + (size_t)MC * 7;

    std::vector<float> starts((size_t)P * N), goals((size_t)P * N), env((size_t)P * esz);
    std::vector<int> env_n(3 * (size_t)P);
    if (!readf(fs, starts) || !readf(fg, goals) || !readf(fe, env) || !readf(fn, env_n)) return 1;

    std::vector<float> path((size_t)P * MAXP * N), ms(P);
    std::vector<int> len(P), stat(3 * (size_t)P);
    std::atomic<int> next{0};
    auto worker = [&] {
        for (int p; (p = next++) < P;) {
            World w;
            w.sph = env.data() + (size_t)p * esz;
            w.box = w.sph + (size_t)MS * 4;
            w.cap = w.box + (size_t)MB * 15;
            w.ns = env_n[3 * p]; w.nb = env_n[3 * p + 1]; w.nc = env_n[3 * p + 2];
            const auto t0 = std::chrono::steady_clock::now();
            g_fk = 0;
            const Result r = plan(&starts[(size_t)p * N], &goals[(size_t)p * N], w, prm, &path[(size_t)p * MAXP * N]);
            ms[p] = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
            stat[3 * p] = r.solved == 1;
            stat[3 * p + 1] = r.iters;
            stat[3 * p + 2] = (int)g_fk;
            len[p] = r.len;
        }
#ifdef SHRIMP_PROF
        for (int i = 0; i < 3; ++i) g_tot[i] += g_cyc[i];
#endif
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    const float wall = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();

#ifdef SHRIMP_PROF
    {
        double sum = 0, calls = 0;
        for (int p = 0; p < P; ++p) calls += stat[3 * p + 2];
        for (int i = 0; i < 3; ++i) sum += (double)g_tot[i];
        fprintf(stderr, "profile (cycles per config check, %.0f checks): FK %.0f (%.0f%%)  obstacles %.0f (%.0f%%)  "
                "self pairs %.0f (%.0f%%)  total %.0f\n", calls, g_tot[0] / calls, 100 * g_tot[0] / sum,
                g_tot[1] / calls, 100 * g_tot[1] / sum, g_tot[2] / calls, 100 * g_tot[2] / sum, sum / calls);
    }
#endif
    writef(fpath, path);
    writef(flen, len);
    writef(fstat, stat);
    writef(fms, ms);
    printf("%.6f\n", wall);
    return 0;
}
