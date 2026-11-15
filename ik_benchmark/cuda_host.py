"""Host program shared by the fused IK benchmarks (crocket_ik.py and crab_ik.py): a CUDA main() that times
latency per target and one launch over all targets, then writes every seed's solution. Each build supplies,
before this text: N (dofs), TS (floats of target per problem), g_mode and launch(n, S, seeds, targets, out)."""

MAIN = r"""
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv)
{
    // <mode> <P> <S> <seeds.f32> <targets.f32> <out.f32> <latency_ms.f32>
    g_mode = argv[1];
    const int P = atoi(argv[2]), S = atoi(argv[3]), K = 5;
    std::vector<float> seeds((size_t)P * S * N), targets((size_t)P * TS), out((size_t)P * S * N), lat(P);
    FILE* f = fopen(argv[4], "rb");
    if (!f || fread(seeds.data(), sizeof(float), seeds.size(), f) != seeds.size()) return 1;
    fclose(f);
    f = fopen(argv[5], "rb");
    if (!f || fread(targets.data(), sizeof(float), targets.size(), f) != targets.size()) return 1;
    fclose(f);
    float *ds, *dt, *dout;
    cudaMalloc(&ds, seeds.size() * sizeof(float));
    cudaMalloc(&dt, targets.size() * sizeof(float));
    cudaMalloc(&dout, out.size() * sizeof(float));
    cudaMemcpy(ds, seeds.data(), seeds.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt, targets.data(), targets.size() * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 2; ++i) launch(P * S, S, ds, dt, dout);
    cudaDeviceSynchronize();
    cudaEvent_t a, e;
    cudaEventCreate(&a);
    cudaEventCreate(&e);

    std::vector<float> batch;
    for (int rep = 0; rep < 5; ++rep) {
        cudaEventRecord(a);
        for (int k = 0; k < K; ++k) launch(P * S, S, ds, dt, dout);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, a, e);
        batch.push_back(ms / K);
    }
    for (int p = 0; p < P; ++p) {  // one target (S seeds) per launch
        cudaEventRecord(a);
        for (int k = 0; k < K; ++k) launch(S, S, ds + (size_t)p * S * N, dt + (size_t)p * TS, dout + (size_t)p * S * N);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, a, e);
        lat[p] = ms / K;
    }
    launch(P * S, S, ds, dt, dout);  // results from the full batch
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 2;
    cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    f = fopen(argv[6], "wb");
    fwrite(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    f = fopen(argv[7], "wb");
    fwrite(lat.data(), sizeof(float), lat.size(), f);
    fclose(f);
    std::nth_element(batch.begin(), batch.begin() + batch.size() / 2, batch.end());
    printf("%.6f\n", batch[batch.size() / 2]);
    return 0;
}
"""


def compile_cu(src: str, stem: str, build_dir, flags=()):
    """Write `src` to build_dir/<stem>.cu and nvcc it (skipped if the source is unchanged)."""
    import os
    import subprocess
    from pathlib import Path

    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    cu, exe = build_dir / f"{stem}.cu", build_dir / stem
    if not exe.exists() or not cu.exists() or cu.read_text() != src:
        cu.write_text(src)
        subprocess.check_call(["nvcc", "-O3", "-std=c++17", os.environ.get("CRAB_ARCH", "-arch=native"),
                               *flags, str(cu), "-o", str(exe)])
    return exe


def run_exe(exe, mode: str, seeds, targets):
    """Run a compiled MAIN. seeds (P, S, dof), targets (P, TS) float32 -> (q (P, S, dof) float64,
    latency_ms per target, batch_ms for all targets in one launch)."""
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np

    P, S, dof = seeds.shape
    with tempfile.TemporaryDirectory() as d:
        sp, tp, op, lp = (Path(d) / n for n in ("seeds.f32", "targets.f32", "out.f32", "lat.f32"))
        np.ascontiguousarray(seeds, np.float32).tofile(sp)
        np.ascontiguousarray(targets, np.float32).tofile(tp)
        batch_ms = float(subprocess.check_output(
            [str(exe), mode, str(P), str(S), str(sp), str(tp), str(op), str(lp)], text=True))
        q = np.fromfile(op, np.float32).reshape(P, S, dof).astype(np.float64)
        lat = np.fromfile(lp, np.float32)
    return q, lat, batch_ms
