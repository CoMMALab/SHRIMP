#include <jit/compiler.hh>
#include <jit/object_cache.hh>
#include <jit/session.hh>

#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

struct CrustyJitContext
{
    std::shared_ptr<crusty::jit::DiskObjectCache> cache;
    crusty::jit::ClangCompiler compiler;
    std::unique_ptr<crusty::jit::JitSession> session;
    std::unordered_map<std::string, void *> symbols;
};

extern "C"
{
    CrustyJitContext *crusty_jit_create()
    {
        auto ctx = new CrustyJitContext();
        ctx->cache = std::make_shared<crusty::jit::DiskObjectCache>(crusty::jit::default_cache_dir());
        ctx->session = std::make_unique<crusty::jit::JitSession>(ctx->cache);
        return ctx;
    }

    void crusty_jit_destroy(CrustyJitContext *ctx)
    {
        delete ctx;
    }

    int crusty_jit_compile(CrustyJitContext *ctx, const char *source, const char *module_id)
    {
        if (!ctx || !source || !module_id)
        {
            // Complain with failure
            llvm::errs() << "crusty::jit: invalid arguments to crusty_jit_compile\n";
            return -1;
        }

        crusty::jit::CompileOptions opts;
#ifdef CRUSTY_EIGEN_INCLUDE_DIR
        // JIT'd sources (SymForce-generated FK/Jacobian, the fused collision kernel)
        // #include <Eigen/Dense>. The Clang driver invocation in ClangCompiler::compile
        // has no notion of this process's own build flags, so Eigen's include dir has
        // to be passed explicitly -- CRUSTY_EIGEN_INCLUDE_DIR is baked in at build time
        // from the Eigen3 CMake package (see crusty_compilations/CMakeLists.txt).
        opts.system_include_dirs.push_back(CRUSTY_EIGEN_INCLUDE_DIR);
#endif
        // The disk cache is keyed by opts.module_id alone. If we used the caller's
        // module_id verbatim, editing the source (e.g. regenerating a kernel template)
        // without also changing module_id would silently hit a stale cached .o forever.
        // Fold a content hash (source + flags) into the key so any source/flag change
        // invalidates the cache, while keeping the caller's id as a readable prefix.
        const std::string cache_key =
            std::string(module_id) + "-" + crusty::jit::hash_source(source, opts);
        opts.module_id = cache_key;

        auto cached_obj = ctx->cache->load_object(cache_key);
        if (cached_obj)
        {
            if (auto err = ctx->session->add_object_file(std::move(cached_obj)))
            {
                llvm::consumeError(std::move(err));
                llvm::errs() << "crusty::jit: add_object_file failed: " << err << "\n";
                return -2;
            }
            return 1; // Cache hit
        }

        try
        {
            auto tsm = ctx->compiler.compile(source, opts);
            if (auto err = ctx->session->add_module(std::move(tsm)))
            {
                llvm::consumeError(std::move(err));
                llvm::errs() << "crusty::jit: add_module failed\n";
                return -3;
            }
        }
        catch (const std::exception &e)
        {
            // llvm::consumeError(std::move(e));
            llvm::errs() << "crusty::jit: compile failed: " << e.what() << "\n";
            return -4;
        }
        return 0; // compiled fresh
    }

    void *crusty_jit_lookup(CrustyJitContext *ctx, const char *symbol)
    {
        if (!ctx || !symbol)
        {
            llvm::errs() << "crusty::jit: invalid arguments to crusty_jit_lookup\n";
            return nullptr;
        }

        // first find directly
        auto it = ctx->symbols.find(symbol);
        if (it != ctx->symbols.end())
        {
            return it->second;
        }

        auto addr = ctx->session->lookup(symbol);
        if (!addr)
        {
            llvm::consumeError(addr.takeError());
            llvm::errs() << "crusty::jit: lookup failed for symbol: " << symbol << "\n";
            return nullptr;
        }
        void *ptr = addr->toPtr<void *>();
        ctx->symbols[symbol] = ptr;
        return ptr;
    }

}