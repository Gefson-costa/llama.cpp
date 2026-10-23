#pragma once

// The persistent repack cache (implemented in repack.cpp), shared by the CPU_REPACK and
// CPU_KLEIDIAI buffers. Each buffer type has its own slot and its own file, so a load
// that puts weights in both caches both.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

// Where repack.cpp builds it (mmap, pread, fdatasync); elsewhere nothing below exists.
#if defined(__linux__) || defined(__APPLE__)
#define GGML_REPACK_CACHE_AVAILABLE 1

namespace ggml::cpu::repack_cache {

enum : int { SLOT_REPACK = 0, SLOT_KLEIDIAI = 1, N_SLOTS = 2 };

// Backs a buffer of data_size bytes with the cache file at path; nullptr → plain allocation.
// salt separates layouts: a file built under another salt is a miss.
uint8_t * attach(int slot, const char * path, size_t data_size, uint64_t salt);
void      register_buffer(int slot, ggml_backend_buffer_t buffer);
void      detach_unused(int slot);
void      buffer_free_buffer(ggml_backend_buffer_t buffer);
// True → the mapped bytes are this tensor's verified repack, skip repacking.
bool      on_set_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                        const void * data, size_t size);
// From compute on a cached buffer: the load has finished, the file can be made COMPLETE.
void      maybe_finalize_async(const struct ggml_tensor * op);

} // namespace ggml::cpu::repack_cache
#endif
