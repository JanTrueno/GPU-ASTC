// LZX decompressor as used by XNA/FNA .xnb files (frames of up to 32 KB, 64 KB window).
// Based on the LZX format as implemented by libmspack (LGPL-free reimplementation from the public
// format description; no code copied).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpuastc {

// Decompresses an XNB LZX payload: a sequence of chunks, each
//   [0xFF frame_size(2, BE)] block_size(2, BE) data[block_size]
// `in` points at the first chunk, `out_size` is the expected decompressed size.
// Returns false on corrupt input.
bool lzx_decompress_xnb(const uint8_t* in, size_t in_size, size_t out_size, std::vector<uint8_t>& out);

} // namespace gpuastc
