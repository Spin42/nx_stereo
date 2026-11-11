// SPDX-License-Identifier: Apache-2.0
// Eight costs of at most 127 fit without truncation (maximum 1016).
// Ten bits per candidate: each lane stores its low word, four lanes share
// one additional word containing their four high bytes.
#define AGG_WORDS (BAND * 10 / 32)
uvec4 unpack_aggregate(uint lo, uint hi, int shift) {
  return uvec4(lo & 1023u, (lo >> 10) & 1023u, (lo >> 20) & 1023u,
               (lo >> 30) | (((hi >> shift) & 255u) << 2));
}
