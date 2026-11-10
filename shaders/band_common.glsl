// Shared by the band shaders: a pixel's BAND candidates are the disparities
// base[pixel] to base[pixel] + BAND - 1, each path cost one byte, four per
// 32-bit word, BAND / 4 words per pixel in row-major order.
#define WORDS (BAND / 4)
// The matching costs, census distances below 64, are packed 6 bits each,
// candidate k at bit 6k of a pixel's COST_WORDS words.
#define COST_WORDS ((BAND * 6 + 31) / 32)
uint cost_at(uint words[COST_WORDS], int k) {
  int bit = 6 * k, w = bit / 32, shift = bit % 32;
  uint low = words[w] >> shift;
  return (shift > 26 ? low | (words[w + 1] << (32 - shift)) : low) & 63u;
}
#include "features.glsl"
