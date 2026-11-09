// Shared by the band shaders: a pixel's BAND candidates are the disparities
// base[pixel] to base[pixel] + BAND - 1, each path cost one byte, four per
// 32-bit word, BAND / 4 words per pixel in row-major order.
#define WORDS (BAND / 4)
// The census features are tiled as census_pair.comp writes them.
int feature_index(int x, int y) {
  return ((y / FEATURE_TILE) * WIDTH + x) * FEATURE_TILE + y % FEATURE_TILE;
}
