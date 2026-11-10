// The census features' layout: a pixel's left and right features share one
// element, stored in tiles of FEATURE_TILE consecutive rows per column.
#define FEATURE_TILE 8
int feature_index(int x, int y) {
  return ((y / FEATURE_TILE) * WIDTH + x) * FEATURE_TILE + y % FEATURE_TILE;
}
