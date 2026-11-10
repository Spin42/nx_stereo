// Copyright 2016 Fixstars Corporation
// SPDX-License-Identifier: Apache-2.0
// libSGM's normalized directional recurrence over BLOCK disparities per lane,
// written so every candidate is formed once: the clamped own value, and the
// cheaper of the two neighbours plus P1. Lane-boundary neighbours arrive
// through relative shuffles; a missing neighbour contributes P2 - P1 so that
// adding P1 reproduces the P2 clamp exactly.
#define LANES (DISPARITIES / BLOCK)
// A workgroup scans LINES lines, LANES * LINES threads: 64 at 64 disparities,
// which Adreno GPUs run as narrower waves than 128.
#define LINES 8
#define PATH_WORD_TYPE uvec2
#define PATH_PACK 8
#define WORDS_PER_PIXEL (DISPARITIES / PATH_PACK)
// Byte volumes are stored in tiles of LINES consecutive rows per column, so a
// horizontal scan's workgroup touches one contiguous span per pixel step.
#define VOLUME_TILE LINES
#define TILE_WORDS (VOLUME_TILE * WORDS_PER_PIXEL)
#define TILE_ROW_WORDS (WIDTH * TILE_WORDS)
#define WORDS_PER_LANE (BLOCK / PATH_PACK)
int volume_word(int x, int y, uint lane) {
  return ((y / VOLUME_TILE) * WIDTH + x) * TILE_WORDS + (y % VOLUME_TILE) * WORDS_PER_PIXEL + int(lane) * WORDS_PER_LANE;
}
struct LaneGeometry {
  bool has_lower;
  bool has_upper;
};
LaneGeometry lane_geometry(uint lane) {
  LaneGeometry g;
  g.has_lower = lane > 0;
  g.has_upper = lane + 1 < LANES;
  return g;
}
uint group_min(uint value) {
  for (uint span = LANES / 2; span > 0; span /= 2) value = min(value, subgroupShuffleXor(value, span));
  return value;
}
uint group_or(uint value) {
  for (uint span = LANES / 2; span > 0; span /= 2) value |= subgroupShuffleXor(value, span);
  return value;
}
void update_path(uint costs[BLOCK], inout uint previous[BLOCK], inout uint last_min, LaneGeometry g) {
  uint a[BLOCK];
  for (int k = 0; k < BLOCK; ++k) a[k] = previous[k] - last_min;
  // Relative shuffles: cheaper than indexed ones on Adreno GPUs. Across a
  // lane group's edges they read another group, which has_lower and
  // has_upper replace below.
  uint lower = subgroupShuffleUp(a[BLOCK - 1], 1u);
  uint upper = subgroupShuffleDown(a[0], 1u);
  lower = g.has_lower ? lower : uint(P2 - P1);
  upper = g.has_upper ? upper : uint(P2 - P1);
  uint minimum = 255;
  for (int k = 0; k < BLOCK; ++k) {
    uint below = k > 0 ? a[k - 1] : lower;
    uint above = k + 1 < BLOCK ? a[k + 1] : upper;
    uint value = min(min(a[k], uint(P2)), min(below, above) + P1) + costs[k];
    previous[k] = value;
    minimum = min(minimum, value);
  }
  last_min = group_min(minimum);
}
PATH_WORD_TYPE pack_lane(uint values[BLOCK], int first) {
  uint lo = values[first] | (values[first + 1] << 8) | (values[first + 2] << 16) | (values[first + 3] << 24);
  uint hi = values[first + 4] | (values[first + 5] << 8) | (values[first + 6] << 16) | (values[first + 7] << 24);
  return uvec2(lo, hi);
}
void unpack_lane(PATH_WORD_TYPE word, out uint values[BLOCK], int first) {
  values[first + 4] = word.y & 255u;
  values[first + 5] = (word.y >> 8) & 255u;
  values[first + 6] = (word.y >> 16) & 255u;
  values[first + 7] = word.y >> 24;
  uint w = word.x;
  values[first] = w & 255u;
  values[first + 1] = (w >> 8) & 255u;
  values[first + 2] = (w >> 16) & 255u;
  values[first + 3] = w >> 24;
}
