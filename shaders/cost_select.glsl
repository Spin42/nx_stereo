// Copyright 2016 Fixstars Corporation
// SPDX-License-Identifier: Apache-2.0
// Selects the winner of a pixel's summed paths, with uniqueness, subpixel and
// winner-key outputs.
struct SelectGeometry {
  int x;
  uint lane_disparity;
  // Candidates right of the image (d > x) never win: they carry a padding
  // term that exceeds every genuine sum, which also excludes them from the
  // second-best search. Winner keys and minima are read from unpadded lanes.
  uint pad[BLOCK];
};
SelectGeometry select_geometry(int x, uint lane) {
  SelectGeometry s;
  s.x = x;
  s.lane_disparity = lane * BLOCK;
  for (int k = 0; k < BLOCK; ++k) s.pad[k] = int(lane) * BLOCK + k <= x ? 0u : 0x8000u;
  return s;
}
uint pick(uint values[BLOCK], uint index) {
  uint t[BLOCK];
  for (int i = 0; i < BLOCK; ++i) t[i] = values[i];
  bool high4 = (index & 4u) != 0u;
  for (int i = 0; i < 4; ++i) t[i] = high4 ? t[i + 4] : t[i];
  bool high2 = (index & 2u) != 0u;
  for (int i = 0; i < 2; ++i) t[i] = high2 ? t[i + 2] : t[i];
  return (index & 1u) != 0u ? t[1] : t[0];
}
void select_sum(uint vertical[BLOCK], int pixel, uint stored[BLOCK], SelectGeometry s) {
  uint values[BLOCK];
  for (int k = 0; k < BLOCK; ++k) values[k] = vertical[k] + stored[k] + s.pad[k];
  uint best = 0xffffffffu;
  for (int k = 0; k < BLOCK; ++k) best = min(best, (values[k] << 16) | (s.lane_disparity + uint(k)));
  best = group_min(best);
  int winner = int(best & 65535u);
  uint minimum = best >> 16;
  // Entries within one disparity of the winner are excluded from second place.
  uint second = 0xffffffffu;
  uint near = s.lane_disparity + 1u - uint(winner);
  for (int k = 0; k < BLOCK; ++k) second = min(second, near + uint(k) <= 2u ? 0xffffffffu : values[k]);
  second = group_min(second);
  int lower_d = max(winner - 1, 0), upper_d = min(winner + 1, DISPARITIES - 1);
  // The lanes holding the winner's neighbours give their sums, below 2^16,
  // in the two halves of a word, and the others 0: an OR across the group
  // gathers both with fixed-distance shuffles, which are much cheaper on
  // Adreno GPUs than a shuffle from a lane that differs per pixel.
  uint lane = s.lane_disparity / BLOCK;
  uint neighbours = (uint(lower_d) / BLOCK == lane ? pick(values, uint(lower_d) % BLOCK) << 16 : 0u) |
                    (uint(upper_d) / BLOCK == lane ? pick(values, uint(upper_d) % BLOCK) : 0u);
  neighbours = group_or(neighbours);
  uint lower = neighbours >> 16, upper = neighbours & 65535u;
  bool accepted = float(second) > float(minimum) * UNIQUENESS;
  float lo = float(lower), hi = float(upper);
  float denominator = lo + hi - 2.0 * float(minimum);
  float correction = clamp((lo - hi) / (2.0 * max(denominator, 1.0)), -0.5, 0.5);
  correction = mix(0.0, correction, denominator > 0.0);
  correction = mix(0.0, correction, winner > 0);
  correction = mix(0.0, correction, winner < min(DISPARITIES - 1, s.x));
  float result = float(winner) + correction;
  if (s.lane_disparity == 0u) {
    left_disparity[pixel] = mix(-1.0, result, accepted);
    winners[pixel] = mix(0xffffffffu, best, accepted);
  }
}
