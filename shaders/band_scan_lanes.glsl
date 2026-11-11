// Copyright 2016 Fixstars Corporation
// SPDX-License-Identifier: Apache-2.0
// Four candidates per lane. The including shader defines store_path.
uint pixel_min(uint value) {
  for (uint span = WORDS / 2; span > 0; span /= 2)
    value = min(value, subgroupShuffleXor(value, span));
  return value;
}

// An out-of-band word is P2 in every byte. Shuffle sources stay in this pixel.
uint word_at(uint packed, int word) {
  uint first = gl_SubgroupInvocationID & ~uint(WORDS - 1);
  uint shuffled = subgroupShuffle(packed, first + uint(clamp(word, 0, WORDS - 1)));
  return word >= 0 && word < WORDS ? shuffled : uint(P2) * 0x01010101u;
}

// Four 6-bit costs span at most two storage words.
uint costs_at(int pixel, int lane) {
  if (pixel < 0) return 0u;
  int bit = 24 * lane, word = bit / 32, shift = bit % 32;
  uint lower = costs[pixel * PIXEL_WORDS + word];
  uint upper = word + 1 < COST_WORDS ? costs[pixel * PIXEL_WORDS + word + 1] : 0u;
  return (lower >> shift) | ((upper << (24 - shift)) << 8);
}

uint advance(uint previous, int delta, uint costs, int lane) {
  uint minimum = min(min(previous & 255u, (previous >> 8) & 255u),
                     min((previous >> 16) & 255u, previous >> 24));
  minimum = pixel_min(minimum);
  // Every byte is at least the minimum, so subtraction cannot borrow.
  uint normalized = previous - minimum * 0x01010101u;
#if BAND == 32
  // An unchanged band needs only its two neighbouring packed words.
  if (delta == 0) {
    uint before = subgroupShuffleUp(normalized, 1u);
    uint after = subgroupShuffleDown(normalized, 1u);
    if (lane == 0) before = uint(P2) * 0x01010101u;
    if (lane == WORDS - 1) after = uint(P2) * 0x01010101u;
    uint lower = (normalized << 8) | (before >> 24);
    uint upper = (normalized >> 8) | (after << 24);
    uint packed = 0u;
    for (int c = 0; c < 4; ++c) {
      uint own = min((normalized >> (8 * c)) & 255u, uint(P2));
      uint neighbour = min((lower >> (8 * c)) & 255u, (upper >> (8 * c)) & 255u) + uint(P1);
      packed |= (min(own, neighbour) + ((costs >> (6 * c)) & 63u)) << (8 * c);
    }
    return packed;
  }
#endif
  int at = 4 * lane + delta - 1;
  int word = at >> 2, shift = 8 * (at & 3);
  uint first = word_at(normalized, word);
  uint second = word_at(normalized, word + 1);
  uint third = word_at(normalized, word + 2);
  uint lower = (first >> shift) | ((second << (24 - shift)) << 8);
  uint upper = (second >> shift) | ((third << (24 - shift)) << 8);
  uint window[6];
  for (int j = 0; j < 6; ++j)
    window[j] = ((j < 4 ? lower : upper) >> (8 * (j % 4))) & 255u;
  uint packed = 0u;
  for (int c = 0; c < 4; ++c) {
    uint own = min(window[c + 1], uint(P2));
    uint neighbour = min(window[c], window[c + 2]) + uint(P1);
    uint value = min(own, neighbour) + ((costs >> (6 * c)) & 63u);
    packed |= value << (8 * c);
  }
  return packed;
}

void scan(int dx, int dy, int line, int lane, bool add) {
  uint previous = 0u;
  int previous_base = -1;
  int pixel = pixel_at(dx, dy, line, 0);
  int b = pixel >= 0 ? max(base_of(pixel), 0) : 0;
  uint words = costs_at(pixel, lane);
  int steps = dy == 0 ? WIDTH : HEIGHT;
  int restart = second_start(dx, dy, line);
  for (int i = 0; i < steps; ++i) {
    int next_pixel = i + 1 < steps ? pixel_at(dx, dy, line, i + 1) : -1;
    int next_b = next_pixel >= 0 ? max(base_of(next_pixel), 0) : 0;
    uint next_words = costs_at(next_pixel, lane);
    if (i == restart) {
      previous = 0u;
      previous_base = -1;
    }
    if (pixel >= 0) {
      previous = advance(previous, previous_base >= 0 ? b - previous_base : 0, words, lane);
      previous_base = b;
      store_path(pixel, lane, previous, add);
    }
    pixel = next_pixel;
    b = next_b;
    words = next_words;
  }
}
