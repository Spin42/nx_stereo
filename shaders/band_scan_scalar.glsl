// A subgroup smaller than WORDS cannot exchange the complete candidate band.
void store_scalar_path(int pixel, uint previous[BAND], bool add);

uint pack_scalar(uint previous[BAND], int word) {
  return previous[4 * word] | (previous[4 * word + 1] << 8) |
         (previous[4 * word + 2] << 16) | (previous[4 * word + 3] << 24);
}

void scan_scalar(int dx, int dy, int line, bool add) {
  uint previous[BAND];
  for (int k = 0; k < BAND; ++k) previous[k] = 0u;
  int previous_base = -1;
  int steps = dy == 0 ? WIDTH : HEIGHT;
  int restart = second_start(dx, dy, line);

  for (int step = 0; step < steps; ++step) {
    if (step == restart) {
      for (int k = 0; k < BAND; ++k) previous[k] = 0u;
      previous_base = -1;
    }
    int pixel = pixel_at(dx, dy, line, step);
    if (pixel < 0) continue;
    int b = max(base_of(pixel), 0);
    int delta = previous_base >= 0 ? b - previous_base : 0;
    uint minimum = previous[0];
    for (int k = 1; k < BAND; ++k) minimum = min(minimum, previous[k]);
    uint words[COST_WORDS];
    for (int word = 0; word < COST_WORDS; ++word)
      words[word] = costs[pixel * PIXEL_WORDS + word];

    uint next[BAND];
    for (int k = 0; k < BAND; ++k) {
      int at = k + delta;
      uint own = at >= 0 && at < BAND ? previous[at] - minimum : uint(P2);
      uint lower = at > 0 && at <= BAND ? previous[at - 1] - minimum : uint(P2);
      uint upper = at >= -1 && at < BAND - 1 ? previous[at + 1] - minimum : uint(P2);
      next[k] = cost_at(words, k) + min(min(own, uint(P2)), min(lower, upper) + uint(P1));
    }
    for (int k = 0; k < BAND; ++k) previous[k] = next[k];
    previous_base = b;
    store_scalar_path(pixel, previous, add);
  }
}
