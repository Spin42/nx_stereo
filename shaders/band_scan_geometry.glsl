// SPDX-License-Identifier: Apache-2.0
// A wide image's diagonal scans fold the diagonals that leave through the
// image's side into the ones that enter through it, a thread scanning one
// then the other, so every thread covers HEIGHT pixels.
bool folded() { return WIDTH >= HEIGHT; }

int lines_of(int dx, int dy) {
  if (dy == 0) return HEIGHT;
  if (dx == 0 || folded()) return WIDTH;
  return WIDTH + HEIGHT - 1;
}

// The step at which a folded diagonal `line` starts its second diagonal, or
// -1.
int second_start(int dx, int dy, int line) {
  return dy != 0 && dx != 0 && folded() && WIDTH - line < HEIGHT ? WIDTH - line : -1;
}

// The pixel of `line` at step `i`, or -1 outside the image; column-major in a
// horizontal scan.
int pixel_at(int dx, int dy, int line, int i) {
  int x, y;
  if (dy == 0) {
    y = line;
    x = dx > 0 ? i : WIDTH - 1 - i;
  } else if (dx != 0 && folded()) {
    // Down and right in a frame mirrored to the direction: step i is row i.
    int second = second_start(dx, dy, line);
    int u = second < 0 || i < second ? line + i : i - second;
    x = dx > 0 ? u : WIDTH - 1 - u;
    y = dy > 0 ? i : HEIGHT - 1 - i;
  } else {
    y = dy > 0 ? i : HEIGHT - 1 - i;
    x = dx > 0 ? line - (HEIGHT - 1) + i : (dx < 0 ? line - i : line);
  }
  if (x < 0 || x >= WIDTH) return -1;
  return dy == 0 ? x * HEIGHT + y : y * WIDTH + x;
}
