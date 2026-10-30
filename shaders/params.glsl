// The SGM penalties and the uniqueness ratio, an input at binding 0, so
// changing them needs no new pipeline.
layout(set = 0, binding = 0) readonly buffer Params { float params[3]; };
int P1;
int P2;
float UNIQUENESS;
void load_params() {
  P1 = int(params[0]);
  P2 = int(params[1]);
  UNIQUENESS = params[2];
}
