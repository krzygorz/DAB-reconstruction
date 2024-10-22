#include <complex.h>
#include <stdlib.h>
#include <fftw3.h>

typedef enum {
  DEMOD_STATUS_WAITING,
  DEMOD_STATUS_SYM,
  DEMOD_STATUS_END,
} dab_demod_status;

typedef struct {
    float complex *input_buf;
    unsigned int input_buf_pos;

    float complex *fft_input_buf;
    float complex *carriers_buf;
    fftwf_plan fftplan;
} dab_demod;

dab_demod *dab_demod_create();

void dab_demod_destroy(dab_demod *demod);

dab_demod_status dab_demod_push(dab_demod *demod, float complex x);
void dab_demod_carriers(dab_demod *demod, float complex *out);
