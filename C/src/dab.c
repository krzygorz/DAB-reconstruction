#include "dab.h"

void get_carriers(float complex *sym_samples, float complex *carriers){
    fftwf_plan plan = fftwf_plan_dft_1d(N_FFT, sym_samples+N_GUARD, carriers, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}
