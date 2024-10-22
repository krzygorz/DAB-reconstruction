#include <assert.h>
#include <string.h>
#include "dab.h"
#include "demod.h"

dab_demod *dab_demod_create(){
    dab_demod *demod = malloc(sizeof(dab_demod));

    demod->input_buf = malloc(sizeof(float complex) * T_SYMBOL);
    demod->input_buf_pos = 0;

    // fft_input_buf could probably be replaced with input_buf+T_GUARD
    demod->fft_input_buf = fftwf_malloc(sizeof(float complex) * N_FFT);
    demod->carriers_buf  = fftwf_malloc(sizeof(float complex) * N_FFT);
    demod->fftplan = fftwf_plan_dft_1d(
        N_FFT,
        demod->fft_input_buf,
        demod->carriers_buf,
        FFTW_FORWARD,
        FFTW_MEASURE
    );
    return demod;
}

void dab_demod_destroy(dab_demod *demod){
    fftwf_destroy_plan(demod->fftplan);
    fftwf_free(demod->fft_input_buf);
    fftwf_free(demod->carriers_buf);
    free(demod->input_buf);
    free(demod);
}

dab_demod_status dab_demod_push(dab_demod *demod, float complex x){
    assert(demod->input_buf_pos <= T_SYMBOL);
    demod->input_buf[demod->input_buf_pos] = x;
    demod->input_buf_pos++;
    if(demod->input_buf_pos == T_SYMBOL){
        return DEMOD_STATUS_SYM;
    }
    return DEMOD_STATUS_WAITING;
}

void dab_demod_carriers(dab_demod *demod, float complex *out){
    // check if the buffer is full
    assert(demod->input_buf_pos == T_SYMBOL);

    // copy the symbol without GI and take FFT
    memcpy(demod->fft_input_buf, demod->input_buf+T_GUARD, sizeof(float complex) * N_FFT);
    fftwf_execute(demod->fftplan);
    
    // copy the last N_CARRIERS/2 FFT bins into the first (negative) half of the carriers array
    memcpy(out, demod->carriers_buf+N_FFT-N_CARRIERS/2, sizeof(float complex) * N_CARRIERS/2);

    // copy the first N_CARRIERS/2 FFT bins (EXCLUDING THE FIRST f=0 BIN!!!) 
    // into the second (positive) half of the carriers array
    memcpy(out+N_CARRIERS/2, demod->carriers_buf+1, sizeof(float complex) * N_CARRIERS/2);
}
