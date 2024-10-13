#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <liquid/liquid.h>
#include "dab.h"
#include "syncsymbol.h"

// Table 23: Relation between the indices i, k' and n and the carrier index k for transmission mode I
static int table_23[48][4] = {
// k' = k_min, column omitted
//  k_min  k_max i  n
    {-768, -737, 0, 1},
    {-736, -705, 1, 2},
    {-704, -673, 2, 0},
    {-672, -641, 3, 1},
    {-640, -609, 0, 3},
    {-608, -577, 1, 2},
    {-576, -545, 2, 2},
    {-544, -513, 3, 3},
    {-512, -481, 0, 2},
    {-480, -449, 1, 1},
    {-448, -417, 2, 2},
    {-416, -385, 3, 3},
    {-384, -353, 0, 1},
    {-352, -321, 1, 2},
    {-320, -289, 2, 3},
    {-288, -257, 3, 3},
    {-256, -225, 0, 2},
    {-224, -193, 1, 2},
    {-192, -161, 2, 2},
    {-160, -129, 3, 1},
    {-128,  -97, 0, 1},
    { -96,  -65, 1, 3},
    { -64,  -33, 2, 1},
    { -32,   -1, 3, 2},
    {   1,   32, 0, 3},
    {  33,   64, 3, 1},
    {  65,   96, 2, 1},
    {  97,  128, 1, 1},
    { 129,  160, 0, 2},
    { 161,  192, 3, 2},
    { 193,  224, 2, 1},
    { 225,  256, 1, 0},
    { 257,  288, 0, 2},
    { 289,  320, 3, 2},
    { 321,  352, 2, 3},
    { 353,  384, 1, 3},
    { 385,  416, 0, 0},
    { 417,  448, 3, 2},
    { 449,  480, 2, 1},
    { 481,  512, 1, 3},
    { 513,  544, 0, 3},
    { 545,  576, 3, 3},
    { 577,  608, 2, 3},
    { 609,  640, 1, 0},
    { 641,  672, 0, 3},
    { 673,  704, 3, 0},
    { 705,  736, 2, 1},
    { 737,  768, 1, 1},
};

static int table_24[4][32] = {
// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15      16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
  {0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1,     0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1},
  {0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0,     0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0},
  {0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3,     0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3},
  {0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2,     0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2}
};

float complex qpsk_mod(int symbol){
    switch(symbol){
        case 0:
            return 1;
        case 1:
            return I;
        case 2:
            return -1;
        case 3:
            return -I;
        default:
            printf("invalid symbol value %d\n", symbol);
            exit(1);
    }
}

void generate_sync_symbol_carriers(float complex *out){
    memset(out, 0, N_FFT*sizeof(float complex));
    for(int k=-N_CARRIERS/2; k <= N_CARRIERS/2; k++){
        // convert subcarrier number to fft index
        // (negative numbers wrap around)
        int fft_idx = k >=0 ? k : N_FFT+k;
        if(k == 0) {
            out[fft_idx] = 0;
            continue;
        }
        int table_idx;
        if (k < 0)
            table_idx = (k+N_CARRIERS/2)/32;
        else
            table_idx = (k-1+N_CARRIERS/2)/32;
        assert(k >= table_23[table_idx][0] && k <= table_23[table_idx][1]);

        int k1 = table_23[table_idx][0];
        int i = table_23[table_idx][2];
        int n = table_23[table_idx][3];
        int phase = (table_24[i][k-k1] + n) % 4;
        out[fft_idx] = qpsk_mod(phase);
    }
}

void generate_sync_symbol_with_gi(float complex *out){
    float complex *fft_in  = fft_malloc(N_FFT*sizeof(float complex));
    float complex *fft_out = fft_malloc(N_FFT*sizeof(float complex));
    generate_sync_symbol_carriers(fft_in);

    fftplan plan = fft_create_plan(N_FFT, fft_in, fft_out, LIQUID_FFT_BACKWARD, 0);
    fft_execute(plan);
    fft_destroy_plan(plan);

    // copy guard interval
    memcpy(out, fft_out + N_FFT - T_GUARD, T_GUARD*sizeof(float complex));
    // copy the whole symbol, after the guard interval
    memcpy(out+T_GUARD, fft_out, N_FFT*sizeof(float complex));
    fft_free(fft_in);
    fft_free(fft_out);
}
// ref_sync_carriers = np.zeros(N_fft, dtype="complex128")
// for k in range(-N_carriers//2,N_carriers//2+1):
//    idx = k if k >=0 else N_fft+k
//    if k == 0:
//        ref_sync_carriers[idx] = 0
//        continue
//    k1,_, i, n = table_23[(table_23[:,0] <= k) & (k <= table_23[:,1])][0]
//    phase = np.pi/2 * (h_table[i, k-k1] + n)
//    ref_sync_carriers[idx] = np.exp(1.0j * phase)
