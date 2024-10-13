#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include "dab.h"
#include "syncsymbol.h"

#define BUF_LEN 4096

// the error between python and c versions is suspiciously high, about 1.5e-5
#define TIME_DOMAIN_TOLERANCE 1e-4
#define CARRIER_TOLERANCE 1e-10

void test_sync_symbol_time_domain() {
    float complex syncsymbol[T_SYMBOL];
    generate_sync_symbol_with_gi(syncsymbol);

    char *path = "data/sync_symbol.bin";
    FILE *f_in = fopen(path,"r");
    if (f_in == NULL) {
        printf("could not open '%s'\n", path);
        exit(1);
    }
    float complex buf [BUF_LEN];
    int n = 0;
    bool error_occurred = false;
    while(!feof(f_in)){
        int rc = fread(buf, sizeof(float complex), BUF_LEN, f_in);
        for(int i=0; i<rc; i++, n++){
            if(n > T_SYMBOL){
                printf("file %s longer than expected", path);
                exit(1);
            }
            float complex ref_sample = buf[i] * N_FFT; //numpy FFT uses different scaling than FFTW
            float complex gen_sample = syncsymbol[n];
            float error = cabs(ref_sample - gen_sample);
            if(error > TIME_DOMAIN_TOLERANCE){
                printf("sample %d doesn't match reference\n", n);
                printf("generated: %f + %fi\n", creal(gen_sample), cimag(gen_sample));
                printf("reference: %f + %fi\n", creal(ref_sample), cimag(ref_sample));
                printf("error magnitude: %f\n", error);
                error_occurred = true;
            }
        }
    }
    fclose(f_in);
    if(error_occurred){
        exit(1);
    }
}
void test_sync_symbol_carriers() {
    float complex carriers[N_FFT];
    generate_sync_symbol_carriers(carriers);

    char *path = "data/sync_carriers.bin";
    FILE *f_in = fopen(path,"r");
    if (f_in == NULL) {
        printf("could not open '%s'\n", path);
        exit(1);
    }
    float complex buf [BUF_LEN];
    int n = 0;
    bool error_occurred = false;
    while(!feof(f_in)){
        int rc = fread(buf, sizeof(float complex), BUF_LEN, f_in);
        for(int i=0; i<rc; i++, n++){
            if(n > N_FFT){
                printf("file %s longer than expected", path);
                exit(1);
            }
            float complex ref_sample = buf[i];
            float complex gen_sample = carriers[n];
            float error = cabs(ref_sample - gen_sample);
            if(error > CARRIER_TOLERANCE){
                printf("carrier %d doesn't match reference\n", n);
                printf("generated: %f + %fi\n", creal(gen_sample), cimag(gen_sample));
                printf("reference: %f + %fi\n", creal(ref_sample), cimag(ref_sample));
                printf("error magnitude: %f\n", error);
                error_occurred = true;
            }
        }
    }
    fclose(f_in);
    if(error_occurred){
        exit(1);
    }
}

int main(){
    test_sync_symbol_carriers();
    test_sync_symbol_time_domain();
    return 0;
}
