// vi: set shiftwidth=4
#include <stdio.h>

#include "dab.h"
#include "nulldetector.h"
#include "syncsymbol.h"

#define BUF_LEN 4096

int main(){
    char *path = "data/dab_short.iq";
    char *dump_path = "dump.txt";
  
    FILE *f_in = fopen(path,"r");
    if (f_in == NULL) {
        fprintf(stderr,"error: could not open '%s'\n", path);
        exit(1);
    }
    FILE *f_dump = fopen(dump_path, "w");
    if (f_dump == NULL) {
        fprintf(stderr,"error: could not open '%s'\n", dump_path);
        exit(1);
    }
    
    float complex syncsymbol[T_SYMBOL];
    generate_sync_symbol_with_gi(syncsymbol);

    channel_cccf channel = channel_cccf_create();
    float noise_floor   = -60.0f;   // noise floor [dB]
    float SNRdB         =  30.0f;   // signal-to-noise ratio [dB]
    channel_cccf_add_awgn(channel, noise_floor, SNRdB);

    float freq_offset_hz = 100;
    channel_cccf_add_carrier_offset(channel, freq_offset_hz / DAB_FS, 0);

    dab_nulldetector *nulldetector = dab_nulldetector_create();
    nulldetector->threshold = 0.1;
    qdetector_cccf detector = qdetector_cccf_create(syncsymbol, T_SYMBOL);
    unsigned int det_buf_len = qdetector_cccf_get_buf_len(detector);

    float complex buf [BUF_LEN];
    unsigned int n = 0;
    unsigned int sync_search_counter = 0;
    unsigned int sync_search_limit = 500+det_buf_len;
    while(!feof(f_in)){
        int rc = fread(buf, sizeof(float complex), BUF_LEN, f_in);
        channel_cccf_execute_block(channel, buf, BUF_LEN, buf);
        for(int i=0; i<rc; i++,n++){
            float complex sample = buf[i];
            if(sync_search_counter == 0 && dab_nulldetector_step(nulldetector, sample)){
                sync_search_counter = 1;
                printf("found null %d\n", n);
            }

            if (sync_search_counter > 0){
                if(sync_search_counter > sync_search_limit){
                    // sync_search_counter = 0;
                    // printf("found null symbol but no sync symbol\n", sync_search_counter);
                    // continue;
                }
                sync_search_counter++;
                // printf("sync search %d\n", sync_search_counter);
                if(qdetector_cccf_execute(detector,sample)){

                    printf("detected frame at %d\n", n-det_buf_len);
                    float rxy       = qdetector_cccf_get_rxy(detector);
                    float tau_hat   = qdetector_cccf_get_tau(detector);
                    float gamma_hat = qdetector_cccf_get_gamma(detector);
                    float dphi_hat  = qdetector_cccf_get_dphi(detector);
                    float phi_hat   = qdetector_cccf_get_phi(detector);
                    printf("rxy=%f\ntau_hat=%f\ngamma_hat=%f\ndphi_hat=%f\nphi_hat=%f\n", rxy, tau_hat, gamma_hat, dphi_hat, phi_hat);
                }
            }
        }
    }
    // dab_nulldetector_destroy(nulldetector);

    fclose(f_dump);
    fclose(f_in);
    return 0;
}

