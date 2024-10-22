// vi: set shiftwidth=4
#include <stdio.h>

#include "dab.h"
#include "nulldetector.h"
#include "syncsymbol.h"
#include "demod.h"

#define BUF_LEN 4096
#define NULL_MARGIN 500

int main(){
    char *path = "data/dab_short.iq";
    // char *path = "data/DAB+_20240712_105527Z_2048000_176640000_float32_iq.raw";
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

    float freq_offset_hz = 0;//100;
    channel_cccf_add_carrier_offset(channel, freq_offset_hz / DAB_FS, 0);

    float max_freq_offset_hz = 50e3;

    dab_nulldetector *nulldetector = dab_nulldetector_create();
    nulldetector->threshold = 0.1;
    qdetector_cccf detector = qdetector_cccf_create(syncsymbol, T_SYMBOL);
    qdetector_cccf_set_range(detector, max_freq_offset_hz / DAB_FS);
    unsigned int det_buf_len = qdetector_cccf_get_buf_len(detector);
    unsigned int sync_search_counter = 0;
    unsigned int sync_search_limit = NULL_MARGIN+T_SYMBOL+det_buf_len;
    unsigned int last_frame = 0;

    dab_demod *demod = dab_demod_create();

    float complex buf [BUF_LEN];
    unsigned int n = 0;
    int locked = 0;
    unsigned frame_count = 0;
    while(!feof(f_in)){
        int rc = fread(buf, sizeof(float complex), BUF_LEN, f_in);
        channel_cccf_execute_block(channel, buf, BUF_LEN, buf);
        for(int i=0; i<rc; i++,n++){
            float complex sample = buf[i];
            if(!locked){
                int found_null = dab_nulldetector_step(nulldetector, sample);
                if(sync_search_counter == 0 && found_null){
                    sync_search_counter = 1;
                    printf("found null %d (%f)\n", n, ((float)n)/T_FRAME);
                    printf("*****************************\n");
                    qdetector_cccf_reset(detector);
                }

                if (sync_search_counter > 0){
                    if(sync_search_counter > sync_search_limit){
                        sync_search_counter = 0;
                        printf("found null symbol but no sync symbol\n");
                        continue;
                    }
                    sync_search_counter++;
                    float complex *detector_buf = qdetector_cccf_execute(detector,sample);
                    if (detector_buf){
                        if(frame_count > 2){
                            locked = 1;
                            for(int j=0; j<det_buf_len; j++){
                                if(dab_demod_push(demod, detector_buf[j]) == DEMOD_STATUS_SYM){
                                  printf("symbol\n");
                                    float complex carriers_buf[N_CARRIERS];
                                    dab_demod_carriers(demod, carriers_buf);
                                    fwrite(carriers_buf, sizeof(float complex), N_CARRIERS, f_dump);
                                    exit(0);
                                }
                            }
                        }
                        printf("detected frame at %d\n", n-det_buf_len);
                        // float rxy       = qdetector_cccf_get_rxy(detector);
                        // float tau_hat   = qdetector_cccf_get_tau(detector);
                        float dphi_hat  = qdetector_cccf_get_dphi(detector);

                        printf("freq offset %.1f Hz\n", dphi_hat * DAB_FS);
                        printf("offset from last frame: %d (difference from T_FRAME: %d)\n", (n-det_buf_len) - last_frame, T_FRAME-((n-det_buf_len) - last_frame));
                        last_frame = n-det_buf_len;
                        sync_search_counter = 0;
                        frame_count++;
                    }
                }
            } else { // if locked
                if(dab_demod_push(demod, sample) == DEMOD_STATUS_SYM){
                    float complex carriers_buf[N_CARRIERS];
                    dab_demod_carriers(demod, carriers_buf);
                    fwrite(carriers_buf, sizeof(float complex), N_CARRIERS, f_dump);
                    exit(0);
                }
            }
        }
    }

    dab_nulldetector_destroy(nulldetector);
    qdetector_cccf_destroy(detector);
    channel_cccf_destroy(channel);
    fclose(f_dump);
    fclose(f_in);
    return 0;
}

