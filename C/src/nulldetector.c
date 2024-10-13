#include "dab.h"
#include "nulldetector.h"

dab_nulldetector *dab_nulldetector_create(){
    dab_nulldetector *nd = malloc(sizeof(dab_nulldetector));
    nd->delay_null = wdelayf_create(T_NULL);
    nd->delay_frame = wdelayf_create(T_FRAME);
    nd->null_window_sum = 0;
    nd->frame_window_sum = 0;
    nd->n = 0;
    nd->threshold = 0.1;
    return nd;
}

void dab_nulldetector_destroy(dab_nulldetector *nd){
    wdelayf_destroy(nd->delay_null);
    wdelayf_destroy(nd->delay_frame);
    free(nd);
}

int dab_nulldetector_step(dab_nulldetector *nd, float complex x){
    /*
     * input -- |x|^2 --+-- movavg(T_NULL) ------------------+
     *                  |                                 compare ---->
     *                  +-- movavg(T_FRAME) ---- *thresh ----+
     *
     * movavg is implemented using a delay line and integrator.
     */
    float norm = creal(x)*creal(x) + cimag(x)*cimag(x);
    nd->null_window_sum += norm;
    nd->frame_window_sum += norm;
    wdelayf_push(nd->delay_null, norm);
    wdelayf_push(nd->delay_frame, norm);
    nd->n++;

    float old_norm_null;
    wdelayf_read(nd->delay_null, &old_norm_null);
    nd->null_window_sum -= old_norm_null;

    float old_norm_frame;
    wdelayf_read(nd->delay_frame, &old_norm_frame);
    nd->frame_window_sum -= old_norm_frame;

    if (nd->n < T_FRAME){
        return 0;
    }
    if (nd->null_window_sum/T_NULL < nd->frame_window_sum/T_FRAME * nd->threshold) {
        return 1;
    }
    return 0;
}
