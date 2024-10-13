#include <liquid/liquid.h>

typedef struct {
    wdelayf delay_null;
    wdelayf delay_frame;
    float null_window_sum;
    float frame_window_sum;
    unsigned int n;
    float threshold;
} dab_nulldetector;

// create a null detector
// The null detector computes moving average of |x|^2 and detects dips
dab_nulldetector *dab_nulldetector_create();

// destroy a null detector
void dab_nulldetector_destroy(dab_nulldetector *nd);

// Run one sample through a null detector
// Returns 1 if a null symbol was detected, 0 otherwise
int dab_nulldetector_step(dab_nulldetector *nd, float complex x);
