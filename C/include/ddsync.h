typedef struct {
    float freq_offset;
    float time_offset;
} ddsync;

ddsync *ddsync_create(float freq_offset, float time_offset);

// run a sample through the sync
void ddsync_execute(ddsync *s, float complex x);

// destroy a null detector
void ddsync_destroy(ddsync *s);

// Update the params
int ddsync_step(ddsync *nd, float complex ref);
