// Duration of a null symbol in samples
#define T_NULL 2656

// Duration of a (non-null) symbol in samples, including the guard interval.
// This is equal to N_FFT + T_GUARD
#define T_SYMBOL 2552

// duration of guard interval
#define T_GUARD 504

// duration of one frame in samples
#define T_FRAME 196608

// number of non-zero carriers
#define N_CARRIERS 1536 

// number of samples in a symbol WITHOUT the guard interval
#define N_FFT 2048

// sampling frequency for DAB signal processing
#define DAB_FS 2048000
