typedef enum {
    DAB_CTRL_NULLSEARCH,
    DAB_CTRL_SYNCSEARCH,
    DAB_CTRL_LOCKED,
} dab_reconstructor_state;

typedef struct {
    dab_reconstructor_state state;
} dab_reconstructor;


