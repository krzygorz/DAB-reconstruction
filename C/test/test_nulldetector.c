#include <stdio.h>

#include "dab.h"
#include "nulldetector.h"

#define BUF_LEN 4096

// quite tight bound, in practice this should be assumed to be higher
const int MAX_DEVIATION = 320;

int main(){
    char *path = "data/dab_short.iq";
    FILE *f_in = fopen(path,"r");
    if (f_in == NULL) {
        printf("could not open '%s'\n", path);
        exit(1);
    }
    float complex buf [BUF_LEN];
    dab_nulldetector *nulldetector = dab_nulldetector_create();
    nulldetector->threshold = 0.1;
    int n = 0;
    while(!feof(f_in)){
        int rc = fread(buf, sizeof(float complex), BUF_LEN, f_in);
        for(int i=0; i<rc; i++, n++){
            float complex sample = buf[i];
            if(dab_nulldetector_step(nulldetector, sample)){
                int mod = (n-T_NULL)%T_FRAME;
                if (mod > MAX_DEVIATION && mod < T_FRAME-MAX_DEVIATION){
                    printf("null detected at %d but it's too far from a null symbol\n", n);
                    exit(1);
                }
            }
        }
    }
    dab_nulldetector_destroy(nulldetector);
    fclose(f_in);
}
