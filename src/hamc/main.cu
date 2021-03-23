#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}
void run_keygen()
{
    //TODO: initialize parity check matrix
    //TODO: initialize generator matrix

    /* Private key */
    // generate parity check matrix
    FILE *fp1, *fp2;
    fp1 = fopen("Private_Key.txt", "a");
    fprintf(fp1, "Private Key: Parity Check Matrix: \n");

    fclose(fp1);

    // create generator matrix


    /* Public Key */
    fp2 = fopen("Public_Key.txt", "a");
    fprintf(fp2, "Public Key: Generator Matrix: \n");


    fclose(fp2);
    //

}
void run_encryption(){}
void run_decryption(){}

void printHelp(){
    printf("HAMC - Hardware Accelerated Mceliece Cryptosystem\n");
    printf("Usage:\n");
    printf("./hamc <arguments>\n");
    printf("Available Arguments:\n");
    printf("\ta (REQUIRED)\n");
    printf("\t\t - action: 1)keygen 2)encrypt 3)decrypt\n");

    printf("\tn\n");
    printf("\t\t - \n");

    printf("\tp\n");
    printf("\t\t - \n");

    printf("\tw\n");
    printf("\t\t - \n");

    printf("\tt\n");
    printf("\t\t - \n");

    printf("\ti\n");
    printf("\t\t - input filename\n");

    printf("\to\n");
    printf("\t\t - output filename\n");
}

int main(int argc, char *argv[]) {
    wbArg_t args;
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    cudaEvent_t astartEvent, astopEvent;
    float aelapsedTime;

    int action, n, p, w, t;
    char *outputFileName, *inputFileName;

    int c;
    opterr = 0;
    while ((c = getopt (argc, argv, "a:npwtioh")) != -1)
        switch(c)
        {
            case 'n':
                n = *optarg;
                break;
            case 'p':
                p = *optarg;
                break;
            case 'w':
                w = *optarg;
                break;
            case 't':
                t = *optarg;
                break;
            case 'i':
                strcpy(inputFileName, (const char*)optarg);
                break;
            case 'o':
                strcpy(outputFileName, (const char*)optarg);
                break;
            case 'a':
                action = *optarg;
                break;
            case 'h':
                printHelp();
                return(1);
            default:
                abort();

        }

    int k = (n - 1) * p;


    // 1)kegen 2) encrypt 3)decrypt
    switch(action)
    {
        case 1: //keygen
            run_keygen();
            break;
        case 2: //encrypt
            run_encryption();
        case 3: //decrypt
            run_decryption();
        default:
            printf("Wrong action given to system.\n");
            printHelp();
            return(1);
    }
}
