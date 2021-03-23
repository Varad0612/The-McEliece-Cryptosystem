#include <cuda_runtime.h>
#include<stdlib.h>
#include <wb.h>
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <unistd.h>

#include "../../hamc/RREFMatrix.cu"

#define uint uint16_t

#define TILE_WIDTH 16

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

void printHelp()
{
    printf("run this executable with the following flags\n");
    printf("\n");
    printf("\t-i <input file name>\n");
    printf("\t-o <output file name>\n");
    printf("\t-s <solution file name>\n");
}



int main(int argc, char *argv[])
{
    wbArg_t args;

    uint *hostA; // The A matrix
    uint *hostC; // The output C matrix
    uint *deviceA; // A matrix on device
    uint *deviceB; // B matrix on device (copy of A)
    uint *deviceC; // C matrix on device
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A

    char *inputFileName, *outputFileName, *solutionFileName;

    int c;
    opterr = 0;
    while ((c = getopt (argc, argv, "i:s:o:h")) != -1)
        switch(c)
        {
            case 'i':
                strcpy(inputFileName, (const char*)optarg);
                break;
            case 's':
                strcpy(solutionFileName, (const char*)optarg);
                break;
            case 'o':
                strcpy(outputFileName, (const char*)optarg);
                break;
            case 'h':
                printHelp();
                return 0;
            default:
                abort();

        }


    args = wbArg_read(argc, argv);


    /* allocate host data for matrix */
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (uint *)wbImport(inputFileName, &numARows, &numAColumns);
    int numBRows = numARows;    // number of rows in the matrix B
    int numBColumns = numAColumns; // number of columns in the matrix B
    int numCRows = numARows;    // number of rows in the matrix C
    int numCColumns = numAColumns; // number of columns in the matrix C
    hostC = (uint *)malloc(numCRows*numCColumns * sizeof(uint));
    wbTime_stop(Generic, "Importing data and creating memory on host");


    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);


    /* allocate the memory space on GPU */
    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(uint));
    cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(uint));
    cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(uint));
    wbTime_stop(GPU, "Allocating GPU memory.");


    dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    /* call CUDA kernel to perform computations */
    wbTime_start(Compute, "Performing CUDA computation for RREF");
    rref_kernel<<<dimGrid, dimBlock>>>(deviceA, deviceB, numARows, numAColumns, deviceC);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");



    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(uint), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    /* Free GPU Memory */
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostC);

    return 0;
}
