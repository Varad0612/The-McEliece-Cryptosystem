#include <stdio.h>

#define mat_element(mat, row_idx, col_idx, cols) \
    mat[row_idx * cols + col_idx]

#define ushort unsigned short

//TODO: check that this function works on gpu
//Add row1 to row2 of matrix A
__device__ void add_rows_kernel(uint16_t *data, uint16_t row1, uint16_t row2,
        uint16_t rows, uint16_t cols)
{

    if(row1 < 0 || row1 >= rows || row2 < 0 || row2 >= rows) {
        printf("add_rows: Matrix index out of range\n");
    }
    for(int i = 0; i < cols; i++) {
        mat_element(data, row2, i, cols) = \
            (mat_element(data, row1, i, cols) ^ mat_element(data, row2, i, cols));
    }
}

//TODO: check that this function works on gpu
// swap two rows of matrix A
__device__ void swap_kernel(uint16_t *data, ushort row1, ushort row2,
        ushort rows, ushort cols)
{
    if(row1 < 0 || row1 >= rows || row2 < 0 || row2 >= rows) {
      printf("Swap: Matrix index out of range\n");
    }
    int temp;
    for(int i = 0; i < cols; i++) {
        temp = mat_element(data, row1, i, cols);
        mat_element(data, row1, i, cols) = mat_element(data, row2, i, cols);
        mat_element(data, row2, i, cols) = temp;
    }
}

/**
 * @brief Reduced Row Echelon Form kernel
 *
 * Converts matrix to RREF
 *
 * @param data - input data as uint16_t pointer
 * @param rows - number of rows in matrix
 * @param cols - number of columns in matrix
 * @param out - output array to write data to as an uint16_t pointer
 * @return void
 */
__global__ void rref_kernel(uint16_t *data, uint16_t *temp, uint16_t rows, uint16_t cols,
        uint16_t *out)
{
    int lead = 0;
    int row_count = rows;
    int col_count = cols;
    //bin_matrix temp = mat_init(row_count, col_count);
    //temp = mat_copy(A);

    //copy host kernel to gpu



    __syncthreads();
    int r = 0;
    while(r < row_count) {
        if(mat_element(out, r, r, cols) == 0) {
            int i;
            for(i = r + 1; i < rows; i++) {
                if(mat_element(out, i, r, cols) == 1) {
                    swap_kernel(out, r, i, rows, cols);
                    break;
                }
            }
            if(i == row_count) {
                printf("Matrix cannot be transformed into row echlon form...");
            }
          }
        else {
            for(int i = 0; i < row_count; i++) {
                if(mat_element(data, i, r, cols) == 1 && i != r) {
                    add_rows_kernel(out, r, i, rows, cols);
                }
            }
            r++;
        }
    }
}
