#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "wb.h"

static char *base_dir;
#define ushort unsigned short

static ushort *generate_data(int height, int width) {
  ushort *data = (ushort *)malloc(sizeof(ushort) * width * height);
  int i;
  for (i = 0; i < width * height; i++) {
    data[i] = ((ushort)(rand() % 20) - 5) / 5.0f;
  }
  return data;
}

static void write_data(char *file_name, ushort *data, int height,
                       int width) {
  int ii, jj;
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d %d\n", height, width);
  for (ii = 0; ii < height; ii++) {
    for (jj = 0; jj < width; jj++) {
      fprintf(handle, "%.2hi", *data++);
      if (jj != width - 1) {
        fprintf(handle, " ");
      }
    }
    if (ii != height - 1) {
      fprintf(handle, "\n");
    }
  }
  fflush(handle);
  fclose(handle);
}


#define mat_element(mat, cols, row_idx, col_idx) \
  mat[(row_idx * cols) + col_idx]


static void swap(ushort *A, int row1, int row2, int rows, int cols)
{
    int temp;
    for(int i = 0; i < cols; i++)
    {
        temp = mat_element(A, cols, row1, i);
        mat_element(A, cols, row1, i) = mat_element(A, cols, row2, i);
        mat_element(A, cols, row2, i) = temp;
    }
}



static void add_rows(ushort *A, int row1, int row2,
        int rows, int cols)
{
  for(int i = 0; i < cols; i++)
  {
    mat_element(A, cols, row2, i) = \
        (mat_element(A, cols, row1, i) ^ mat_element(A, cols, row2, i));
  }
}



//Function to obtain the row reduced echlon form of a matrix A
static void matrix_rref(ushort *A, ushort *B, int rows, int cols)
{
  ushort *temp = (ushort *)calloc(sizeof(ushort), rows * cols);
  for (int i = 0; i < rows*cols;i++)
      temp[i] = A[i];


  int r = 0;
  while(r < rows) {
    if(mat_element(temp, cols, r, r) == 0) {
      int i;
      for(i = r + 1; i < rows; i++) {
        if(mat_element(temp, cols, i, r) == 1) {
          swap(temp, r, i, rows, cols);
          break;
        }
      }
      if(i == rows) {
          printf("Matix cannot be transformed into row echlon form...");
          exit(1);
      }
    }
    else {
      for(int i = 0; i < rows; i++) {
        if(mat_element(temp, cols, i, r) == 1 && i != r) {
          add_rows(temp, r, i, rows, cols);
        }
      }
      r++;
    }
  }
  return;
}

static void create_dataset(int datasetNum, int numARows, int numACols) {
    int numBRows = numACols;
    int numBCols = numACols;

    const char *dir_name =
        wbDirectory_create(wbPath_join(base_dir, datasetNum));

    char *input0_file_name = wbPath_join(dir_name, "input.raw");
    char *output_file_name = wbPath_join(dir_name, "output.raw");

    ushort *input_data = generate_data(numARows, numACols);
    ushort *output_data = (ushort *)calloc(sizeof(ushort), numBRows * numBCols);

    matrix_rref(input_data, output_data, numARows, numACols);

    //compute(output_data, input_data, numARows, numACols, numBRows, numBCols, numCRows, numCCols);

    write_data(input0_file_name, input_data, numARows, numACols);
    write_data(output_file_name, output_data, numBRows, numBCols);

    free(input_data);
    free(output_data);
}

int main() {
  base_dir = wbPath_join(wbDirectory_current(),
                         "RREF", "Dataset");

  create_dataset(0, 16, 16);
  create_dataset(1, 64, 64);
  return 0;
}
