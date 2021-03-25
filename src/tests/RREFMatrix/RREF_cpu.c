#include <stdio.h>
#include <stdlib.h>
#include "RREF_cpu.h"

#define mat_element(mat, cols, row_idx, col_idx) \
  mat[(row_idx * cols) + col_idx]


void swap(ushort *A, int row1, int row2, int rows, int cols)
{
    int temp;
    for(int i = 0; i < cols; i++)
    {
        temp = mat_element(A, cols, row1, i);
        mat_element(A, cols, row1, i) = mat_element(A, cols, row2, i);
        mat_element(A, cols, row2, i) = temp;
    }
}



void add_rows(ushort *A, int row1, int row2, int rows, int cols)
{
  for(int i = 0; i < cols; i++)
  {
    mat_element(A, cols, row2, i) = \
        (mat_element(A, cols, row1, i) ^ mat_element(A, cols, row2, i));
  }
}



//Function to obtain the row reduced echlon form of a matrix A
void matrix_rref(ushort *A, ushort *B, int rows, int cols)
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
