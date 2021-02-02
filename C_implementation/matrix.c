#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "matrix.h"
#include "qc_mdpc.h"
#include "utility.h"

//initialize the matrix
bin_matrix mat_init(int rows, int cols)
{
  if(rows <= 0 || cols <= 0)
  {
    return NULL;
  }
  bin_matrix A;
  A = (bin_matrix)safe_malloc(sizeof(struct matrix));
  A->cols = cols;
  A->rows = rows; 
  A->data = (unsigned short *)safe_malloc(rows*cols*sizeof(unsigned short)); 
  return A;
}


#define mat_element(mat, row_idx, col_idx) \
  mat->data[row_idx * (mat->cols) + col_idx]

//Return the matrix element at position given by the indices
unsigned short get_matrix_element(bin_matrix mat, int row_idx, int col_idx)
{
  if(row_idx < 0 || row_idx >= mat->rows || col_idx < 0 || col_idx >= mat->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  return mat->data[row_idx * (mat->cols) + col_idx];
}

//Set the value of matix element at position given by the indices to "val"
void set_matrix_element(bin_matrix A, int row_idx, int col_idx, unsigned short val)
{
  if(row_idx < 0 || row_idx >= A->rows || col_idx < 0 || col_idx >= A->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  mat_element(A, row_idx, col_idx) = val;
}

//Set the indicated row of the matrix A equal to the vector vec
void set_matrix_row(bin_matrix A, int row, unsigned short* vec)
{
  if(row < 0 || row >= A->rows)
  {
    printf("Row index out of range\n");
    exit(0);
  }
  for(int i = 0; i < A->cols; i++)
  {
    set_matrix_element(A, row, i, vec[i]);
  }
}

//Delete the matrix and free the space in memory
void delete_matrix(bin_matrix A)
{
  free(A);
}

//Return the transpose of the matrix A
bin_matrix transpose(bin_matrix A)
{
  bin_matrix B;
  B = mat_init(A->cols, A->rows);
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      set_matrix_element(B, j, i, mat_element(A, i, j));
    }
  }
  return B;
}

//Copy the data of matrix A to matrix B
bin_matrix mat_copy(bin_matrix A)
{
  bin_matrix B;
  int i;
  
  B = mat_init(A->rows, A->cols);                    
  memcpy(B->data, A->data, (A->rows)*(A->cols)*(sizeof(unsigned short)));
  return B;
}

//Add row1 to row2 of matrix A
bin_matrix add_rows(bin_matrix A,int row1, int row2)
{
  if(row1 < 0 || row1 >= A->rows || row2 < 0 || row2 >= A->rows)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  for(int i = 0; i < A->cols; i++)
  {
    mat_element(A, row2, i) = (mat_element(A, row1, i) ^ mat_element(A, row2, i));
  }
  return A;
}

//Add two matrices
bin_matrix add_matrix(bin_matrix A, bin_matrix B)
{
  if(A->rows != B->rows || A->cols != B->cols)
  {
    printf("Incompatible dimenions for matrix addition.\n");
    exit(0);
  }
  bin_matrix temp = mat_init(A->rows, A->cols);
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      set_matrix_element(temp, i, j, (mat_element(A, i, j) ^ mat_element(B, i, j)));
    }
  }
  return temp;
}

//Function to swap two rows of matrix A
void swap(bin_matrix A, int row1, int row2)
{
  if(row1 < 0 || row1 >= A->rows || row2 < 0 || row2 >= A->rows)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  int temp;
  for(int i = 0; i < A->cols; i++)
  {
    temp = mat_element(A, row1, i);
    mat_element(A, row1, i) = mat_element(A, row2, i);
    mat_element(A, row2, i) = temp;
  }
}

//Function to obtain the row reduced echlon form of a matrix A
bin_matrix matrix_rref(bin_matrix A)
{
  int lead = 0;
  int row_count = A->rows;
  int col_count = A->cols;
  bin_matrix temp = mat_init(row_count, col_count);
  temp = mat_copy(A);

  int r = 0;
  while(r < row_count)
  {
    if(mat_element(temp, r, r) == 0)
    {
      int i;
      for(i = r + 1; i < temp->rows; i++)
      {
        if(mat_element(temp, i, r) == 1)
        {
          swap(temp, r, i);
          break;
        }
      }
      if(i == row_count)
      {
      	printf("Matix cannot be transformed into row echlon form...");
        exit(1);
      }
    }
    else
    {
      for(int i = 0; i < row_count; i++)
      {
        if(mat_element(temp, i, r) == 1 && i != r)
        {
          add_rows(temp, r, i);
        }
      }
      r++;
    }
  }
  return temp;
}


//Multiplication of two matrices A and B stored in C
bin_matrix matrix_mult(bin_matrix A, bin_matrix B)
{
  if (A->cols != B->rows)
  {
    printf("Matrices are incompatible, check dimensions...\n");
    exit(0);
  }
  
  bin_matrix C;
  C = mat_init(A->rows, B->cols);
  bin_matrix B_temp = transpose(B);

  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0  ; j < B->cols; j++)
    {
      unsigned short val = 0;
      for(int k = 0; k < B->rows; k++)
      {
        val = (val ^ (mat_element(A, i, k) & mat_element(B_temp, j, k)));
      }
      mat_element(C, i, j) = val;
    }
  }
    
  return C;
}

//Set matrix as identity matrix
void make_indentity(bin_matrix A)
{
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      if(i == j)
      {
        mat_element(A, i, j) = 1;
      }
      else
      {
        mat_element(A, i, j) = 0;
      }
    }
  }
}

bool is_identity(bin_matrix A)
{
  bool flag = true;
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      if(i == j)
      {
        if(mat_element(A, i, j) == 0)
        {
          flag = false;
          return flag;
        }
      }
      else
      {
        if(mat_element(A, i, j) == 1)
        {
          flag = false;
          return flag;
        }
      }
    }
  }
  return flag;
}

//Checks if the matrix is a zero matrix
int is_zero_matrix(bin_matrix A)
{
  int flag = 1;
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      if(mat_element(A, i, j) != 0)
      {
        flag = 0;
        return flag;
      }
    }
  }
  return flag;
}

//Checks if two matrices are equal
int mat_is_equal(bin_matrix A, bin_matrix B)
{
  int flag = 1;
  if(A->rows != B->rows || A->cols != B->cols)
  {
    flag = 0;
    return flag;
  }
  for(int i = 0; i < A->rows; i++)
  {
    for(int j = 0; j < A->cols; j++)
    {
      if(mat_element(A, i, j) != mat_element(B, i, j))
      {
        flag = 0;
        return flag;
      }
    }
  }
  return flag;
}

//Add the elements of row1 to row2 in the column index range [a,b]  
bin_matrix add_rows_new(bin_matrix A,int row1, int row2, int a, int b)
{
  if(row1 < 0 || row1 >= A->rows || row2 < 0 || row2 >= A->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  for(int i = a; i < b; i++)
  {
    mat_element(A, row2, i) = (mat_element(A, row1, i) ^ mat_element(A, row2, i));
  }
  return A;
}

//Add col1 and col2 from in the row index range [a,b]
bin_matrix add_cols(bin_matrix A,int col1, int col2, int a, int b)
{
  if(col1 < 0 || col1 >= A->cols || col2 < 0 || col2 >= A->cols)
  {
    printf("Matrix index out of range\n");
    exit(0);
  }
  for(int i = a; i < b; i++)
  {
    mat_element(A, i, col2) = (mat_element(A, i, col1) ^ mat_element(A, i, col2));
  }
  return A;
}

//Inverse of matrix
bin_matrix circ_matrix_inverse(bin_matrix A)
{
  if(A->rows != A->cols)
  {
    printf("Inverse not possible...\n");
    exit(0);
  }

  if(is_identity(A))
  {
    return A;
  }

  bin_matrix B;
  B = mat_init(A->rows, A->cols);
  make_indentity(B);


  int i;
  int flag, prev_flag = 0;

  for(i = 0; i < A->cols; i++)
  {
    if(mat_element(A, i, i) == 1)
    {      
      for(int j = 0; j <  A->rows; j++)
      {
        if(i != j && mat_element(A, j, i) == 1)
        {
          add_rows_new(B, i, j, 0, A->cols);
          add_rows_new(A, i, j, i, A->cols);
        }
      }
    }
    else
    {
      int k;
      for(k = i + 1; k < A->rows; k++)
      {
        if(mat_element(A, k, i) == 1)
        {
          add_rows(B, k, i);
          add_rows(A, k, i);
          i = i - 1;
          break;
        } 
      }
    }
  }
  //printf("Out of for loop...\n");
  if(!is_identity(A))
  {
    printf("Could not find inverse, exiting...\n");  
    exit(-1);
  }

  
  return B;
}

//Obtain the specified number of rows and columns
bin_matrix mat_splice(bin_matrix A, int row1, int row2, int col1, int col2)
{
  int row_count = row2 - row1 + 1;
  int col_count = col2 - col1 + 1;
  int idx1, idx2;
  
  bin_matrix t = mat_init(row_count, col_count);
  for(int i = 0; i < row_count; i++)
  {
    idx1 = row1 + i;
    for(int j = 0; j < col_count; j++)
    {
      idx2 = col1 + j;
      set_matrix_element(t, i, j, mat_element(A, idx1, idx2));
    }
  }
  return t;
}

//Finding the basis of the kernel space of a matrix A
bin_matrix mat_kernel(bin_matrix A)
{
  int row_count = A->rows;
  int col_count = A->cols;

  bin_matrix temp = mat_init(col_count, row_count + col_count);

  bin_matrix ans = mat_init(col_count, col_count - row_count);
  for(int i = 0; i < temp->rows; i++)
  {
    for(int j = 0; j < row_count; j++)
    {
      set_matrix_element(temp, i, j, mat_element(A, j, i));
    }
  }

  for(int i = 0; i < col_count; i++)
  {
    set_matrix_element(temp, i, i + row_count, 1);
  }

  int r = 0;
  while(r < row_count)
  {
    if(mat_element(temp, r, r) == 0)
    {
      int i;
      for(i = r + 1; i < temp->rows; i++)
      {
        if(mat_element(temp, i, r))
        {
          swap(temp, r, i);
          break;
        }
      }
      if(i == temp->rows)
      {
        ans = mat_splice(temp, row_count, col_count - 1, row_count, row_count + col_count - 1);
        return (matrix_rref(ans));
      }
    }
    else
    {
      for(int i = 0; i < temp->rows; i++)
      {
        if(mat_element(temp, i, r) && i != r)
        {
          add_rows(temp, r, i);
        }
      }
      r++;
    }
  }
  ans = mat_splice(temp, row_count, col_count - 1, row_count, row_count + col_count - 1);
  return (matrix_rref(ans));
}

//Concatenate the matrices A and B as [A|B]
bin_matrix concat_horizontal(bin_matrix A, bin_matrix B)
{
  if(A->rows != B->rows)
  {
    printf("Incompatible dimensions of the two matrices. Number of rows should be same.\n");
    exit(0);
  }
  bin_matrix temp = mat_init(A->rows, A->cols + B->cols);
  for(int i = 0; i < temp->rows; i++)
  {
    for(int j = 0; j < temp->cols; j++)
    {
      if(j < A->cols)
      {
        set_matrix_element(temp, i, j, mat_element(A, i, j));
      }
      else
      {
        set_matrix_element(temp, i, j, mat_element(B, i, j - A->cols));
      }
    }
  }
  return temp;
}

//Concatenate the matrices A and B vertically
bin_matrix concat_vertical(bin_matrix A, bin_matrix B)
{
  if(A->cols != B->cols)
  {
    printf("Incompatible dimensions of the two matrices. Number of rows should be same.\n");
    exit(0);
  }
  bin_matrix temp = mat_init(A->rows + B->rows, A->cols);
  for(int i = 0; i < temp->rows; i++)
  {
    for(int j = 0; j < temp->cols; j++)
    {
      if(i < A->rows)
      {
        set_matrix_element(temp, i, j, mat_element(A, i, j));
      }
      else
      {
        set_matrix_element(temp, i, j, mat_element(B, i - A->rows, j));
      }
    }
  }
  return temp;
}

//Printing the matrix
void print_matrix(bin_matrix A)
{
  for(int i = 0; i < A->rows; i++)
  {
    for (int j = 0; j < A->cols; j++)
    {
      printf("%hu ", mat_element(A, i, j));
    }
    printf("\n");
  }
}


