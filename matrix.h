#include <stdbool.h>
#include "utility.h"

#ifndef _MATRIX_H
#define _MATRIX_H

typedef struct matrix
{
   int rows;             //number of rows.
   int cols;             //number of columns.
   unsigned short *data;
}*bin_matrix;

bin_matrix mat_init(int rows, int cols);
unsigned short get_matrix_element(bin_matrix mat, int row_idx, int col_idx);
void set_matrix_element(bin_matrix A, int row_idx, int col_idx, unsigned short val);
void set_matrix_row(bin_matrix A, int row, unsigned short* vec);
void delete_matrix(bin_matrix A);
bin_matrix transpose(bin_matrix A);
bin_matrix mat_copy(bin_matrix A);
bin_matrix add_rows(bin_matrix A,int row1, int row2);
bin_matrix add_rows_new(bin_matrix A,int row1, int row2, int i1, int i2);
bin_matrix add_cols(bin_matrix A,int col1, int col2, int a, int b);
bin_matrix add_matrix(bin_matrix A, bin_matrix B);
void swap(bin_matrix A, int row1, int row2);
bin_matrix matrix_rref(bin_matrix A);
bin_matrix matrix_mult(bin_matrix A, bin_matrix B);
void make_indentity(bin_matrix A);
bool is_identity(bin_matrix A);
int is_zero_matrix(bin_matrix A);
int mat_is_equal(bin_matrix A, bin_matrix B);
bin_matrix matrix_inverse(bin_matrix A);
bin_matrix circ_matrix_inverse(bin_matrix A);
bin_matrix mat_splice(bin_matrix A, int row1, int row2, int col1, int col2);
bin_matrix mat_kernel(bin_matrix A);
bin_matrix concat_horizontal(bin_matrix A, bin_matrix B);
bin_matrix concat_vertical(bin_matrix A, bin_matrix B);
void print_matrix(bin_matrix A);

#endif
