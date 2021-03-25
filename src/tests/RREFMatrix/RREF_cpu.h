#ifndef RREF_CPU_H
#define RREF_CPU_H

#ifdef __cplusplus
extern "C" {
#endif

#define ushort unsigned short

void swap(ushort *A, int row1, int row2, int rows, int cols);
void add_rows(ushort *A, int row1, int row2, int rows, int cols);
void matrix_rref(ushort *A, ushort *B, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif //RREF_CPU_H
