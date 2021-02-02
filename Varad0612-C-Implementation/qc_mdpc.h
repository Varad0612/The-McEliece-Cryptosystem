#include "matrix.h"
#include "utility.h"


#ifndef QC_MDPC_H
#define QC_MDPC_H
typedef struct qc_mdpc
{
	unsigned short* row;
	int n0, p, w, t, n, k, r;
}*mdpc;

mdpc qc_mdpc_init(int n0, int p, int w, int t);
void delete_qc_mdpc(mdpc A);
int random_val(int min, int max, unsigned seed);
int get_row_weight(unsigned short* row, int min, int max);
void reset_row(unsigned short* row, int min, int max);
unsigned short* shift(unsigned short* row, int x, int len);
unsigned short* splice(unsigned short* row, int min, int max);
bin_matrix make_matrix(int rows, int cols, unsigned short* vec, int x);
bin_matrix generator_matrix(mdpc code);
bin_matrix parity_check_matrix(mdpc code);
int get_max(int* vec, int len);
bin_matrix encode(bin_matrix vec, mdpc code);
bin_matrix decode(bin_matrix word, mdpc code);

#endif