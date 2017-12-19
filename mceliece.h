#include "qc_mdpc.h"
#include "matrix.h"
#include "utility.h"


#ifndef MCELIECE_H
#define MCELIECE_H

typedef struct mceliece
{
   mdpc code;
   bin_matrix public_key;
}*mcc;

mcc mceliece_init(int n0, int p, int w, int t);
void delete_mceliece(mcc A);
bin_matrix get_error_vector(int len, int t);
bin_matrix encrypt(bin_matrix msg, mcc crypt);
bin_matrix decrypt(bin_matrix word, mcc crypt);
#endif