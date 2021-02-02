#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "qc_mdpc.h"
#include "mceliece.h"
#include "utility.h"
#include <time.h>

//Initialize the mceliece cryptosystem
mcc mceliece_init(int n0, int p, int w, int t)
{
	mcc crypt;
	crypt = (mcc)safe_malloc(sizeof(struct mceliece));
	crypt->code = qc_mdpc_init(n0, p, w, t);
	crypt->public_key = generator_matrix(crypt->code);
	//printf("mceliece generated...\n");
	return crypt;
}

//Delete the cryptosystem
void delete_mceliece(mcc A)
{
	delete_qc_mdpc(A->code);
	delete_matrix(A->public_key);
	free(A);
}
//Generate a random error vector of length len of weight t
bin_matrix get_error_vector(int len, int t)
{
	 bin_matrix error = mat_init(1, len);
	 int weight = 0;
	 int idx;
	 while(weight < t)
	 {
	 	idx = random_val(1, len - 1, -1);
	 	if(!get_matrix_element(error, 0, idx))
	 	{
	 		set_matrix_element(error, 0, idx, 1);
	 		weight++;
	 	}
	 }
	 return error;
}

//Encrypting the message to be sent
bin_matrix encrypt(bin_matrix msg, mcc crypt)
{
	if(msg->cols != crypt->public_key->rows)
	{
		printf("Length of message is incorrect.\n");
		exit(0);
	}
	bin_matrix error = get_error_vector(crypt->code->n, crypt->code->t);
	//printf("error generated...\n");
	bin_matrix word = add_matrix(matrix_mult(msg, crypt->public_key), error);
	//printf("Messsage encrypted....\n");
	return word;
}

//Decrypting the recieved message
bin_matrix decrypt(bin_matrix word, mcc crypt)
{
	if(word->cols != crypt->code->n)
	{
		printf("Length of message is incorrect.\n");
		exit(0);
	}
	//printf("Decryption started...\n");
	bin_matrix msg = decode(word, crypt->code);
	msg = mat_splice(msg, 0, msg->rows - 1, 0, crypt->code->k - 1);
	return msg;
}