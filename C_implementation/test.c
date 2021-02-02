#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "qc_mdpc.h"
#include "mceliece.h"
#include <time.h>
#include "utility.h"


int main(int argc, char const *argv[])
{

	// int n0 = 2;
	// int t = 84;
	// int p = 4801;
	// int w = 90;

	int n0 = 2;
	int t = 20;
	int p = 1000;
	int w = 45;
	
	printf("Starting Encryption...\n");
	clock_t start, end;
	double cpu_time_used;
	start = clock();
	mcc crypt = mceliece_init(n0, p, w, t);
	bin_matrix msg = mat_init(1, crypt->code->k);
	//Initializing the message a random message
	for(int i = 0; i < crypt->code->k; i++)
	{
		int z = rand() % 2;
		set_matrix_element(msg, 0, i, z);
	}

	bin_matrix v = encrypt(msg, crypt);
	bin_matrix s = decrypt(v, crypt);
	
	if(mat_is_equal(msg, s))
	{
		end = clock();
		printf("Decryption successful...\n");
		cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
		printf("Time taken by cryptosystem: %f\n", cpu_time_used);
	}
	else
	{
		printf("Failure....\n");
	}
	delete_mceliece(crypt);
	return 0;
}