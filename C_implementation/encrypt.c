#include "qc_mdpc.h"
#include "matrix.h"
#include "mceliece.h"
#include <stdlib.h>
#include <stdio.h>


int main(int argc, char const *argv[])
{
	int n0, p, w, t;
	printf("Enter n0: ");
	scanf("%d", &n0);
	printf("Enter p: ");
	scanf("%d", &p);
	printf("Enter w: ");
	scanf("%d", &w);
	printf("Enter t: ");
	scanf("%d", &t);
	int k = (n0 - 1) * p;
	printf("Enter Message of length %d: \n", k);
	unsigned short inp;

	bin_matrix msg = mat_init(1, k);
	for(int i = 0; i < k; i++)
	{
		scanf("%hu", &inp);
		set_matrix_element(msg, 0, i, inp);
	}

	mcc crypt = mceliece_init(n0, p, w, t);
	bin_matrix m = encrypt(msg, crypt);
	
	FILE *fp1;
	fp1 = fopen("Encryption.txt", "a");
	fprintf(fp1, "Encrypted message: \n");
	for(int i = 0; i < m->cols; i++)
	{
		fprintf(fp1, "%hu ", get_matrix_element(m, 0, i));
	}
	fclose(fp1);

	return 0;
}