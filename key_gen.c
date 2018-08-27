#include "qc_mdpc.h"
#include "matrix.h"
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
	mdpc code = qc_mdpc_init(n0, p, w, t);
	bin_matrix H = parity_check_matrix(code);
	//printf("H Generated...\n");
	bin_matrix G = generator_matrix(code);
	//printf("G Generated...\n");
	FILE *fp1, *fp2;
	fp1 = fopen("Private_Key.txt", "a");
	fprintf(fp1, "Private Key: Parity Check Matrix: \n");
	for(int i = 0; i < H->rows; i++)
	{
		for(int j = 0; j < H->cols; j++)
		{
			fprintf(fp1, "%hu ", get_matrix_element(H, i, j));
		}
		fprintf(fp1, "\n \n");
	}
	fclose(fp1);

	fp2 = fopen("Public_Key.txt", "a");
	fprintf(fp2, "Public Key: Generator Matrix: \n");
	for(int i = 0; i < G->rows; i++)
	{
		for(int j = 0; j < G->cols; j++)
		{
			fprintf(fp2, "%hu ", get_matrix_element(G, i, j));
		}
		fprintf(fp2, "\n \n");
	}
	fclose(fp2);

	return 0;
}