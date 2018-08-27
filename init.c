#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "qc_mdpc.h"
#include "mceliece.h"

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

	mcc crypt = mceliece_init(n0, p, w, t);
	int opt;
	printf("Action: 0)Quit	1)keygen	2)encrypt	3)decrypt\n");
	printf("Type 0, 1, 2 or 3...\n");
	
	while(true)
	{
		printf("Action: ");
		scanf("%d", &opt);
		if(opt == 0)
		{
			break;
		}
		else if(opt == 1)
		{
			bin_matrix H = parity_check_matrix(crypt->code);
			bin_matrix G = generator_matrix(crypt->code);
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
			printf("Keys Generated...\n");
		}
		else if(opt == 2)
		{
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
			printf("Encrypted...\n");
		}
		else if(opt == 3)
		{
			printf("Enter code of length %d: ", k);
			unsigned short inp;
			bin_matrix msg = mat_init(1, k);
			for(int i = 0; i < k; i++)
			{
				scanf("%hu", &inp);
				set_matrix_element(msg, 0, i, inp);
			}
			mcc crypt = mceliece_init(n0, p, w, t);
			bin_matrix m = decrypt(msg, crypt);
			
			FILE *fp1;
			fp1 = fopen("Decryption.txt", "a");
			fprintf(fp1, "Decrypted message: \n");
			for(int i = 0; i < m->cols; i++)
			{
				fprintf(fp1, "%hu ", get_matrix_element(m, 0, i));
			}
			fclose(fp1);
			printf("Decrypted...\n");
		}
	}
	return 0;
}