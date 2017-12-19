#include "qc_mdpc.h"
#include <stdlib.h>
#include "matrix.h"
#include "utility.h"
#include <stdio.h>
#include <time.h>


mdpc qc_mdpc_init(int n0, int p, int w, int t)
{
	mdpc code;
	code = (mdpc)safe_malloc(sizeof(struct qc_mdpc));
	code->n0 = n0;
	code->p = p;
	code->w = w;
	code->t = t;
	code->n = n0 * p;
	code->r = p;
	code->k = (n0 - 1) * p;
	unsigned seed;
	code->row = (unsigned short*)calloc(n0 * p, sizeof(unsigned short));
	printf("Input seed or -1 to use default seed: ");
	scanf("%u", &seed);
	time_t tx;
	if(seed == -1)
	{
	  srand((unsigned) time(&tx));
	}
 	else
  	{
    	srand(seed);
  	}

	while(1)
    {
    	int flag = 0;
    	int idx;
    	while(flag < w)
		{
			idx = random_val(0, (n0 * p) - 1, seed);
			if(!code->row[idx])
		    {
		      code->row[idx] = 1;
		      flag = flag + 1;
		    }
		}
    	if((get_row_weight(code->row, (n0 - 1) * p, (n0 * p)-1)) % 2 == 1)
		{
			break;
		}
    	reset_row(code->row, 0, n0 * p);
    }
	printf("MDPC code generated....\n");
	return code;
}

//Delete the qc-mdpc code
void delete_qc_mdpc(mdpc A)
{
  free(A);
}

//Returns a random integer in the range [min, max]
int random_val(int min, int max, unsigned seed)
{
	int r;
	const unsigned int range = 1 + max - min;
	const unsigned int buckets = RAND_MAX / range;
	const unsigned int limit = buckets * range;
    
	do
	{
		r = rand();
	} while (r >= limit);

	return min + (r / buckets);
}

//Return the weight of the given row from the indices min to max
int get_row_weight(unsigned short* row, int min, int max)
{
	int weight = 0;
	int i;
	for(i = min; i < max + 1; i++)
	{
  		if(row[i] == 1)
		{
			weight++;
		}
	}
	return weight;
}

//Reset all positions in the row to 0
void reset_row(unsigned short* row, int min, int max)
{
	int i;
	for(i = min; i < max + 1; i++)
	{
	  row[i] = 0;
	}
}

//Rotate the row x positions to the right
unsigned short* shift(unsigned short* row, int x, int len)
{
	unsigned short* temp = (unsigned short*)calloc(len, sizeof(unsigned short));
	int i;
	for(i = 0; i < len; i++)
	{
	  temp[(i + x) % len] = row[i];
	}
	return temp;
}

//Create a binary circular matrix
bin_matrix make_matrix(int rows, int cols, unsigned short* vec, int x)
{
	bin_matrix mat = mat_init(rows, cols);
	set_matrix_row(mat, 0, vec);
	int i;
	for(i = 1; i < rows; i++)
	{
	  vec = shift(vec, x, cols);
	  set_matrix_row(mat, i, vec);
	}
	return mat;
}

//Splice the row for the given range (does not include max)
unsigned short* splice(unsigned short* row, int min, int max)
{
	unsigned short* temp = (unsigned short*)calloc(max - min, sizeof(unsigned short));
	int i;
	for(i = min; i < max; i++)
	{
	  temp[i - min] = row[i];
	}
	return temp;
}

//Constructing the pariy check matrix
bin_matrix parity_check_matrix(mdpc code)
{
	clock_t start, end;
	double cpu_time_used;
	start = clock();
	bin_matrix H = make_matrix(code->p, code->p, splice(code->row, 0, code->p), 1);
	int i;
	for(i = 1; i < code->n0; i++)
	{
	  bin_matrix M = make_matrix(code->p, code->p, splice(code->row, i * code->p, (i + 1) * code->p), 1);
	  H = concat_horizontal(H, M);
	}
	end = clock();
	cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
	printf("Time for H: %f\n", cpu_time_used);
	// printf("H: \n");
	// print_matrix(H);
	//printf("Parity matrix generated...\n");
	return H;
}

//Constructing the generator matrix
bin_matrix generator_matrix(mdpc code)
{
	clock_t start, end;
	double cpu_time_used;
	start = clock();
	bin_matrix H = parity_check_matrix(code);


	//End of modified code
	printf("Construction of G started...\n");
	bin_matrix H_inv = circ_matrix_inverse(make_matrix(code->p, code->p, splice(code->row, (code->n0 - 1) * code->p, code->n), 1));
	//printf("H_inv generated...\n");
	//printf("stop\n");
	bin_matrix H_0 = make_matrix(code->p, code->p, splice(code->row, 0, code->p), 1);
	bin_matrix Q = transpose(matrix_mult(H_inv,  H_0));
	//printf("Transpose obtained...\n");
	bin_matrix M;
	int i;
	for(i = 1; i < code->n0 - 1; i++)
	{
	  M = make_matrix(code->p, code->p, splice(code->row, i * code->p, (i + 1) * code->p), 1);
	  M = transpose(matrix_mult(H_inv, M));
	  Q = concat_vertical(Q, M);
	}
	bin_matrix I = mat_init(code->k, code->k);
	make_indentity(I);
	bin_matrix G = concat_horizontal(I, Q);

	//bin_matrix G = mat_kernel(H);
	//G = matrix_rref(G);
	end = clock();
	cpu_time_used = ((double) (end - start))/ CLOCKS_PER_SEC;
	printf("Time for G: %f\n", cpu_time_used);
	printf("Generator matrix generated....\n");
	return G;
}

//Returns the maximum element of the array
int get_max(int* vec, int len)
{
	int max = vec[0];
	int i;
	for(i = 1; i < len; i++)
	{
		if(vec[i] > max)
		{
			max = vec[i];
		}
	}
	return max;
}

//Encoding the vector vec as a MDPC codeword
bin_matrix encode(bin_matrix vec, mdpc code)
{
  bin_matrix G = generator_matrix(code);
  bin_matrix msg = matrix_mult(vec, G);
  return msg;
}

//Decoding the codeword
bin_matrix decode(bin_matrix word, mdpc code)
{
  bin_matrix H = parity_check_matrix(code);
  bin_matrix syn  = matrix_mult(H, transpose(word));
  int limit = 10;
  int delta = 5;
  int i,j,k,x;

	for(i = 0; i < limit; i++)
	{
		//printf("Iteration: %d\n", i);
		int unsatisfied[word->cols];
		for(x = 0; x < word->cols; x++)
		{
			unsatisfied[x] = 0;
		}
		for(j = 0; j < word->cols; j++)
		{
			for(k = 0; k < H->rows; k++)
			{
				if(get_matrix_element(H, k, j) == 1)
				{
					if(get_matrix_element(syn, k, 0) == 1)
					{
						unsatisfied[j] = unsatisfied[j] + 1;
					}
				}
			}
		}
		// printf("No. of unsatisfied equations for each bit: \n");
		// for(int idx = 0; idx < word->cols; idx++)
		// {
		// 	printf("b%d: %d \n", idx, unsatisfied[idx]);
		// }
		int b = get_max(unsatisfied, word->cols) - delta;
		for(j = 0; j < word->cols; j++)
		{
			if(unsatisfied[j] >= b)
			{
				set_matrix_element(word, 0, j, (get_matrix_element(word, 0, j) ^ 1));
				syn = add_matrix(syn, mat_splice(H, 0, H->rows - 1, j, j));
			}
		}
		// printf("Syndrome: ");
		// print_matrix(syn);
		// printf("\n");
		//printf("Iteration: %d\n", i);
		if(is_zero_matrix(syn))
		{
			return word;
		}
	}
	printf("Decoding failure...\n");
	exit(0);
}



