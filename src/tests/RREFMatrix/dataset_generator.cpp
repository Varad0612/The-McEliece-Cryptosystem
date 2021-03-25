#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "wb.h"
#include "RREF_cpu.c"

static char *base_dir;

static ushort *generate_data(int height, int width) {
  ushort *data = (ushort *)malloc(sizeof(ushort) * width * height);
  int i;
  for (i = 0; i < width * height; i++) {
    //data[i] = ((ushort)(rand() % 20) - 5) / 5.0f;
    data[i] = (ushort)(rand() % 2);
  }
  return data;
}

static void write_data(char *file_name, ushort *data, int height,
                       int width) {
  int ii, jj;
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d %d\n", height, width);
  for (ii = 0; ii < height; ii++) {
    for (jj = 0; jj < width; jj++) {
      fprintf(handle, "%.2hi", *data++);
      if (jj != width - 1) {
        fprintf(handle, " ");
      }
    }
    if (ii != height - 1) {
      fprintf(handle, "\n");
    }
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, int numARows, int numACols) {
    int numBRows = numACols;
    int numBCols = numACols;

    const char *dir_name =
        wbDirectory_create(wbPath_join(base_dir, datasetNum));

    char *input0_file_name = wbPath_join(dir_name, "input.raw");
    char *output_file_name = wbPath_join(dir_name, "output.raw");

    ushort *input_data = generate_data(numARows, numACols);
    ushort *output_data = (ushort *)calloc(sizeof(ushort), numBRows * numBCols);

    matrix_rref(input_data, output_data, numARows, numACols);

    //compute(output_data, input_data, numARows, numACols, numBRows, numBCols, numCRows, numCCols);

    write_data(input0_file_name, input_data, numARows, numACols);
    write_data(output_file_name, output_data, numBRows, numBCols);

    free(input_data);
    free(output_data);
}

int main() {
  base_dir = wbPath_join(wbDirectory_current(),
                         "RREF", "Dataset");

  create_dataset(0, 16, 16);
  create_dataset(1, 64, 64);
  return 0;
}
