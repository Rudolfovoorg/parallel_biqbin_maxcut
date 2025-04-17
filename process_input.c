#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>   // for numbering of output files
#include <mpi.h>

#include "biqbin.h"

extern FILE *output;
extern BiqBinParameters params;
extern GlobalVariables globals; 
extern int main_problem_size;

// macro to handle the errors in the input reading
#define READING_ERROR(file,cond,message)\
        if ((cond)) {\
            fprintf(stderr, "\nError: "#message"\n");\
            fclose(file);\
            return 1;\
        }


void print_symmetric_matrix(double *Mat, int N) {

    double val;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            val = (i >= j) ? Mat[i + j*N] : Mat[j + i*N];
            printf("%.0f", val);  // Output: 13
        }
        printf("\n");
    }
}

int open_output_file(char* filename) {
    // Create the output file
    char output_path[200];
    sprintf(output_path, "%s.output", filename);

    // Check if the file already exists, if so append _<NUMBER> to the end of the output file name
    struct stat buffer;
    int counter = 1;
    
    while (stat(output_path, &buffer) == 0)
        sprintf(output_path, "%s.output_%d", filename, counter++);

    output = fopen(output_path, "w");
    if (!output) {
        fprintf(stderr, "Error: Cannot create output file.\n");
        return 1;
    }

    return 0;
}

void close_output_file() {
    fclose(output);
}


void set_parameters(BiqBinParameters params_in) {
    params = params_in;
}

void print_parameters(BiqBinParameters params_in) {
    printf("BiqBin parameters:\n");
    #define P(type, name, format, def_value) \
        printf("%20s = " format "\n", #name, params_in.name);

    PARAM_FIELDS
    #undef P

    if (output) {
        fprintf(output, "BiqBin parameters:\n");
        #define P(type, name, format, def_value) \
        fprintf(output ? output : stdout, "%20s = " format "\n", #name, params_in.name);

        PARAM_FIELDS
        #undef P
    }
}