#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include <sys/stat.h> // for numbering of output files
#include <mpi.h>

#include "biqbin.h"
#include "wrapper.h"

#ifdef PURE_C
extern FILE *output;
#endif
extern BiqBinParameters params;
extern Problem *SP;
extern Problem *PP;
extern int BabPbSize;

// macro to handle the errors in the input reading
#define READING_ERROR(file, cond, message)          \
    if ((cond))                                     \
    {                                               \
        fprintf(stderr, "\nError: " #message "\n"); \
        fclose(file);                               \
        exit(1);                                    \
    }

void print_symmetric_matrix(double *Mat, int N)
{

    double val;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            val = (i >= j) ? Mat[i + j * N] : Mat[j + i * N];
            printf("%24.16e", val);
        }
        printf("\n");
    }
}

int processCommandLineArguments(int argc, char **argv, int rank)
{

    int read_error = 0;

#ifdef PURE_C
    if (argc != 3)
    {
        if (rank == 0)
            fprintf(stderr, "Usage: ./biqbin file.rudy file.params\n");
        read_error = 1;
        return read_error;
    }

#else
    (void)argc;
    (void)argv;
#endif

    /***** only master process creates output file and reads the whole graph *****/

    // Control the command line arguments
    if (rank == 0)
    {

// Create the output file
#ifdef PURE_C
        char output_path[200];
        sprintf(output_path, "%s.output", argv[1]);

        // Check if the file already exists, if so aappend _<NUMBER> to the end of the output file name
        struct stat buffer;
        int counter = 1;

        while (stat(output_path, &buffer) == 0)
            sprintf(output_path, "%s.output_%d", argv[1], counter++);

        output = fopen(output_path, "w");
        if (!output)
        {
            fprintf(stderr, "Error: Cannot create output file.\n");
            read_error = 1;
            MPI_Bcast(&read_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
            return read_error;
        }

        // Read the input file instance
        double *adj;
        int adj_N;
        adj = readData(argv[1], &adj_N);
        if (!adj)
        {
            read_error = 1;
        }
        else
        {
            read_error = process_adj_matrix(adj, adj_N);
            free(adj);
        }
#else
        read_error = wrapped_read_data();
#endif

        // bcast first read_error then whole graph
        MPI_Bcast(&read_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (read_error)
            return read_error;
        else
        {
            MPI_Bcast(&(SP->n), 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(SP->L, SP->n * SP->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
    else
    {

        MPI_Bcast(&read_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (read_error)
            return read_error;

        // allocate memory for original problem SP and subproblem PP
        alloc(SP, Problem);
        alloc(PP, Problem);

        MPI_Bcast(&(SP->n), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // allocate memory for objective matrices for SP and PP
        alloc_matrix(SP->L, SP->n, double);
        alloc_matrix(PP->L, SP->n, double);

        MPI_Bcast(SP->L, SP->n * SP->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // IMPORTANT: last node is fixed to 0
        // --> BabPbSize is one less than the size of problem SP
        BabPbSize = SP->n - 1; // num_vertices - 1;
        PP->n = SP->n;

        int N2 = SP->n * SP->n;
        int incx = 1;
        int incy = 1;
        dcopy_(&N2, SP->L, &incx, PP->L, &incy);
    }

#ifdef PURE_C
    // Resolve once on rank 0, then use identical values on every rank.
    if (rank == 0)
        read_error = readParameters(argv[2], rank);
    MPI_Bcast(&read_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (read_error)
        return read_error;
    broadcastParameters(&params);
#endif

    /* adjust parameters */
    // change number of added cutting planes per iteration to n*10
    if (params.adjust_TriIneq)
        params.TriIneq = SP->n * 10;

#ifdef PURE_C

    if (rank == 0)
    {
        // print parameters to output file
        fprintf(output, "BiqBin parameters:\n");
#define P(type, name, format, def_value) \
    fprintf(output, "%20s = " format "\n", #name, params.name);
        PARAM_FIELDS
#undef P
    }
#endif
    return read_error;
}

void defaultParameters(BiqBinParameters *values)
{
#define P(type, name, format, def_value) values->name = def_value;
    PARAM_FIELDS
#undef P
}

const char *validateParameters(const BiqBinParameters *values)
{
#define P(type, name, format, def_value) \
    if (!isfinite((double)values->name) || values->name < 0) \
        return #name " must be finite and nonnegative";
    PARAM_FIELDS
#undef P
    if (values->init_bundle_iter < 1 || values->max_bundle_iter < 1)
        return "bundle iteration counts must be positive";
    if (values->init_bundle_iter > values->max_bundle_iter)
        return "init_bundle_iter must not exceed max_bundle_iter";
    if (values->max_outer_iter < 1)
        return "max_outer_iter must be positive";
    if (values->max_outer_iter > INT_MAX - values->extra_iter ||
        values->triag_iter > INT_MAX - values->pent_iter ||
        values->triag_iter + values->pent_iter > INT_MAX - values->hept_iter)
        return "iteration count sums exceed the native integer range";
    if (values->TriIneq < 1 || values->TriIneq > MaxTriIneqAdded ||
        values->PentIneq < 1 || values->PentIneq > MaxPentIneqAdded ||
        values->HeptaIneq < 1 || values->HeptaIneq > MaxHeptaIneqAdded)
        return "cut counts must be positive and not exceed the compiled capacity";
    if (values->adjust_TriIneq > 1 || values->include_Pent > 1 ||
        values->include_Hepta > 1 || values->root > 1 || values->use_diff > 1)
        return "boolean solver parameters must be 0 or 1";
    if (values->branchingStrategy != LEAST_FRACTIONAL && values->branchingStrategy != MOST_FRACTIONAL)
        return "branchingStrategy must be 0 or 1";
    return NULL;
}

void broadcastParameters(BiqBinParameters *values)
{
#define MPI_PARAM_int MPI_INT
#define MPI_PARAM_double MPI_DOUBLE
#define P(type, name, format, def_value) MPI_Bcast(&values->name, 1, MPI_PARAM_##type, 0, MPI_COMM_WORLD);
    PARAM_FIELDS
#undef P
#undef MPI_PARAM_int
#undef MPI_PARAM_double
}

/* Legacy parameter-file input for the standalone native executable. */
int readParameters(const char *path, int rank)
{
    FILE *paramfile = fopen(path, "r");
    char line[512];
    int line_number = 0;
    BiqBinParameters seen = {0};
    defaultParameters(&params);
    if (!paramfile)
    {
        if (rank == 0)
            fprintf(stderr, "Error: parameter file %s not found.\n", path);
        return 1;
    }
    while (fgets(line, sizeof(line), paramfile))
    {
        ++line_number;
        if (!strchr(line, '\n') && !feof(paramfile)) goto invalid;
        char *comment = strchr(line, '#');
        if (comment) *comment = '\0';
        char *key = line;
        while (isspace((unsigned char)*key)) ++key;
        if (!*key) continue;
        char *end_key = key;
        while (*end_key && !isspace((unsigned char)*end_key) && *end_key != '=') ++end_key;
        char *value = end_key;
        while (isspace((unsigned char)*value)) ++value;
        if (*value != '=') goto invalid;
        *end_key = '\0';
        ++value;
        while (isspace((unsigned char)*value)) ++value;
        char *end_value;
        errno = 0;
        double number = strtod(value, &end_value);
        if (value == end_value || errno || !isfinite(number)) goto invalid;
        while (isspace((unsigned char)*end_value)) ++end_value;
        if (*end_value) goto invalid;
        int matched = 0;
#define P(type, name, format, def_value) \
        if (strcmp(key, #name) == 0) { \
            if (seen.name) goto invalid; \
            if (strcmp(#type, "int") == 0) { \
                char *integer_end; \
                errno = 0; \
                long integer = strtol(value, &integer_end, 10); \
                if (integer_end == value || errno || integer < INT_MIN || integer > INT_MAX) goto invalid; \
                while (isspace((unsigned char)*integer_end)) ++integer_end; \
                if (*integer_end) goto invalid; \
            } \
            params.name = (type)number; seen.name = 1; matched = 1; \
        }
        PARAM_FIELDS
#undef P
        if (!matched) goto invalid;
        continue;
invalid:
        fprintf(stderr, "Error: invalid parameter at %s:%d\n", path, line_number);
        fclose(paramfile);
        return 1;
    }
    int failed = ferror(paramfile);
    fclose(paramfile);
    const char *error = validateParameters(&params);
    if (error) fprintf(stderr, "Error: %s\n", error);
    return failed || error != NULL;
}

/*** read graph file ***/
#ifdef PURE_C
double *readData(const char *instance, int *adj_N)
{
    // open input file
    FILE *f = fopen(instance, "r");
    if (f == NULL)
    {
        fflush(stdout);
        fprintf(stderr, "Error: problem opening input file %s\n", instance);
        return NULL;
    }
    printf("Input file: %s\n", instance);
    fflush(stdout);
    fprintf(output, "Input file: %s\n", instance);
    int num_vertices;
    int num_edges;

    READING_ERROR(f, fscanf(f, "%d %d \n", &num_vertices, &num_edges) != 2,
                  "Problem reading number of vertices and edges");
    READING_ERROR(f, num_vertices <= 0, "Number of vertices has to be positive");

    // OUTPUT information on instance
    BIQBIN_LOG_INFO("Graph has %d vertices and %d edges.", num_vertices, num_edges);
    fprintf(output, "\nGraph has %d vertices and %d edges.\n", num_vertices, num_edges);
    // read edges and store them in matrix Adj
    // NOTE: last node is fixed to 0
    int i, j;
    double weight;

    // Adjacency matrix Adj: allocate and set to 0
    double *Adj;
    alloc_matrix(Adj, num_vertices, double);
    *adj_N = num_vertices; // RK

    for (int edge = 0; edge < num_edges; ++edge)
    {

        READING_ERROR(f, fscanf(f, "%d %d %lf \n", &i, &j, &weight) != 3,
                      "Problem reading edges of the graph");

        READING_ERROR(f, ((i < 1 || i > num_vertices) || (j < 1 || j > num_vertices)),
                      "Problem with edge. Vertex not in range");

        int int_test = (int)weight;
        READING_ERROR(f, (double)int_test != weight, "Edge weight value error! All edge weights need to be integers!");

        Adj[num_vertices * (j - 1) + (i - 1)] = weight;
        Adj[num_vertices * (i - 1) + (j - 1)] = weight;
    }

    fclose(f);
    // RK BabPbSize = num_vertices - 1; // num_vertices - 1;
    return Adj;
}
#endif

/// @brief Builds the main Problem L matrix allocates Problem SP and PP global variables
/// @param Adj Adjacency matrix of the instance
/// @param num_vertices number of vertices in the graph
/// @return 0 if success 1 if fail
int process_adj_matrix(const double *Adj, int num_vertices)
{

    // Check if the problem size is compatible with Biqbin
    if (num_vertices > NMAX)
    {
        fprintf(stderr, "Number of vertices larger than NMAX in biqbin_cpp_api.h\n");
        abort_alloc_fail(1);
    }
    BabPbSize = num_vertices - 1; // RK // num_vertices - 1;
    // allocate memory for original problem SP and subproblem PP
    alloc(SP, Problem);
    alloc(PP, Problem);

    // size of matrix L
    SP->n = num_vertices; // num vertices

    // allocate memory for objective matrices for SP and PP
    alloc_matrix(SP->L, SP->n, double);
    alloc_matrix(PP->L, SP->n, double);

    // IMPORTANT: last node is fixed to 0
    // --> BabPbSize is one less than the size of problem SP
    PP->n = SP->n;

    /********** construct SP->L from Adj **********/
    /*
     * SP->L = [ Laplacian,  Laplacian*e; (Laplacian*e)',  e'*Laplacian*e]
     */
    // NOTE: we multiply with 1/4 afterwards when subproblems PP are created!
    //       (in function createSubproblem)
    // NOTE: Laplacian is stored in upper left corner of L

    // (1) construct vec Adje = Adj*e
    double *Adje;
    alloc_vector(Adje, num_vertices, double);

    for (int ii = 0; ii < num_vertices; ++ii)
    {
        for (int jj = 0; jj < num_vertices; ++jj)
        {
            Adje[ii] += Adj[jj + ii * num_vertices];
        }
    }

    // (2) construct Diag(Adje)
    double *tmp;
    alloc_matrix(tmp, num_vertices, double);
    Diag(tmp, Adje, num_vertices);

    // (3) fill upper left corner of L with Laplacian = tmp - Adj,
    //     vector parts and constant part
    double sum_row = 0.0;
    double sum = 0.0;

    // NOTE: skip last vertex!!
    for (int ii = 0; ii < num_vertices; ++ii)
    {
        for (int jj = 0; jj < num_vertices; ++jj)
        {

            // matrix part of L
            if ((ii < num_vertices - 1) && (jj < num_vertices - 1))
            {
                SP->L[jj + ii * num_vertices] = tmp[jj + ii * num_vertices] - Adj[jj + ii * num_vertices];
                sum_row += SP->L[jj + ii * num_vertices];
            }
            // vector part of L
            else if ((jj == num_vertices - 1) && (ii != num_vertices - 1))
            {
                SP->L[jj + ii * num_vertices] = sum_row;
                sum += sum_row;
            }
            // vector part of L
            else if ((ii == num_vertices - 1) && (jj != num_vertices - 1))
            {
                SP->L[jj + ii * num_vertices] = SP->L[ii + jj * num_vertices];
            }
            // constant term in L
            else
            {
                SP->L[jj + ii * num_vertices] = sum;
            }
        }
        sum_row = 0.0;
    }

    int N2 = SP->n * SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, SP->L, &incx, PP->L, &incy);

    free(Adje);
    free(tmp);

    return 0;
}
