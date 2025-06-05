#include <math.h>
#include <string.h>

#include "biqbin.h"

// macro to handle the errors in the input reading
#define READING_ERROR(file,cond,message)\
        if ((cond)) {\
            fprintf(stderr, "\nError: "#message"\n");\
            fclose(file);\
            return 1;\
        }

// macro to handle the errors in the input reading
#define BQP_READING_ERROR(file,cond,message,...)\
        if ((cond)) {\
            fprintf(stderr, "\nError reading input file %s at line %d.", instance, line_cnt); \
            fprintf(stderr, "\n"#message); \
            fprintf(stderr, "\n" __VA_ARGS__);\
            fclose(file);\
            return 1;\
        }

typedef struct InputData
{
    int m, n;
    double *A;
    double *b;
    double *F;
    double *c;
    char name[255];
} InputData;

extern FILE *output;
extern Problem *SP;             
extern Problem *PP;            
extern int BabPbSize;

/* global variables for BQP->MC tansformation */
double const_val = 0.0;             // constant value used in BQP -> MC tranformation. The opt. value of the original problem is const_val - OP_MC
double rho = 0.0;                   // used in BQP -> MC transformation: exact penalty parameter = 2*rho + 1
double *F_obj_data;
double *c_obj_data;

/*** read input file containing data for linearly constrained BQP: 
     objective: F,c, constraints: A,b ***/
void post_process_BQP_input(InputData input_data) {

    int m = input_data.m;
    int n = input_data.n;
    double *A_con = input_data.A;
    double *b_con = input_data.b;
    double *F_obj = input_data.F;
    double *c_obj = input_data.c;
    double value;

    // print_matrix(F_obj,n,n);

    // copy F_obj to F_obj_data to hold original data
    alloc_matrix(F_obj_data, n, double); 
    int size_sq = n*n;
    int inc = 1;
    dcopy_(&size_sq, F_obj, &inc, F_obj_data, &inc);

    // copy c_obj to c_obj_data to hold original data
    alloc_vector(c_obj_data, n, double); 
    dcopy_(&n, c_obj, &inc, c_obj_data, &inc);

    /* constant term 1/4e'Fe + 1/2c'e */
    double constant = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            constant += 0.25 * F_obj[n * i + j];         
        }
    }
    
    for (int i = 0; i < n; ++i) {
        constant += 0.5 * c_obj[i];
    }    
    
    /* transformation from {0,1} to {-1,1} model */

    // b = b - 0.5*Ae
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            b_con[i] -= 0.5*A_con[j + i*n];
        }
    }
    //print_matrix(b_con,m,1);
 
    // A = 1/2*A (update now, in b we still needed the old value of A)
    size_sq = m*n;
    value = 0.5;
    dscal_(&size_sq, &value , A_con, &inc);
    

    // c = 1/2(Fe + c)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c_obj[i] += F_obj[n * i + j];         
        }
    }
    value = 0.5;
    dscal_(&n, &value , c_obj, &inc);

    // F = 1/4F
    value = 0.25;
    size_sq = n*n;
    dscal_(&size_sq, &value , F_obj, &inc);


    /* construct matrix C = [F, 0.5c; 0.5c' constant] */
    double *C;
    alloc_matrix(C, n+1, double);

    for (int ii = 0; ii < n+1; ++ii) {            
        for (int jj = 0; jj < n+1; ++jj) {

            // matrix part F
            if ( (ii < n) && (jj < n) ) {
                C[jj + ii * (n+1)] = F_obj[jj + ii * n];       
            }
            // vector part c
            else if ( (jj == n) && (ii != n)  ) {
                C[jj + ii * (n+1)] = 0.5*c_obj[ii];
            }
            // vector part c
            else if ( (ii == n) && (jj != n)  ) {
                C[jj + ii * (n+1)] = 0.5*c_obj[jj];
            }
            // constant term
            else { 
                C[jj + ii * (n+1)] = constant;
            }
        }
    } 

    //print_matrix(C,n+1,n+1);

    /* compute r1,r2 = max/min <C,X>, s.t. diag(X) = e, X psd */
    double *tmp_X;
    alloc_matrix(tmp_X, n+1, double);

    // r1 = max <C,X> ...
    double r1;
    ipm_mc_pk(C, n+1, tmp_X, &r1, 0);
    
    //printf("r1: %lf\n", r1);

    // r2 = min <C,X> = - max<-C,X> ...
    double r2;
    size_sq = (n+1)*(n+1);
    double *tmp_C;
    alloc_matrix(tmp_C, n+1, double);
    dcopy_(&size_sq, C, &inc, tmp_C, &inc);

    value = -1.0;
    dscal_(&size_sq, &value , tmp_C, &inc);
    ipm_mc_pk(tmp_C, n+1, tmp_X, &r2, 0);
    r2 = -r2;
    
    //printf("r2: %lf\n", r2);
    
    rho = fabs(r1) > fabs(r2) ? fabs(r1) : fabs(r2);
    printf("\nParameter rho: %lf\n", rho); 
    fprintf(output,"\nParameter rho: %lf\n", rho);

    /* exact penalty paramter */
    double pen = ceil(2*rho + 1);

    printf("Exact penalty parameter: %.0lf\n", pen);  
    fprintf(output,"Exact penalty parameter: %.0lf\n", pen);  


    /* construct objective matrix L_tmp for the max-cut problem */
    /*
     * L_tmp = -4 * [ F + pen*(A'A),  0.5*c - pen*(A'b); 0.5*c'' - pen*(b'A),  pen*(b'b)]
     */

    /* compute F = pen*(A'*A) + F */
    char TRANSA = 'N';
    char TRANSB = 'T';
    double beta = 1.0;
    double alpha = pen;

    dgemm_(&TRANSA, &TRANSB, &n, &n, &m, &alpha, A_con, &n, A_con, &n, &beta, F_obj, &n);
    
    //print_matrix(F_obj,n,n);
    //printf("\n");

    /* compute c = 0.5c - pen * A'b */
    char TRANS = 'N';
    alpha = -pen;
    beta = 0.5;
    int LDA = n;
    
    dgemv_(&TRANS, &n, &m, &alpha, A_con, &LDA, b_con, &inc, &beta, c_obj, &inc);

    //print_matrix(c_obj, n, 1);


    /* compute norm squared of b */
    double btb = ddot_(&m, b_con, &inc, b_con, &inc);
    //printf("%lf\n",btb);

    /* compute matrix L_tmp = -4 * [F c; c' pen*(b'b)] */
    double *L_tmp;
    alloc_matrix(L_tmp, n+1, double);


    for (int ii = 0; ii < n+1; ++ii) {            
        for (int jj = 0; jj < n+1; ++jj) {

            // matrix part of L
            if ( (ii < n) && (jj < n) ) {
                L_tmp[jj + ii * (n+1)] = F_obj[jj + ii * n];  
            }
            // vector part of L
            else if ( (jj == n) && (ii != n)  ) {
                L_tmp[jj + ii * (n+1)] = c_obj[ii];
            }
            // vector part of L
            else if ( (ii == n) && (jj != n)  ) {
                L_tmp[jj + ii * (n+1)] = L_tmp[ii + jj * (n+1)];
            }
            // constant term in L
            else { 
                L_tmp[jj + ii * (n+1)] = constant + pen * btb;
            }
        }
    } 

    //print_matrix(L_tmp,n+1,n+1);

    /* constant value val = sum(sum(L_tmp)) that will be added to the objective of the max-cut problem in the end! */
    for (int ii = 0; ii < n+1; ++ii) {            
        for (int jj = 0; jj < n+1; ++jj) {
            const_val += L_tmp[jj + ii * (n+1)];
        }
    } 

    printf("const_value: %lf\n\n", const_val);
    fprintf(output, "const_value: %lf\n\n", const_val);

    printf("************************** MAX-CUT output **********************************\n");
    fprintf(output, "************************** MAX-CUT output **********************************\n");

    /* build adjancency matrix of the underlying graph of L_tmp */
    /* Adj = 4*(L_tmp - diag(diag(L_tmp))) */
    // we multiply by 4 to get integer values in the end!

    double *Adj;
    alloc_matrix(Adj, n+1, double);
    for (int ii = 0; ii < n+1; ++ii) {            
        for (int jj = 0; jj < n+1; ++jj) {
            if (ii != jj)
                Adj[jj + ii * (n+1)] = 4*L_tmp[jj + ii * (n+1)];
        }
    } 
    //print_matrix(Adj,n+1,n+1);


    // we follow with the same steps as in readData function (this part of the code is copied):
    // construct diag(Ae) and Laplacian

    // allocate memory for original problem SP and subproblem PP
    alloc(SP, Problem);
    alloc(PP, Problem);

    // size of matrix L
    SP->n = n+1;   
    PP->n = SP->n;              

    // allocate memory for objective matrices for SP and PP
    alloc_matrix(SP->L, SP->n, double);
    alloc_matrix(PP->L, SP->n, double);

    // IMPORTANT: last node is fixed to 0
    // --> BabPbSize is one less than the size of problem SP
    BabPbSize = SP->n - 1;
    
    

    /********** construct SP->L from Adj **********/
    /*
     * SP->L = [ Laplacian,  Laplacian*e; (Laplacian*e)',  e'*Laplacian*e]
     */
    // NOTE: we multiply with 1/4 afterwards when subproblems PP are created!
    //       (in function createSubproblem)
    // NOTE: Laplacian is stored in upper left corner of L

    // (1) construct vec Adje = Adj*e 
    double *Adje;
    alloc_vector(Adje, n+1, double);

    for (int ii = 0; ii < n+1; ++ii) {
        for (int jj = 0; jj < n+1; ++jj) {
            Adje[ii] += Adj[jj + ii * (n+1)];
        }
    }

    // (2) construct Diag(Adje)
    double *tmp;
    alloc_matrix(tmp, n+1, double);
    Diag(tmp, Adje, n+1);

    // (3) fill upper left corner of L with Laplacian = tmp - Adj,
    //     vector parts and constant part      
    double sum_row = 0.0;
    double sum = 0.0;

    // NOTE: skip last vertex (it is fixed to 0)!!
    for (int ii = 0; ii < n+1; ++ii) {            
        for (int jj = 0; jj < n+1; ++jj) {

            // matrix part of L
            if ( (ii < n) && (jj < n) ) {
                SP->L[jj + ii * (n+1)] = tmp[jj + ii * (n+1)] - Adj[jj + ii * (n+1)]; 
                sum_row += SP->L[jj + ii * (n+1)];       
            }
            // vector part of L
            else if ( (jj == n) && (ii != n)  ) {
                SP->L[jj + ii * (n+1)] = sum_row;
                sum += sum_row;
            }
            // vector part of L
            else if ( (ii == n) && (jj != n)  ) {
                SP->L[jj + ii * (n+1)] = SP->L[ii + jj * (n+1)];
            }
            // constant term in L
            else { 
                SP->L[jj + ii * (n+1)] = sum;
            }
        }
        sum_row = 0.0;
    }  

    //print_matrix(SP->L,n+1,n+1);

    /* free stuff */
    free(C);
    free(tmp_X);
    free(tmp_C); 
    free(L_tmp);
    free(Adj);
    free(Adje);
    free(tmp);
}

int read_data_BQP(const char *instance) {

    InputData input_data;
    // input data  file line counter
    int line_cnt = 0;

    // open input file
    FILE *f = fopen(instance, "r");
    if (f == NULL) {
        fflush(stdout);
        fprintf(stderr, "Error: problem opening input file %s\n", instance);
        exit(1);
    }
    printf("Input file: %s\n", instance);
    fprintf(output,"Input file: %s\n", instance);


    // Read n: number of variables
    //      m: number of constraints
    int n;
    int m;

    line_cnt++;
    BQP_READING_ERROR(f, fscanf(f, "%d %d \n", &n, &m) != 2, 
                      "Problem reading number of variables and constraints. Number of arguments != 2");
    BQP_READING_ERROR(f, n <= 0, 
                      "Number of vertices has to be positive.", "Got n = %d\n", n);

    input_data.m = m;
    input_data.n = n;
    
    // OUTPUT information on instance
    fprintf(stdout, "\nInstance has %d variables and %d constraints.\n", n, m);
    fprintf(output, "\nInstance has %d variables and %d constraints.\n", n, m);


    // read matrix A in the constraints Ax = b 
    // NOTE: constraints should be integer!
    char letter;
    line_cnt++;
    BQP_READING_ERROR(f, fscanf(f, "%c\n", &letter) != 1 || letter != 'A', 
                      "Expected letter 'A'",  "Got letter '%c'\n", letter);

    int i, j;
    double value;

    // matrix A_con: allocate and set to 0 
    double *A_con;
    alloc_vector(A_con, m*n, double);
    input_data.A = A_con;

    char line[256];
    while (fgets(line, sizeof(line), f) != NULL)
    {
        line_cnt++;
        // Check for the 'b' line
        if (strcmp(line, "b\n") == 0)
        {
            break;
        }
               
        BQP_READING_ERROR(f, sscanf(line, "%d %d %lf \n", &i, &j, &value) != 3, 
                          "Matrix A: Number of parameters != 3."); 

        BQP_READING_ERROR(f, ((i < 1 || i > m) || (j < 1 || j > n) || (value != (long) value)), 
                          "Matrix A: Entries not in range or value is not an integer."
                          "(i < 1 || i > m) || (j < 1 || j > n) || (value != (long) value)", 
                          "Got: %s\n", line);
        
        A_con[ n * (i - 1) + (j - 1) ] = value;   

    }
    //print_matrix(A_con, m, n);


    // read vector b in the constraints Ax = b 
    // NOTE: constraints should be integer!
    
    // vector b_con: allocate and set to 0 
    double *b_con;
    alloc_vector(b_con, m, double);    
    input_data.b = b_con;

    while (fgets(line, sizeof(line), f) != NULL)
    {
        line_cnt++;
        // Check for the 'F' line
        if (strcmp(line, "F\n") == 0)
        {
            break;
        }

        BQP_READING_ERROR(f, sscanf(line, "%d %lf \n", &i, &value) != 2, 
                          "Vector b: Number of parameters != 2."); 

        BQP_READING_ERROR(f, (i < 1 || i > m || (value != (long) value)), 
                          "Vector b: Entries not in range or value is not an integer."
                          "i < 1 || i > m || (value != (long) value)",
                          "Got: %s\n", line);   
       
       b_con[ i-1 ] = value;
    }
    //print_matrix(b_con, m, 1);

    
    // read matrix F in the in the objective x'Fx + c'x
    // NOTE: constraints should be integer!
    
    // matrix F_obj: allocate and set to 0 
    double *F_obj;
    alloc_matrix(F_obj, n, double);  
    input_data.F = F_obj;


    while (fgets(line, sizeof(line), f) != NULL)
    {
        line_cnt++;
        // Check for the 'c' line
        if (strcmp(line, "c\n") == 0)
        {
            break;
        }
               
        BQP_READING_ERROR(f, sscanf(line, "%d %d %lf \n", &i, &j, &value) != 3,
                          "Matrix F: Number of parameters != 3."); 

        BQP_READING_ERROR(f, ((i < 1 || i > n) || (j < 1 || j > n) || (value != (long) value)),
                          "Matrix F: Entries not in range or value is not an integer."
                          "(i < 1 || i > n) || (j < 1 || j > n) || (value != (long) value).", 
                          "Got: %s\n", line);  
        
        // This needs to be done differently !!!
        if (i == j) {
            F_obj[ n * (i - 1) + (j - 1) ] = value;
        }
        else{
            F_obj[ n * (i - 1) + (j - 1) ] = value;   
            F_obj[ n * (j - 1) + (i - 1) ] = value;   

        }
    }

    // vector c_obj: allocate and set to 0 
    double *c_obj;
    alloc_vector(c_obj, n, double);    
    input_data.c = c_obj;

    while (fgets(line, sizeof(line), f) != NULL)
    {
        line_cnt++;

        BQP_READING_ERROR(f, sscanf(line, "%d %lf \n", &i, &value) != 2,
                          "Vector c: Number of parameters != 2."); 

        BQP_READING_ERROR(f, (i < 1 || i > n  || (value != (long) value)),
                          "Vector c: Entries not in range or value is not an integer."
                          "i < 1 || i > n  || (value != (long) value)",
                          "Got: %s\n", line);  
       
       c_obj[ i-1 ] = value;
    }
    post_process_BQP_input(input_data);

    return 0;    
}

