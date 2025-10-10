#include <math.h>
#include <string.h>
#include <stdio.h>

#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // for std::vector

#include "bqp_api.h"
namespace py = pybind11;

// ***********************************************
// Functions copied from biqbin source code
// ***********************************************

void Diag(double *X, const double *y, int n) {

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j)
                X[i + i * n] = y[i];
            else
                X[j + i * n] = 0.0;
        }
    }
}

void diag(const double *X, double *y, int n) {

    for (int i = 0; i < n; ++i) {
        y[i] = X[i + i * n];
    }
}

void ipm_mc_pk(double *L, int n, double *X, double *phi, int print) {

    /* variables for blas and lapack routines */
    int inc = 1;
    char up = 'U';              // for lapack take upper triangular part of matrix 
    char side = 'L';            // in matrix product AB, the left matrix is symmetric 
    int info;                   // test whether lapack function succeded
    double alpha, beta;         // scalars in linear combination (lapack)
    double gap;                 // duality gap

    /* other variables */
    int i, j, k;                // loop iter
    double *p, *p2, *p3;        // pointers in loops
    double psi;                 // value of primal problem
    double mu;                  // ZX = mu * I (parametrized optimality condition)
    double alpha_p, alpha_d;    // step lengths

    /* dual variables */
    double *y;                  // dual variable to diagonal constraints
    double *Z;                  // dual variable to X >= 0

    double *b;                  // vector of ones
    double *dX, *dy, *dZ;       
    double *Zi;                 // inv(Z)
    double *M;                  // M * dy = rhs
    double *dy1, *dX1;
    double *tmp, *tmp2;         // need non-symm matrices when computing for instance Zi*diag(dy)*X       

    /*************************************************
     * initial positive definite matrices X, Z and y *
     * primal, dual cost, gap                         *
     *************************************************/
    alloc_vector(y, n, double);
    alloc_matrix(Z, n, double);

    int nn = n*n;

    /* set y to zero vector */
    for (int i = 0; i < n; ++i)
        y[i] = 0.0;

    /* y = 1.1 + sum(abs(L))' */
    for (i = 1, p = L, p2 = y; i <= nn; ++i, ++p) {
        *p2 += fabs(*p);
        if ((i % n) == 0) {
            *p2 += 1.1;
            ++p2;
        }
    }

    /* vector b of all ones */
    alloc_vector(b, n, double);
    for (i = 0; i < n; ++i)
        b[i] = 1.0;
    
    /* X = eye(n) = Diag(b) */
    Diag(X, b, n);

    /* Z = Diag(y) - L */
    Diag(Z, y, n);                           /* Z = Diag(y) */
    alpha = -1.0;
    daxpy_(&nn,&alpha,L,&inc,Z,&inc);        /* Z = Z - L */


    /* phi = ones(n,1)'*y */                 /* initial dual value */
    *phi = ddot_(&n,b,&inc,y,&inc);

    /* psi = L(:)'*X(:) */                   /* initial primal value */
    psi = ddot_(&nn,L,&inc,X,&inc);

    /* mu = Z(:)'*X(:) / (2*n); */           /* initial complementarity */ 
    mu = ddot_(&nn,Z,&inc,X,&inc) / (2.0 * n);


    /* print output */
    if (print) {
        puts("iter     log10(gap)     primal        dual");
        puts("*******************************************"); 
    }

    /*  allocate space */
    alloc_matrix(dX, n, double);
    alloc_vector(dy, n, double);
    alloc_matrix(dZ, n, double);

    alloc_vector(dy1, n, double);
    alloc_matrix(dX1, n, double);
    alloc_matrix(tmp, n, double);
    alloc_matrix(tmp2, n, double);

    alloc_matrix(Zi, n, double);    
    alloc_matrix(M, n, double);   

    /*************
     * main loop *
     *************/
    gap = fabs(*phi - psi);
    
    for (i = 1; gap > 1e-2; ++i) {/* while duality gap too large */
        
        /******** compute inverse of Z ********/
        dcopy_(&nn,Z,&inc,Zi,&inc);         /* copy Z to Zi */   

        dpotrf_(&up,&n,Zi,&n,&info);        /* computes Cholesky factorization */
        if (info != 0) {
            fprintf(stderr, "%s: Problem with Cholesky factorization \
                (line: %d).\n", __func__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD,10);
        }

        dpotri_(&up,&n,Zi,&n,&info);        /* computes Zi = inv(Z) */
        if (info != 0) {
            fprintf(stderr, "%s: Problem with computation of inverse matrix \
                (line: %d).\n", __func__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD,10);
        }

        /* NOTE: only upper triangular part of Zi is ok and can be used! */
        /* copy strictly upper triangular part of Zi to strictly lower triangular part */
        for (j = 0; j < n; ++j) {       
            for (k = 0; k < j; ++k)
                Zi[j+n*k] = Zi[k+n*j];
        }


        /********   predictor step (mu = 0) solves:       ********
         ********   Z * X + Diag(dy1) * X + Z * dX1 = 0   ********/

        /* M = Zi .* X */
        for (j = 0, p = M; j < nn; ++j, ++p)
            *p = Zi[j] * X[j];

        /* copy -b to dy1 */
        alpha = -1.0;
        dcopy_(&n,b,&inc,dy1,&inc);  // copy
        dscal_(&n,&alpha,dy1,&inc);  // scale     

        /* solve the system: dy1 = (Zi .* X) \ (-e) */
        dposv_(&up, &n, &inc, M, &n, dy1, &n, &info);
        /* NOTE: M and dy1 are changed on exit! */

        if (info != 0) {
            fprintf(stderr, "%s: predictor step: problem in solving linear system \
                (line: %d).\n", __func__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD,10);
        }

        /* dX1 = -Zi*diag(dy1)*X - X */
        // NOTE: be carefull to multiply in column-major ordering!

        /* 1. step: tmp = -diag(dy1)*X */
        /* = multiply j-th row of X by -dy1[j]        (FORTRAN) */
        /* = multiply j-th column of X by -dy[j]      (C)       */
        p3 = tmp; 
        for (j = 0, p = X, p2 = dy1; j < nn; ++j, ++p, ++p3) {
            *p3 = -(p2[j%n]) * (*p);
        }

        /* 2. step: Zi * tmp */
        alpha = 1.0;
        beta = 0.0;
        dsymm_(&side,&up,&n,&n,&alpha,Zi,&n,tmp,&n,&beta,dX1,&n);
     
        /* dX1 = dX1 - X */
        alpha = -1.0;
        daxpy_(&nn,&alpha,X,&inc,dX1,&inc);

        /* symmetrize: dX1 = (dX1 + dX1')/2 */
        for (j = 0; j < n; ++j) {       
            for (k = 0; k <= j; ++k)    
                dX1[k+n*j] = dX1[j+n*k] = 0.5 * (dX1[j+n*k] + dX1[k+n*j]);
        }

  
        /*************** corrector step solves:  ******************/  
        /******** diag(dy2)*X + Z*dX2 - mu*I + diag(dy1)*dX1 = 0 **/

        /* dy2 = M \ (mu*diag(Zi) - (Zi .* dX1)*dy1) */

        /* dy = diag(Zi) */
        diag(Zi,dy,n);     

        /* tmp = Zi .* dX1 */
        for (j = 0, p = tmp; j < nn; ++j, ++p)
            *p = Zi[j] * dX1[j];

        /* dy = -tmp*dy1 + mu*dy */
        alpha = -1.0;
        dsymv_(&up,&n,&alpha,tmp,&n,dy1,&inc,&mu,dy,&inc);

        // NOTE: M was changed during dposv
        /* M = Zi .* X */
        for (j = 0, p = M; j < nn; ++j, ++p)
            *p = Zi[j] * X[j];

        /* dy2 = M \ dy */
        /* dy2 is saved in dy! */
        dposv_(&up, &n, &inc, M, &n, dy, &n, &info);
        if (info != 0) {
            fprintf(stderr, "%s: corrector step: problem in solving linear system \
                (line: %d).\n", __func__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD,10);
        }

        /* dX2 = mu*Zi - Zi*( diag(dy2) * X + diag(dy1) * dX1) */

        /* 1. step: tmp = -diag(dy2)*X */
        /* = multiply j-th row of X by -dy2[j]       (FORTRAN) */
        /* = multiply j-th column of X by -dy2[j]    (C)       */    
        /* NOTE: dy2 = dy */
        p3 = tmp; 
        for (j = 0, p = X, p2 = dy; j < nn; ++j, ++p, ++p3) {
            *p3 = -(p2[j%n]) * (*p);
        }

        /* 2. step: tmp2 = -diag(dy1)*dX1 */
        p3 = tmp2; 
        for (j = 0, p = dX1, p2 = dy1; j < nn; ++j, ++p, ++p3) {
            *p3 = -(p2[j%n]) * (*p);
        }
        
        /* tmp = tmp + tmp2 */
        alpha = 1.0;
        daxpy_(&nn,&alpha,tmp2,&inc,tmp,&inc);

        /* dX2 = Zi * tmp*/
        /* NOTE: dX2 is stored in dX */
        alpha = 1.0;
        beta = 0.0;
        dsymm_(&side,&up,&n,&n,&alpha,Zi,&n,tmp,&n,&beta,dX,&n);
        
        /* dX = dX2 = mu*Zi + dX2 */
        daxpy_(&nn,&mu,Zi,&inc,dX,&inc);

      
        /**** final steps ****/
        alpha = 1.0;
        daxpy_(&n,&alpha,dy1,&inc,dy,&inc);     /* dy = dy1 + dy */
        daxpy_(&nn,&alpha,dX1,&inc,dX,&inc);    /* dX = dX1 + dX */
        
        /* symmetrize: dX = (dX + dX')/2 */
        for (j = 0; j < n; ++j) {       
            for (k = 0; k <= j; ++k)    
                dX[k+n*j] = dX[j+n*k] = 0.5 * (dX[j+n*k] + dX[k+n*j]);
        }

        /* dZ = Diag(dy) */
        Diag(dZ,dy,n);
                

        /*********** find step lengths alpha_p and alpha_d ***********/
        
        /* line search on primal: X  = X + alpha_p * dX  psd matrix */
        alpha_p = 1.0;
        info = 1;
        while (info != 0) {
            dcopy_(&nn,X,&inc,tmp,&inc);            /* tmp = X */
            daxpy_(&nn,&alpha_p,dX,&inc,tmp,&inc);  /* tmp = alpha_p * dX + tmp */
            dpotrf_(&up,&n,tmp,&n,&info);

            if (info != 0)
                alpha_p *= 0.8;
        }
        
        if (alpha_p < 1.0)                          /* stay away from boundary */
            alpha_p *= 0.95;


        /* line search on dual */
        alpha_d = 1.0;
        info = 1;
        while (info != 0) {
            dcopy_(&nn,Z,&inc,tmp,&inc);            /* tmp = Z */
            daxpy_(&nn,&alpha_d,dZ,&inc,tmp,&inc);  /* tmp = alpha_d * dZ + tmp */
            dpotrf_(&up,&n,tmp,&n,&info);

            if (info != 0)
                alpha_d *= 0.8;
        }
        
        if (alpha_d < 1.0)                          /* stay away from boundary */
            alpha_d *= 0.95;


        /******** update ********/
        daxpy_(&nn,&alpha_p,dX,&inc,X,&inc);        /* X = alpha_p * dX + X */    
        daxpy_(&n,&alpha_d,dy,&inc,y,&inc);         /* y = alpha_d * dy + y */
        daxpy_(&nn,&alpha_d,dZ,&inc,Z,&inc);        /* Z = alpha_d * dZ + Z */

        /* mu = Z(:)'*X(:) / (2*n); */            
        mu = ddot_(&nn,Z,&inc,X,&inc) / (2.0 * n);   

        /* speed up for long steps */
        if (alpha_p + alpha_d > 1.6)
            mu *= 0.5;
        if (alpha_p + alpha_d > 1.9)
            mu *= 0.2;

        /***** objective values *****/

        /* phi = ones(n,1)'*y */                 
        *phi = ddot_(&n,b,&inc,y,&inc);

        /* psi = L(:)'*X(:) */                   
        psi = ddot_(&nn,L,&inc,X,&inc);

        gap = fabs(*phi - psi);


        /* print output */
        if (print)
            printf("%3d %11.2f %14.5f %14.5f \n",i,log10(gap),psi,*phi);

    } // end of main loop

    if (print)
        puts("*******************************************");

    /************* free memory *************/
    free(y);
    free(Z);
    free(b);    
    free(dX);
    free(dy);
    free(dZ);
    free(Zi);
    free(M);
    free(dy1);
    free(dX1);
    free(tmp);
    free(tmp2);
}


// ***********************************************
// Functions copied from serial_biqbin_general_bqp source code and converted for a python wrapper
// ***********************************************

/* global variables for BQP->MC tansformation */
double const_val = 0.0;             // constant value used in BQP -> MC tranformation. The opt. value of the original problem is const_val - OP_MC
double rho = 0.0;                   // used in BQP -> MC transformation: exact penalty parameter = 2*rho + 1
double *F_obj_data;
double *c_obj_data;


/// @brief Original post_process_BQP_input from https://github.com/Rudolfovoorg/serial_biqbin_general_bqp/blob/main/process_input.c
/// @param input_data 
/// @param adj_N number of vertices in adjacency matrix
/// @return adjacency matrix
double* post_process_BQP_input(InputData input_data, int* adj_N) {
    /*** read input file containing data for linearly constrained BQP: 
     objective: F,c, constraints: A,b ***/
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

    /* exact penalty paramter */
    double pen = ceil(2*rho + 1);

    printf("Exact penalty parameter: %.0lf\n", pen);

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

    printf("************************** MAX-CUT output **********************************\n");

    /* build adjancency matrix of the underlying graph of L_tmp */
    /* Adj = 4*(L_tmp - diag(diag(L_tmp))) */
    // we multiply by 4 to get integer values in the end!

    double *Adj;
    *adj_N = n + 1;
    alloc_matrix(Adj, n+1, double);
    for (int ii = 0; ii < n+1; ++ii) {            
        for (int jj = 0; jj < n+1; ++jj) {
            if (ii != jj)
                Adj[jj + ii * (n+1)] = 4*L_tmp[jj + ii * (n+1)];
        }
    } 
    //print_matrix(Adj,n+1,n+1);
    /* free stuff */
    free(C);
    free(tmp_X);
    free(tmp_C); 
    free(L_tmp);

    return Adj;
}

/// @brief Original read_data from https://github.com/Rudolfovoorg/serial_biqbin_general_bqp/blob/main/process_input.c
///        converted to fit the Python wrapper.
/// @param instance path to instance file
/// @param adj_N vertices in adjacency matrix
/// @return adjacency matrix
double* read_data_bqp(const char *instance, int *adj_N) {

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
    double* adj = post_process_BQP_input(input_data, adj_N);
    free(input_data.A);
    free(input_data.b);
    free(input_data.F);
    free(input_data.c);
    return adj;
}

/// @brief Convert Max-Cut solution back into BQP solution and return it to the Python wrapper
/// @param maxcut results Python dictionary
/// @param problem_size BabPbSize (1 less than number of vertices in Max-Cut graph)
/// @return BQP results in a python dictionary
py::dict read_solution_bqp(py::dict &result, int problem_size) {

    py::dict result_dict;
    result_dict["rho"] = rho;
    result_dict["const_value"] = const_val;
    
    
    // check for infeasibility of original problem
    double best_sol = result["maxcut"]["computed_val"].cast<double>();
    if (const_val - best_sol > rho) {
        printf("Original problem is INFEASIBLE due to condition const_val - opt_MC > rho: %lf > %lf.\n", const_val - best_sol, rho);
        result_dict["feasible_solution"] = false;
        return result_dict;
    }
    
    result_dict["feasible_solution"] = true;


    // output value
    // normal termination
    if (!result["meta_data"]["time_limit_reached"].cast<bool>()) {
        printf("Minimum value of the original problem is const_val - Maximum value: %.0lf\n", const_val - best_sol);
        
    } else {
            printf("TIME LIMIT REACHED.\n");
            printf("Best value = %.0lf\n", const_val - best_sol);
    }
    result_dict["computed_val"] = const_val - best_sol;
    
    // output solution
    // extern double *F_obj_data;
    // extern double *c_obj_data; 
    // max-cut has symmetrix solutions: if x is opt.sol then -x is also. 
    // We need to check both to see which one is the minimizer of the original problem

    // compute opt_value = x'Fx + c'x
    std::vector<int> solution_x = result["maxcut"]["x"].cast<std::vector<int>>();
    double opt_value = 0.0;
    for (int ii = 0; ii < problem_size; ++ii) {            
        for (int jj = 0; jj < problem_size; ++jj) {
            opt_value += (solution_x[ii])*F_obj_data[jj + ii * problem_size]*(solution_x[jj]);
        }
        opt_value += c_obj_data[ii]*(solution_x[ii]);
    }
    
    std::vector<int> bqp_sol_x;
    printf("Solution = ( ");
    if ( (int)(const_val - best_sol) == (int)(opt_value) ) {
        for (int i = 0; i < problem_size; ++i) {
            printf("%d ", solution_x[i]);
            bqp_sol_x.push_back(solution_x[i]);
        }
    }
    else {
        for (int i = 0; i < problem_size; ++i) {
            printf("%d ", 1 - solution_x[i]); // flip
            bqp_sol_x.push_back(1 - solution_x[i]);
        }
    }
    
    printf(")\n");
    result_dict["x"] = py::cast(bqp_sol_x);


    free(F_obj_data);
    free(c_obj_data);
    return result_dict;
}



/// @brief Read the bqp problem file return the adjacency matrix
/// @param instance path to instance file
/// @return adjacency matrix
py::array_t<double> read_data_bqp_python(const std::string &instance)
{
    double *adj;
    int adj_N;
    adj = read_data_bqp(instance.c_str(), &adj_N);

    return py::array_t<double>({adj_N, adj_N}, adj);
}

/// @brief Read and parse bqp json data
/// @param instance Built in DataGetterBQPJson
/// @return adjacency matrix
py::array_t<double> read_data_bqp_json_python(py::dict &instance) 
{
    double *adj;
    int adj_N;
    InputData input_data;
    input_data.n = instance["number_of_variables"].cast<int>();
    input_data.m = instance["number_of_constraints"].cast<int>();

    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html - under "vectorizing functions"
    // grab the underlying numpy array
    py::array_t<double> A_array = instance["Am"].cast<py::array_t<double>>();
    py::array_t<double> F_array = instance["Fm"].cast<py::array_t<double>>();
    py::array_t<double> c_array = instance["cm"].cast<py::array_t<double>>();
    py::array_t<double> b_array = instance["bm"].cast<py::array_t<double>>();

    // grab the buffer info
    auto A_buf = A_array.request();
    auto F_buf = F_array.request();
    auto c_buf = c_array.request();
    auto b_buf = b_array.request();

    // cast buffer info into a double* raw pointer, it is owned by Python
    input_data.A = static_cast<double*>(A_buf.ptr);
    input_data.F = static_cast<double*>(F_buf.ptr);
    input_data.c = static_cast<double*>(c_buf.ptr);
    input_data.b = static_cast<double*>(b_buf.ptr);

    // Original function
    adj = post_process_BQP_input(input_data, &adj_N);

    return py::array_t<double>({adj_N, adj_N}, adj);
}

PYBIND11_MODULE(bqp_data_processing_PLACEHOLDER, bqp)
{
    bqp.def("read_data_bqp", &read_data_bqp_python);
    bqp.def("read_data_bqp_json", &read_data_bqp_json_python);
    bqp.def("read_solution_bqp", &read_solution_bqp);
}