#ifndef BQP_API_H
#define BQP_API_H

#ifdef __cplusplus
extern "C" {
#endif

/****** BLAS  ******/
// level 1 blas
extern void dscal_(int *n, double *alpha, double *X, int *inc);
extern void dcopy_(int *n, double *X, int *incx, double *Y, int *incy);
extern double dnrm2_(int *n, double *x, int *incx);
extern void daxpy_(int *n, double *alpha, double *X, int *incx, double *Y, int *incy);
extern double ddot_(int *n, double *X, int *incx, double *Y, int *incy);

// level 2 blas
extern void dsymv_(char *uplo, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void dgemv_(char *uplo, int *m, int *n, double *alpha, double *A, int *lda, double *X, int *incx, double *beta, double *Y, int *incy);
extern void dsyr_(char *uplo, int *n, double *alpha, double *x, int *incx, double *A, int *lda);

// level 3 blas
extern void dsymm_(char *side, char *uplo, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
extern void dsyrk_(char *UPLO, char *TRANS, int *N, int *K, double *ALPHA, double *A, int *LDA, double *BETA, double *C, int *LDC);
extern void dgemm_(char *transa, char *transb, int *l, int *n, int *m, double *alpha, const void *a, int *lda, void *b, int *ldb, double *beta, void *c, int *ldc);

/****** LAPACK  ******/

// computes Cholesky factorization of positive definite matrix
extern void dpotrf_(char *uplo, int *n, double *X, int *lda, int *info);

// computes the inverse of a real symmetric positive definite
// matrix  using the Cholesky factorization
extern void dpotri_(char *uplo, int *n, double *X, int *lda, int *info);

// computes solution to a real system of linear equations with symmetrix matrix
extern void dsysv_(char *uplo, int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, double *work, int *lwork, int *info);

// computes solution to a real system of linear equations with positive definite matrix
extern void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info);

// macro to handle the errors in the input reading
#define READING_ERROR(file,cond,message)\
        if ((cond)) {\
            fprintf(stderr, "\nError: "#message"\n");\
            fclose(file);\
            exit(1);\
        }

// macro to handle the errors in the input reading
#define BQP_READING_ERROR(file,cond,message,...)\
        if ((cond)) {\
            fprintf(stderr, "\nError reading input file %s at line %d.", instance, line_cnt); \
            fprintf(stderr, "\n"#message); \
            fprintf(stderr, "\n" __VA_ARGS__);\
            fclose(file);\
            exit(1);\
}

/* macros for allocating vectors and matrices */
#define alloc_vector(var, size, type)                                                       \
    var = (type *)calloc((size), sizeof(type));                                             \
    if (var == NULL)                                                                        \
    {                                                                                       \
        fprintf(stderr,                                                                     \
                "\nError: Memory allocation problem for variable " #var " in %s line %d\n", \
                __FILE__, __LINE__);                                                        \
        exit(1);                                                          \
    }

#define alloc(var, type) alloc_vector(var, 1, type)
#define alloc_matrix(var, size, type) alloc_vector(var, (size) * (size), type)

/************************************************************************************************************/
/* The main problem and any subproblems are stored using the following structure. */
typedef struct Problem
{
    double *L;      // Objective matrix
    int n;          // size of L
    int NIneq;      // number of triangle inequalities
    int NPentIneq;  // number of pentagonal inequalities
    int NHeptaIneq; // number of heptagonal inequalities
    int bundle;     // size of bundle
} Problem;

typedef struct InputData
{
    int m, n;
    double *A;
    double *b;
    double *F;
    double *c;
} InputData;

#ifdef __cplusplus
}
#endif

#endif /*BQP_API */
