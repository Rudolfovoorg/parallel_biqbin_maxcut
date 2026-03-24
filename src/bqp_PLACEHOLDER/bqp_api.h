#ifndef BQP_API_H
#define BQP_API_H

#ifdef __cplusplus
extern "C"
{
#endif

    /****** BLAS  ******/
    // level 1 blas
    extern void dscal_(const int *n, const double *alpha, double *X, const int *inc);
    extern void dcopy_(const int *n, const double *X, const int *incx, double *Y, const int *incy);
    extern double dnrm2_(const int *n, const double *x, const int *incx);
    extern void daxpy_(const int *n, const double *alpha, const double *X, const int *incx, double *Y, const int *incy);
    extern double ddot_(const int *n, const double *X, const int *incx, const double *Y, const int *incy);

    // level 2 blas
    extern void dsymv_(const char *uplo, const int *n, const double *alpha, const double *A, const int *lda, const double *x,
                       const int *incx, const double *beta, double *y, const int *incy);
    extern void dgemv_(const char *uplo, const int *m, const int *n, const double *alpha, const double *A, const int *lda,
                       const double *X, const int *incx, const double *beta, double *Y, const int *incy);
    extern void dsyr_(const char *uplo, const int *n, const double *alpha, const double *x, const int *incx, double *A, const int *lda);

    // level 3 blas
    extern void dsymm_(const char *side, const char *uplo, const int *m, const int *n, const double *alpha, const double *A,
                       const int *lda, const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
    extern void dsyrk_(const char *UPLO, const char *TRANS, const int *N, const int *K, const double *ALPHA, const double *A,
                       const int *LDA, const double *BETA, double *C, const int *LDC);
    extern void dgemm_(const char *transa, const char *transb, const int *l, const int *n, const int *m, const double *alpha,
                       const void *a, const int *lda, const void *b, const int *ldb, const double *beta, void *c, const int *ldc);

    /****** LAPACK  ******/

    // computes Cholesky factorization of positive definite matrix
    extern void dpotrf_(const char *uplo, const int *n, double *X, const int *lda, int *info);

    // computes the inverse of a real symmetric positive definite
    // matrix  using the Cholesky factorization
    extern void dpotri_(const char *uplo, const int *n, double *X, const int *lda, int *info);

    // computes solution to a real system of linear equations with symmetrix matrix
    extern void dsysv_(const char *uplo, const int *n, const int *nrhs, double *A, const int *lda, int *ipiv, double *B,
                       const int *ldb, double *work, const int *lwork, int *info);

    // computes solution to a real system of linear equations with positive definite matrix
    extern void dposv_(const char *uplo, const int *n, const int *nrhs, double *A, const int *lda, double *B, const int *ldb, int *info);

// macro to handle the errors in the input reading
#define READING_ERROR(file, cond, message)          \
    if ((cond))                                     \
    {                                               \
        fprintf(stderr, "\nError: " #message "\n"); \
        fclose(file);                               \
        exit(1);                                    \
    }

// macro to handle the errors in the input reading
#define BQP_READING_ERROR(file, cond, message, ...)                                       \
    if ((cond))                                                                           \
    {                                                                                     \
        fprintf(stderr, "\nError reading input file %s at line %d.", instance, line_cnt); \
        fprintf(stderr, "\n" #message);                                                   \
        fprintf(stderr, "\n" __VA_ARGS__);                                                \
        fclose(file);                                                                     \
        exit(1);                                                                          \
    }

/* macros for allocating vectors and matrices */
#define alloc_vector(var, size, type)                                                       \
    var = (type *)calloc((size), sizeof(type));                                             \
    if (var == NULL)                                                                        \
    {                                                                                       \
        fprintf(stderr,                                                                     \
                "\nError: Memory allocation problem for variable " #var " in %s line %d\n", \
                __FILE__, __LINE__);                                                        \
        exit(1);                                                                            \
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
