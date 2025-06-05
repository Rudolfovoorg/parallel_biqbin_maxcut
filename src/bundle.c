#include "biqbin.h"

#define ABS(i) ((i)>0 ? (i) : -(i))

/******************** Bundle method *********************/
/* Bundle method for solving Max-Cut SDP relaxation
 * strengthened with cutting planes.
 ******************************************************/
void bundle_method(Problem *PP, double *t, int bdl_iter) {

    extern double f;                // opt. value of SDP 
    extern double *g;               // subgradient 
    extern double *X;               // primal matrix X
    extern double *X_test;          
    extern double *X_bundle;        // bundle of matrices Xi
    extern double *F;               // bundle of <L,Xi>
    extern double *G;               // bundle of gradients
    extern double *dual_gamma;           // dual variable to cutting plane inequalities
    extern double *dgamma;          // step vector for dual_gamma
    extern double *gamma_test;      
    extern double *lambda;          // contains scalars of convex combinations of bundle matrices
    extern double *eta;             // dual variable to dual_gamma >= 0 constraint 

    // number of cutting planes
    int m = PP->NIneq + PP->NPentIneq + PP->NHeptaIneq; 
         
    int k;                                  // bundle size
    int nn = PP->n * PP->n;     
    double alpha, beta;                     // variables in blas/lapack routines
    int inc = 1;
    char TRANS;
    double f_test;                          // test value of dual function
    double del;                             // scalar for decision if serious or null step:
                                            // f - f_appr(gamma_test)

    double lmax;                            // lmax = max(lambda)
    int subtracted, next_bundle;            // for purging of the bundle    
    double temp; 
    int bdl_cnt = 0;                        // number of iterations of bundle method

    // allocate memory
    double *zeta;
    alloc_vector(zeta, MaxBundle, double);


    /***** main loop *****/
    while ( bdl_cnt < bdl_iter ) {

        // increase iteration count
        ++bdl_cnt;

        // size of bundle
        k = PP->bundle;

        /*** compute lambda, eta and dgamma ***/

        // zeta = -F - G'*dual_gamma
        dcopy_(&k, F, &inc, zeta, &inc); // copy F into zeta

        alpha = -1.0;
        beta = -1.0;
        TRANS = 'T';
        dgemv_(&TRANS, &m, &k, &alpha, G, &m, dual_gamma, &inc, &beta, zeta, &inc);

        /*** solve QP ***/
        lambda_eta(PP, zeta, G, dual_gamma, dgamma, lambda, eta, t);

        /*** make a step: gamma_test = dual_gamma + dgamma; ***/
        for (int i = 0; i < m; ++i)
            gamma_test[i] = dual_gamma[i] + dgamma[i];

        /*** evaluate function at gamma_test ***/
        f_test = fct_eval(PP, gamma_test, X_test, g);

        /* del = f - f_appr(gamma_test) = f - (F'lambda + gamma_test'*G*lambda) */
        dcopy_(&k, F, &inc, zeta, &inc); // copy F into zeta

        alpha = 1.0;
        beta = 1.0;
        TRANS = 'T';
        dgemv_(&TRANS, &m, &k, &alpha, G, &m, gamma_test, &inc, &beta, zeta, &inc);
        
        del = f - ddot_(&k, zeta, &inc, lambda, &inc);

        /* lmax = max(lambda) */ 
        lmax = -BIG_NUMBER;   
        for (int i = 0; i < k; ++i)  
            lmax = (lambda[i] > lmax) ? lambda[i] : lmax;

        /* 
         * decision whether to make SERIOUS or NULL step
         */         
        if (f - f_test > 0.05 * del) { // SERIOUS STEP
        
            // dual_gamma = gamma_test
            dcopy_(&m, gamma_test, &inc, dual_gamma, &inc);  

            // f = f_test
            f = f_test;

            /*** compute primal X (as convex combination)
             * --> retrive X from lambda and bundle matrices 
             * X = X_bundle * lambda
             ***/
            alpha = 1.0;
            beta = 0.0;
            TRANS = 'N';
            dgemv_(&TRANS, &nn, &k, &alpha, X_bundle, &nn, lambda, &inc, &beta, X, &inc);
            

            // update t
            *t *= 1.01;

            /*** purge bundle (all k elements) ***/
            subtracted = 0;
            next_bundle = 0;

            for (int i = 0; i < k; ++i) {

                // remove bundle element if lambda is small
                if (lambda[i] < 0.01 * lmax) {
                    ++subtracted;
                } 
                else {  // keep bundle element
                
                    if (i > next_bundle) {
                        F[next_bundle] = F[i];
                        dcopy_(&m, G + m*i, &inc, G + m*next_bundle, &inc);
                        dcopy_(&nn, X_bundle + nn*i, &inc, X_bundle + nn*next_bundle, &inc);
                    }
                        
                    ++next_bundle;
                }
            }

            // update bundle count
            PP->bundle -= subtracted;

            // Check bundle size for overflow (can not append more)
            if (PP->bundle == MaxBundle) {
                fprintf(stderr, "\nError: Bundle size too large! Adjust MaxBundle in biqbin.h.\n");
                MPI_Abort(MPI_COMM_WORLD,10);
            }

            /* add new information to the bundle */
            /* G = [G g]
             * X = [X X_test(:)]
             * F = [F L(:)'*X_test(:)] */
            dcopy_(&m, g, &inc, G + m * PP->bundle, &inc);
            dcopy_(&nn, X_test, &inc, X_bundle + nn * PP->bundle, &inc);

            temp = 0.0;
            for (int i = 0; i < PP->n; ++i) {
                for (int j = i; j < PP->n; ++j) {
                    if (i == j) {
                        temp += PP->L[i + i*PP->n] * X_test[i + i*PP->n];
                    }
                    else {
                        temp += 2 * PP->L[j + i*PP->n] * X_test[j + i*PP->n];
                    }
                }
            }
            F[PP->bundle] = temp;

            // update bundle count
            ++(PP->bundle);

        }
        else {  // NULL STEP

            // update t
            *t /= 1.01;

            /*** purge bundle (first k-1 elements) ***/
            subtracted = 0;
            next_bundle = 0;

            for (int i = 0; i < k-1; ++i) {

                // remove bundle element if lambda is small
                if (lambda[i] < 0.01 * lmax) {
                    ++subtracted;
                } 
                else { // keep bundle element

                    if (i > next_bundle) {
                        F[next_bundle] = F[i];
                        dcopy_(&m, G + m*i, &inc, G + m*next_bundle, &inc);
                        dcopy_(&nn, X_bundle + nn*i, &inc, X_bundle + nn*next_bundle, &inc);
                    }
                            
                    ++next_bundle;
                }
            }

            // update bundle count
            PP->bundle -= subtracted;

            // Check bundle size for overflow (can not append more)
            if (PP->bundle == MaxBundle) {
                fprintf(stderr, "\nError: Bundle size too large! Adjust MaxBundle in biqbin.h.\n");
                MPI_Abort(MPI_COMM_WORLD,10);
            }

            /* G = [G g G(:,k)]
             * X = [X X_test(:) X(:,k)]
             * F = [F L(:)'*X_test(:) F(:,k)] */

            // first copy G(:,k), X(:,k) and F(:,k) into right position
            F[PP->bundle] = F[k-1];
            dcopy_(&m, G + m * (k-1), &inc, G + m * PP->bundle, &inc);
            dcopy_(&nn, X_bundle + nn * (k-1), &inc, X_bundle + nn * PP->bundle, &inc);

            // add g, X_test and L(:)'*X_test(:)
            dcopy_(&m, g, &inc, G + m * (PP->bundle-1), &inc);
            dcopy_(&nn, X_test, &inc, X_bundle + nn * (PP->bundle-1), &inc);

            temp = 0.0;
            for (int i = 0; i < PP->n; ++i) {
                for (int j = i; j < PP->n; ++j) {
                    if (i == j) {
                        temp += PP->L[i + i*PP->n] * X_test[i + i*PP->n];
                    }
                    else {
                        temp += 2 * PP->L[j + i*PP->n] * X_test[j + i*PP->n];
                    }
                }
            }
            F[PP->bundle-1] = temp;

            // update bundle count
            ++(PP->bundle);
       
        }    
    }   

    free(zeta);

}

/*** evaluate dual function: compute its value f and subgradient g ***/
double fct_eval(const Problem *PP, double *dual_gamma, double *X, double *g) {

    int n = PP->n;
    int m = PP->NIneq + PP->NPentIneq + PP->NHeptaIneq;
    int nn = n * n;
    int inc = 1;
    double f;   // function value
 
    /* L0 = L - A^T(dual_gamma) */
    double *L0;
    alloc_matrix(L0, n, double);
    dcopy_(&nn, PP->L, &inc, L0, &inc);

    if (m > 0)
        op_Bt(PP, L0, dual_gamma);

    /* solve basic SDP relaxation */
    ipm_mc_pk(L0, n, X, &f, 0);

    if (m > 0) {
        /* compute function value f: add sum(dual_gamma) 
         * and subgradient g */
        for (int i = 0; i < m; ++i) {
            f += dual_gamma[i];
            g[i] = 1.0; 
        }

        op_B(PP, g, X);
    }

    free(L0);

    return f;
}


/* Solve lambda-eta problem
 *
 * INPUT:
 *          PP ... current subproblem
 *        zeta ... -F - G'*dual_gamma;
 *           G ... subgradients
 *       dual_gamma ... current dual_gamma      
 *           t ... penalty parameter in the bundle method
 *
 * OUTPUT:
 *      dgamma ... step direction
 * lambda, eta ... solutions of QP
 *           t ... penalty parameter in the bundle method
 */
void lambda_eta(const Problem *PP, double *zeta, double *G, double *dual_gamma, double *dgamma, double *lambda, double *eta, double *t) {

    int m = PP->NIneq + PP->NPentIneq + PP->NHeptaIneq;      // number of inequalities
    int k = PP->bundle;                     // size of bundle
    int inc = 1;
    int done = 0;                           // flag to finish
    double dir_prev = 0.0;                  // norm of previous direction
    double dir_curr = 0.0;                  // norm of current direction
    int cnt = 0;                            // counter for number of iterations

    /* set eta to 0 */
    for (int i = 0; i < m; ++i)
        eta[i] = 0.0;

    double *Q;
    alloc_matrix(Q, k, double);

    /* compute Q = t*(G'*G) */
    char UPLO = 'L';
    char TRANS = 'T';
    double beta = 0.0;
    double alpha = *t;

    dsyrk_(&UPLO, &TRANS, &k, &m, &alpha, G, &m, &beta, Q, &k);

    /* allocate variables */
    double *tmp, *c;

    alloc_vector(tmp, m, double);
    alloc_vector(c, k, double);

    /*************
     * main loop *
     *************/
    while (!done) {

        ++cnt;

        /* c = zeta - t*G'*eta */
        dcopy_(&k, zeta, &inc, c, &inc); // copy zeta into c

        TRANS = 'T';
        alpha = -(*t);
        beta = 1.0;
        dgemv_(&TRANS, &m, &k, &alpha, G, &m, eta, &inc, &beta, c, &inc);

        /* solve QP */
        solve_lambda(k,Q,c,lambda);

        /* compute tmp = G*lambda */
        TRANS = 'N';
        alpha = 1.0;
        beta = 0.0;
        dgemv_(&TRANS, &m, &k, &alpha, G, &m, lambda, &inc, &beta, tmp, &inc);

        /* eta = max(0, -dual_gamma/t + tmp) */
        /* dgamma = t*(eta-tmp) */
        for (int i = 0; i < m; ++i) {
            eta[i] = -dual_gamma[i]/(*t) + tmp[i];
            eta[i] = (eta[i] > 0) ? eta[i] : 0;
            dgamma[i] = (*t) * (eta[i] - tmp[i]);
        }

        /* dir_curr = norm(dgamma) */
        dir_curr = dnrm2_(&m, dgamma, &inc);

        /* stop when the norm of step is small enough */
        if ( ABS(dir_curr - dir_prev)/(1.0 + dir_curr) < 1e-5 )
            done = 1;
        
        if (cnt >= 50) {
            done = 1;
            *t *= 0.95;
        }
        
        dir_prev = dir_curr;
    }

    free(tmp);
    free(c);
    free(Q);
}



/*
 * solves
 * (QP)  min <lambda,c> + 1/2 <lambda, Q lambda>
 *        s.t. sum(lambda_i) = 1, lambda >= 0.
 * 
 * input: Q, c, k
 * output: lambda ... optimal solution point
 *
 * dual: 
 * (D)   max y - 1/2 <lambda, Q lambda>
 *       s.t. z = c + Q lambda - e y           
 *       lambda >= 0, z >=0 , y free
 */
void solve_lambda(int k, double *Q, double *c, double *lambda) {

    // BLAS/LAPACK variables
    int inc = 1;
    int inc_e = 0;
    char UPLO = 'L';
    double e = 1.0;     // for vector with all ones when calling ddot_
    int INFO;

    double temp;
    int cnt = 0;        // iteration counter

    /*** starting triplet ***/
    
    // lambda = e/k
    for (int i = 0; i < k; ++i)
        lambda[i] = 1.0/k;

    // Qlam = Q*lam
    double *Qlam;
    alloc_vector(Qlam, k, double);

    double alpha = 1.0;
    double beta = 0.0;
    dsymv_(&UPLO, &k, &alpha, Q, &k, lambda, &inc, &beta, Qlam, &inc);

    // tmp = Qlam + c 
    double *tmp;
    alloc_vector(tmp, k, double);
    
    double mintmp = BIG_NUMBER;

    for (int i = 0; i < k; ++i) {
        tmp[i] = Qlam[i] + c[i];
        mintmp = (tmp[i] < mintmp) ? tmp[i] : mintmp;
    }

    /*
     * if mintmp > 1
     *      y = 0; z = tmp;
     * else
     *      y = mintmp - 1; z = tmp - y;
     */
    double y;
    double dy;      // step for y
    double *z;
    alloc_vector(z, k, double);

    if (mintmp > 1) {
        y = 0.0;
        dcopy_(&k, tmp, &inc, z, &inc);
    }
    else {
        y = mintmp - 1.0;
        for (int i = 0; i < k; ++i)
            z[i] = tmp[i] - y;
    }

    // barrier parameter mu = (z' * lambda) / k
    double mu = ddot_(&k, z, &inc, lambda, &inc) / k;
    mu *= 0.5;

    // adjoint Aty = y*e (e...vector of all ones)
    double *Aty;
    alloc_vector(Aty, k, double);

    for (int i = 0; i < k; ++i)
        Aty[i] = y;

    // primal residual: 1 - e'*lambda
    double res_p = 1.0 - ddot_(&k, &e, &inc_e, lambda, &inc); 

    // initial dual cost: y - 1/2*lambda'*Qlam
    temp = ddot_(&k, lambda, &inc, Qlam, &inc);         
    double D_cost = y - 0.5 * temp;

    // intial primal cost: lambda'*c + 1/2 *lambda'*Qlam   
    double P_cost =  ddot_(&k, lambda, &inc, c, &inc) + 0.5 * temp;

    // duality gap
    double gap = P_cost - D_cost;    

    // step lengths for primal and dual variables
    double alpha_p, alpha_d;                    

    /* allocate space for other variables */
    int size = k + 1;
    int size_square = size * size;

    double *rhs, *dlambda, *dz;
    alloc_vector(rhs, size, double);
    alloc_vector(dlambda, k, double);
    alloc_vector(dz, k, double);

    // NOTE: M = [-Q-Diag(z./lambda) e; e' 0] --> only z and lambda change
    // --> M_temp = [-Q e; e' 0]
    double *M, *M_temp;
    alloc_matrix(M, size, double);
    alloc_matrix(M_temp, size, double);

    /* constuct M_temp = [-Q e; e' 0] */
    // NOTE: 0 is already there due to calloc in alloc_matrix!
    for (int i = 0; i < size; ++i) {
        for (int j = i; j < size; ++j) {

            // matrix part
            if ( (i < k) && (j < k) )
                M_temp[j + i*size] = -Q[j + i*k];

            // vector part
            else if ( (i < k) && (j == k) ) 
                M_temp[j + i*size] = 1.0;
        }
    }

    // used in dsysv
    int LWORK;
    int *IPIV;
    alloc_vector(IPIV, size, int); 
    double WORK_TEMP;   // first call to dsysv returns optimal LWORK in WORK_TEMP
    double *WORK;

    /*************
     * main loop *
     *************/
    while( ABS(gap) > 1e-5 ) {

        ++cnt;

        /* built matrix M: remove Diag(z./lambda) from diagonal (1:k) of M_temp */
        dcopy_(&size_square, M_temp, &inc, M, &inc);
        for (int i = 0; i < k; ++i)
            M[i + i*size] -= z[i]/lambda[i];

        /* 
         * right hand side (predictor):    
         * rhs(1:k) = c - AT_y + Qlam
         * rhs(k+1) = res_p
         */
        for (int i = 0; i < k; ++i)
            rhs[i] = c[i] - Aty[i] + Qlam[i];

        rhs[k] = res_p;

        /* 
         * solve linear system and extract solution
         * dw = M\rhs;  
         * dlam = dw(1:k);
         * dy = dw(k+1);
         */
        //NOTE: dw is stored in rhs!

        // to get optimal LWORK you need to run dsysv twice!
        LWORK = -1;
        dsysv_(&UPLO, &size, &inc, M, &size, IPIV, rhs, &size, &WORK_TEMP, &LWORK, &INFO);

        // allocate space for WORK
        LWORK = (int)WORK_TEMP;
        alloc_vector(WORK, LWORK, double);

        dsysv_(&UPLO, &size, &inc, M, &size, IPIV, rhs, &size, WORK, &LWORK, &INFO);

        if (INFO != 0) {
            fprintf(stderr, "%s: Problem in solving linear system \
                (line: %d).\n", __func__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD,10);
        }

        free(WORK);

        /* dlambda = rhs(1:k) and dy = rhs(k+1) */
        dy = rhs[k];
        dcopy_(&k, rhs, &inc, dlambda, &inc);

        /* corrector for dz 
         * dz = 1./lambda .* (mu*e-lambda.*z - z.*dlambda)
         */
        for (int i = 0; i < k; ++i)
            dz[i] = 1.0/lambda[i] * (mu - lambda[i]*z[i] - z[i]*dlambda[i]);


        /* find step lengths alpha_p and alpha_d s.t.
         * lambda + alpha_p * dlambda >= 0 and
         * z + alpha_d * dz >= 0 */

        // alpha_p = max(-dlambda./lam)
        // alpha_d = max(-dz./z)
        alpha_p = -BIG_NUMBER;
        alpha_d = -BIG_NUMBER;

        for (int i = 0; i < k; ++i) {
            temp = -dlambda[i]/lambda[i];
            alpha_p = (temp > alpha_p) ? temp : alpha_p;

            temp = -dz[i]/z[i];
            alpha_d = (temp > alpha_d) ? temp : alpha_d;
        }

        if (alpha_p > 0)
            alpha_p = (0.99/alpha_p < 1.0) ? 0.99/alpha_p : 1.0;
        else
            alpha_p = 1.0;

        if (alpha_d > 0)
            alpha_d = (0.99/alpha_d < 1.0) ? 0.99/alpha_d : 1.0;
        else
            alpha_d = 1.0;

        /**** update ****/
        /* lambda = lambda + alpha_p * dlambda
                y = y + alpha_d * dy
                z = z + alpha_d * dz
         */     
        y += alpha_d * dy;
        daxpy_(&k, &alpha_p, dlambda, &inc, lambda, &inc);
        daxpy_(&k, &alpha_d, dz, &inc, z, &inc);

        // adjoint
        for (int i = 0; i < k; ++i)
            Aty[i] = y;

        // primal residual
        res_p = 1 - ddot_(&k, lambda, &inc, &e, &inc_e);

        // Qlam = Q*lambda
        alpha = 1.0;
        beta = 0.0;
        dsymv_(&UPLO, &k, &alpha, Q, &k, lambda, &inc, &beta, Qlam, &inc);

        /* mu = (z' * lambda) / k 
         * mu = .4*mu
         */
        mu = ddot_(&k, lambda, &inc, z, &inc) / k;
        mu *= 0.4;

        if (alpha_p + alpha_d > 1.8)
            mu *= 0.2; 


        // dual cost: y - 1/2*lambda'*Qlam
        temp = ddot_(&k, lambda, &inc, Qlam, &inc);         
        D_cost = y - 0.5 * temp;

        // primal cost: lam'*c + 1/2 *lam'*Qlam   
        P_cost =  ddot_(&k, lambda, &inc, c, &inc) + 0.5 * temp;

        // duality gap
        gap = P_cost - D_cost; 

        if (cnt > 30)
            break; 
    } 

    free(Qlam);
    free(tmp);
    free(z);
    free(Aty);
    free(rhs);
    free(dlambda);
    free(dz);
    free(M);
    free(M_temp);
    free(IPIV);
}
