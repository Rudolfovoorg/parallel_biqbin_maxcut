#include <math.h>

#include "biqbin.h"

extern Triangle_Inequality *Cuts;           // vector of triangle inequality constraints
extern Pentagonal_Inequality *Pent_Cuts;    // vector of pentagonal inequality constraints
extern Heptagonal_Inequality *Hepta_Cuts;   // vector of heptagonal inequality constraints


/***************** diag *********************/
/* 
 * operator diag: save diagonal of matrix X in vector y
 */
void diag(const double *X, double *y, int n) {

    for (int i = 0; i < n; ++i) {
        y[i] = X[i + i * n];
    }
}


/***************** Diag *********************/
/* 
 * operator Diag: make diagonal matrix X from vector y
 */
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


/*** IMPORTANT NOTE
 *  Since gamma is nonnegative dual multiplier we need to save
 *  cutting plane <H, X(pent,pent)> >= 1 as -0.5(X(pent,pent) without diagonal termn) <= 1!!
 *  Inequality has to be in the form B(X) <= 1!!!!
 *  Similarly for heptagonal inequalities: -1/3(X(hept,hept) without diagonal termn) <= 1!! 
***/

/***************** op_B *********************/
/*
 * computes y = y - B(X), where operator B
 * corresponds to cutting planes: triangle, pentagonal and heptagonal inequalities
 */
void op_B(const Problem *P, double *y, const double *X) {

    int N = P->n;
    int type, ii, jj, kk, ll, mm, nn, oo;
    
    /* triangle inequalities */
    for (int ineq = 0; ineq < P->NIneq; ++ineq) {

        type = Cuts[ineq].type;
        ii   = Cuts[ineq].i;
        jj   = Cuts[ineq].j;
        kk   = Cuts[ineq].k;

        switch (type) {
            
            case 1:
                y[ineq] -= (-X[ii + jj * N] - X[ii + kk * N] - X[jj + kk * N]);
                break;
            case 2:
                y[ineq] -= (-X[ii + jj * N] + X[ii + kk * N] + X[jj + kk * N]);
                break;
            case 3:
                y[ineq] -= (X[ii + jj * N] - X[ii + kk * N] + X[jj + kk * N]);
                break;
            default: //type == 4
                y[ineq] -= (X[ii + jj * N] + X[ii + kk * N] - X[jj + kk * N]);
        }
    }    

    /* pentagonal inequalities */
    for (int ineq = 0; ineq < P->NPentIneq; ++ineq) {

        type = Pent_Cuts[ineq].type;
        ii   = Pent_Cuts[ineq].permutation[0];
        jj   = Pent_Cuts[ineq].permutation[1];
        kk   = Pent_Cuts[ineq].permutation[2];
        ll   = Pent_Cuts[ineq].permutation[3];
        mm   = Pent_Cuts[ineq].permutation[4];

        switch (type) {

            case 1: // H1 = ee^T, where e is vector of all ones 
                y[P->NIneq + ineq] -= 0.5 * (-X[ii + jj * N] - X[ii + kk * N] - X[ii + ll * N] - X[ii + mm * N] - X[jj + kk * N] - X[jj + ll * N] - X[jj + mm * N] - X[kk + ll * N] - X[kk + mm * N] - X[ll + mm * N] );
                break;
            case 2: // e[0] = -1; H2 = ee^T
                y[P->NIneq + ineq] -= 0.5 * (X[ii + jj * N] + X[ii + kk * N] + X[ii + ll * N] + X[ii + mm * N] - X[jj + kk * N] - X[jj + ll * N] - X[jj + mm * N] - X[kk + ll * N] - X[kk + mm * N] - X[ll + mm * N] );
                break;
            default: // type = 3; // e[0] = -1, e[1] = -1; H3 = ee^T
                y[P->NIneq + ineq] -= 0.5 * (-X[ii + jj * N] + X[ii + kk * N] + X[ii + ll * N] + X[ii + mm * N] + X[jj + kk * N] + X[jj + ll * N] + X[jj + mm * N] - X[kk + ll * N] - X[kk + mm * N] - X[ll + mm * N] );                    
        }

    }

    /* heptagonal inequalities */
    for (int ineq = 0; ineq < P->NHeptaIneq; ++ineq) {

        type = Hepta_Cuts[ineq].type;
        ii   = Hepta_Cuts[ineq].permutation[0];
        jj   = Hepta_Cuts[ineq].permutation[1];
        kk   = Hepta_Cuts[ineq].permutation[2];
        ll   = Hepta_Cuts[ineq].permutation[3];
        mm   = Hepta_Cuts[ineq].permutation[4]; 
        nn   = Hepta_Cuts[ineq].permutation[5];
        oo   = Hepta_Cuts[ineq].permutation[6];   

        switch (type) {

            case 1: // H1 = ee^T, where e is vector of all ones 
                y[P->NIneq + P->NPentIneq + ineq] -= 1.0/3.0 * (-X[ii + jj * N] - X[ii + kk * N] - X[ii + ll * N] - X[ii + mm * N] - X[ii + nn * N] - X[ii + oo * N] - X[jj + kk * N] - X[jj + ll * N] - X[jj + mm * N] - X [jj + nn * N] - X[jj + oo * N] - X[kk + ll * N] - X[kk + mm * N] - X[kk + nn * N] - X[kk + oo * N] - X[ll + mm * N] - X[ll + nn * N] - X[ll + oo * N] - X[mm + nn * N] - X[mm + oo * N] - X[nn + oo * N]);
                break;
            case 2: // e[0] = -1; H2 = ee^T
                y[P->NIneq + P->NPentIneq + ineq] -= 1.0/3.0 * (X[ii + jj * N] + X[ii + kk * N] + X[ii + ll * N] + X[ii + mm * N] + X[ii + nn * N] + X[ii + oo * N] - X[jj + kk * N] - X[jj + ll * N] - X[jj + mm * N] - X[jj + nn * N] - X[jj + oo * N] - X[kk + ll * N] - X[kk + mm * N] - X[kk + nn * N] - X[kk + oo * N] - X[ll + mm * N] - X[ll + nn * N] - X[ll + oo * N] - X[mm + nn * N] - X[mm + oo * N] - X[nn + oo * N]);
                break;    
            case 3: // H1 = ee^T, where e is vector of all ones 
                y[P->NIneq + P->NPentIneq + ineq] -= 1.0/3.0 * (-X[ii + jj * N] + X[ii + kk * N] + X[ii + ll * N] + X[ii + mm * N] + X[ii + nn * N] + X[ii + oo * N] + X[jj + kk * N] + X[jj + ll * N] + X[jj + mm * N] + X[jj + nn * N] + X[jj + oo * N] - X[kk + ll * N] - X[kk + mm * N] - X[kk + nn * N] - X[kk + oo * N] - X[ll + mm * N] - X[ll + nn * N] - X[ll + oo * N] - X[mm + nn * N] - X[mm + oo * N] - X[nn + oo * N]);
                break;   
            default: // e[0] = -1, e[1] = -1, e[2] = -1; H4 = ee^T
                y[P->NIneq + P->NPentIneq + ineq] -= 1.0/3.0 * (-X[ii + jj * N] - X[ii + kk * N] + X[ii + ll * N] + X[ii + mm * N] + X[ii + nn * N] + X[ii + oo * N] - X[jj + kk * N] + X[jj + ll * N] + X[jj + mm * N] + X [jj + nn * N] + X[jj + oo * N] + X[kk + ll * N] + X[kk + mm * N] + X[kk + nn * N] + X[kk + oo * N] - X[ll + mm * N] - X[ll + nn * N] - X[ll + oo * N] - X[mm + nn * N] - X[mm + oo * N] - X[nn + oo * N]);   
        }  
    }
              
}


/***************** op_Bt *********************/
/*
 * computes X = X - Bt(t), where operator B
 * corresponds to cutting planes: triangle, pentagonal and heptagonal inequalities
 */
void op_Bt(const Problem *P, double *X, const double *tt) {

    int N = P->n;
    int type, ii, jj, kk, ll, mm, nn, oo;
    double temp;

    /***** triangle inequalities *****/ 
    for (int ineq = 0; ineq < P->NIneq; ++ineq) {

        type = Cuts[ineq].type;
        ii   = Cuts[ineq].i;
        jj   = Cuts[ineq].j;
        kk   = Cuts[ineq].k;

        temp = 0.5 * tt[ineq];

        switch (type) {
            
            case 1:
                X[ii + jj * N] += temp; 
                X[ii + kk * N] += temp; 
                X[jj + kk * N] += temp; 
                X[jj + ii * N] += temp;
                X[kk + ii * N] += temp;
                X[kk + jj * N] += temp;
                break;
            case 2:
                X[ii + jj * N] += temp; 
                X[ii + kk * N] -= temp; 
                X[jj + kk * N] -= temp; 
                X[jj + ii * N] += temp;
                X[kk + ii * N] -= temp;
                X[kk + jj * N] -= temp;
                break;
            case 3:
                X[ii + jj * N] -= temp; 
                X[ii + kk * N] += temp; 
                X[jj + kk * N] -= temp; 
                X[jj + ii * N] -= temp;
                X[kk + ii * N] += temp;
                X[kk + jj * N] -= temp;
                break;
            default: //type == 4
                X[ii + jj * N] -= temp; 
                X[ii + kk * N] -= temp; 
                X[jj + kk * N] += temp; 
                X[jj + ii * N] -= temp;
                X[kk + ii * N] -= temp;
                X[kk + jj * N] += temp;
        }
    }

    /***** pentagonal inequalities *****/
    for (int ineq = 0; ineq < P->NPentIneq; ++ineq) {

        type = Pent_Cuts[ineq].type;
        ii   = Pent_Cuts[ineq].permutation[0];
        jj   = Pent_Cuts[ineq].permutation[1];
        kk   = Pent_Cuts[ineq].permutation[2];
        ll   = Pent_Cuts[ineq].permutation[3];
        mm   = Pent_Cuts[ineq].permutation[4];

        temp = 0.25 * tt[P->NIneq + ineq]; // 0.5 due to symmetry and 0.5 due to pentagonal inequality -0.5*(...) <= 1 

        switch (type) {
            case 1:
                X[ii + jj * N] += temp; 
                X[ii + kk * N] += temp; 
                X[ii + ll * N] += temp;
                X[ii + mm * N] += temp;
                X[jj + kk * N] += temp;
                X[jj + ll * N] += temp;
                X[jj + mm * N] += temp;
                X[kk + ll * N] += temp;
                X[kk + mm * N] += temp;
                X[ll + mm * N] += temp;

                X[jj + ii * N] += temp; 
                X[kk + ii * N] += temp; 
                X[ll + ii * N] += temp;
                X[mm + ii * N] += temp;
                X[kk + jj * N] += temp;
                X[ll + jj * N] += temp;
                X[mm + jj * N] += temp;
                X[ll + kk * N] += temp;
                X[mm + kk * N] += temp;
                X[mm + ll * N] += temp;
                break;

            case 2:
                X[ii + jj * N] -= temp; 
                X[ii + kk * N] -= temp; 
                X[ii + ll * N] -= temp;
                X[ii + mm * N] -= temp;
                X[jj + kk * N] += temp;
                X[jj + ll * N] += temp;
                X[jj + mm * N] += temp;
                X[kk + ll * N] += temp;
                X[kk + mm * N] += temp;
                X[ll + mm * N] += temp;

                X[jj + ii * N] -= temp; 
                X[kk + ii * N] -= temp; 
                X[ll + ii * N] -= temp;
                X[mm + ii * N] -= temp;
                X[kk + jj * N] += temp;
                X[ll + jj * N] += temp;
                X[mm + jj * N] += temp;
                X[ll + kk * N] += temp;
                X[mm + kk * N] += temp;
                X[mm + ll * N] += temp;
                break;    

            case 3:
                X[ii + jj * N] += temp; 
                X[ii + kk * N] -= temp; 
                X[ii + ll * N] -= temp;
                X[ii + mm * N] -= temp;
                X[jj + kk * N] -= temp;
                X[jj + ll * N] -= temp;
                X[jj + mm * N] -= temp;
                X[kk + ll * N] += temp;
                X[kk + mm * N] += temp;
                X[ll + mm * N] += temp;

                X[jj + ii * N] += temp; 
                X[kk + ii * N] -= temp; 
                X[ll + ii * N] -= temp;
                X[mm + ii * N] -= temp;
                X[kk + jj * N] -= temp;
                X[ll + jj * N] -= temp;
                X[mm + jj * N] -= temp;
                X[ll + kk * N] += temp;
                X[mm + kk * N] += temp;
                X[mm + ll * N] += temp;
                break;    
        }

    }    

    /***** heptagonal inequalities *****/
    for (int ineq = 0; ineq < P->NHeptaIneq; ++ineq) {

        type = Hepta_Cuts[ineq].type;
        ii   = Hepta_Cuts[ineq].permutation[0];
        jj   = Hepta_Cuts[ineq].permutation[1];
        kk   = Hepta_Cuts[ineq].permutation[2];
        ll   = Hepta_Cuts[ineq].permutation[3];
        mm   = Hepta_Cuts[ineq].permutation[4]; 
        nn   = Hepta_Cuts[ineq].permutation[5];
        oo   = Hepta_Cuts[ineq].permutation[6];

        temp = 1.0/6.0 * tt[P->NIneq + P->NPentIneq + ineq]; // 0.5 due to symmetry and 1/3 due to heptagonal inequality -1/3*(...) <= 1 

        switch (type) {
            case 1:
                X[ii + jj * N] += temp; 
                X[ii + kk * N] += temp; 
                X[ii + ll * N] += temp;
                X[ii + mm * N] += temp;
                X[ii + nn * N] += temp;
                X[ii + oo * N] += temp;
                X[jj + kk * N] += temp;
                X[jj + ll * N] += temp;
                X[jj + mm * N] += temp;
                X[jj + nn * N] += temp;
                X[jj + oo * N] += temp;
                X[kk + ll * N] += temp;
                X[kk + mm * N] += temp;
                X[kk + nn * N] += temp;
                X[kk + oo * N] += temp;
                X[ll + mm * N] += temp;
                X[ll + nn * N] += temp;
                X[ll + oo * N] += temp;
                X[mm + nn * N] += temp;
                X[mm + oo * N] += temp;
                X[nn + oo * N] += temp;

                X[jj + ii * N] += temp; 
                X[kk + ii * N] += temp; 
                X[ll + ii * N] += temp;
                X[mm + ii * N] += temp;
                X[nn + ii * N] += temp;
                X[oo + ii * N] += temp;
                X[kk + jj * N] += temp;
                X[ll + jj * N] += temp;
                X[mm + jj * N] += temp;
                X[nn + jj * N] += temp;
                X[oo + jj * N] += temp;
                X[ll + kk * N] += temp;
                X[mm + kk * N] += temp;
                X[nn + kk * N] += temp;
                X[oo + kk * N] += temp;
                X[mm + ll * N] += temp;
                X[nn + ll * N] += temp;
                X[oo + ll * N] += temp;
                X[nn + mm * N] += temp;
                X[oo + mm * N] += temp;
                X[oo + nn * N] += temp;
                break;

            case 2:
                X[ii + jj * N] -= temp; 
                X[ii + kk * N] -= temp; 
                X[ii + ll * N] -= temp;
                X[ii + mm * N] -= temp;
                X[ii + nn * N] -= temp;
                X[ii + oo * N] -= temp;
                X[jj + kk * N] += temp;
                X[jj + ll * N] += temp;
                X[jj + mm * N] += temp;
                X[jj + nn * N] += temp;
                X[jj + oo * N] += temp;
                X[kk + ll * N] += temp;
                X[kk + mm * N] += temp;
                X[kk + nn * N] += temp;
                X[kk + oo * N] += temp;
                X[ll + mm * N] += temp;
                X[ll + nn * N] += temp;
                X[ll + oo * N] += temp;
                X[mm + nn * N] += temp;
                X[mm + oo * N] += temp;
                X[nn + oo * N] += temp;

                X[jj + ii * N] -= temp; 
                X[kk + ii * N] -= temp; 
                X[ll + ii * N] -= temp;
                X[mm + ii * N] -= temp;
                X[nn + ii * N] -= temp;
                X[oo + ii * N] -= temp;
                X[kk + jj * N] += temp;
                X[ll + jj * N] += temp;
                X[mm + jj * N] += temp;
                X[nn + jj * N] += temp;
                X[oo + jj * N] += temp;
                X[ll + kk * N] += temp;
                X[mm + kk * N] += temp;
                X[nn + kk * N] += temp;
                X[oo + kk * N] += temp;
                X[mm + ll * N] += temp;
                X[nn + ll * N] += temp;
                X[oo + ll * N] += temp;
                X[nn + mm * N] += temp;
                X[oo + mm * N] += temp;
                X[oo + nn * N] += temp;
                break;  

            case 3:
                X[ii + jj * N] += temp; 
                X[ii + kk * N] -= temp; 
                X[ii + ll * N] -= temp;
                X[ii + mm * N] -= temp;
                X[ii + nn * N] -= temp;
                X[ii + oo * N] -= temp;
                X[jj + kk * N] -= temp;
                X[jj + ll * N] -= temp;
                X[jj + mm * N] -= temp;
                X[jj + nn * N] -= temp;
                X[jj + oo * N] -= temp;
                X[kk + ll * N] += temp;
                X[kk + mm * N] += temp;
                X[kk + nn * N] += temp;
                X[kk + oo * N] += temp;
                X[ll + mm * N] += temp;
                X[ll + nn * N] += temp;
                X[ll + oo * N] += temp;
                X[mm + nn * N] += temp;
                X[mm + oo * N] += temp;
                X[nn + oo * N] += temp;

                X[jj + ii * N] += temp; 
                X[kk + ii * N] -= temp; 
                X[ll + ii * N] -= temp;
                X[mm + ii * N] -= temp;
                X[nn + ii * N] -= temp;
                X[oo + ii * N] -= temp;
                X[kk + jj * N] -= temp;
                X[ll + jj * N] -= temp;
                X[mm + jj * N] -= temp;
                X[nn + jj * N] -= temp;
                X[oo + jj * N] -= temp;
                X[ll + kk * N] += temp;
                X[mm + kk * N] += temp;
                X[nn + kk * N] += temp;
                X[oo + kk * N] += temp;
                X[mm + ll * N] += temp;
                X[nn + ll * N] += temp;
                X[oo + ll * N] += temp;
                X[nn + mm * N] += temp;
                X[oo + mm * N] += temp;
                X[oo + nn * N] += temp;
                break;      

            case 4:
                X[ii + jj * N] += temp; 
                X[ii + kk * N] += temp; 
                X[ii + ll * N] -= temp;
                X[ii + mm * N] -= temp;
                X[ii + nn * N] -= temp;
                X[ii + oo * N] -= temp;
                X[jj + kk * N] += temp;
                X[jj + ll * N] -= temp;
                X[jj + mm * N] -= temp;
                X[jj + nn * N] -= temp;
                X[jj + oo * N] -= temp;
                X[kk + ll * N] -= temp;
                X[kk + mm * N] -= temp;
                X[kk + nn * N] -= temp;
                X[kk + oo * N] -= temp;
                X[ll + mm * N] += temp;
                X[ll + nn * N] += temp;
                X[ll + oo * N] += temp;
                X[mm + nn * N] += temp;
                X[mm + oo * N] += temp;
                X[nn + oo * N] += temp;

                X[jj + ii * N] += temp; 
                X[kk + ii * N] += temp; 
                X[ll + ii * N] -= temp;
                X[mm + ii * N] -= temp;
                X[nn + ii * N] -= temp;
                X[oo + ii * N] -= temp;
                X[kk + jj * N] += temp;
                X[ll + jj * N] -= temp;
                X[mm + jj * N] -= temp;
                X[nn + jj * N] -= temp;
                X[oo + jj * N] -= temp;
                X[ll + kk * N] -= temp;
                X[mm + kk * N] -= temp;
                X[nn + kk * N] -= temp;
                X[oo + kk * N] -= temp;
                X[mm + ll * N] += temp;
                X[nn + ll * N] += temp;
                X[oo + ll * N] += temp;
                X[nn + mm * N] += temp;
                X[oo + mm * N] += temp;
                X[oo + nn * N] += temp;
                break;    

        }        
    }    
}
