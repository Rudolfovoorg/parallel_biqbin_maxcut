#include <stdio.h>

#include "biqbin.h"
#include "global_var.h"

extern BabSolution *BabSol;     // global solution of B&B algorithm defined in heap.c
extern GlobalVariables *globals;
void allocMemory(void) {
    /* 
     * SP, SP->n, SP->L, PP, PP->n and PP->L 
     * are all allocated and defined in readData (process_input.c),
     * before this function is called
     */
    int N = globals->SP->n;

    /* triangle inequalities */
    alloc_vector(globals->Cuts, MaxTriIneqAdded, Triangle_Inequality);
    alloc_vector(globals->List, params.TriIneq, Triangle_Inequality);

    /* pentagonal inequalities */
    alloc_vector(globals->Pent_Cuts, MaxPentIneqAdded, Pentagonal_Inequality);
    alloc_vector(globals->Pent_List, params.PentIneq, Pentagonal_Inequality);

    /* heptagonal inequalities */
    alloc_vector(globals->Hepta_Cuts, MaxHeptaIneqAdded, Heptagonal_Inequality);
    alloc_vector(globals->Hepta_List, params.HeptaIneq, Heptagonal_Inequality);

    /* primal and dual variables */
    alloc_matrix(globals->X, N, double);
    alloc_matrix(globals->Z, N, double);
    alloc_vector(globals->X_bundle, N * N * MaxBundle, double);
    alloc_matrix(globals->X_test, N, double);
    alloc_vector(globals->dual_gamma, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double);
    alloc_vector(globals->dgamma, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double);
    alloc_vector(globals->gamma_test, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double);
    alloc_vector(globals->lambda, MaxBundle, double);
    alloc_vector(globals->eta, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double);
    alloc_vector(globals->F, MaxBundle, double);
    alloc_vector(globals->G, (MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded) * MaxBundle, double);   
    alloc_vector(globals->g, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double); 
}


void freeMemory(void) {

    free(globals->SP->L);
    free(globals->SP);
    free(globals->PP->L);
    free(globals->PP);

    free(globals->Cuts);
    free(globals->List);

    free(globals->Pent_Cuts);
    free(globals->Pent_List);

    free(globals->Hepta_Cuts);
    free(globals->Hepta_List);

    free(globals->X);
    free(globals->Z);
    free(globals->X_bundle);
    free(globals->X_test);
    free(globals->dual_gamma);
    free(globals->dgamma);
    free(globals->gamma_test);
    free(globals->lambda);
    free(globals->eta);
    free(globals->F);
    free(globals->G);
    free(globals->g);

    free(globals);
    free(BabSol);
}
