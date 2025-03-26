#include <stdio.h>

#include "biqbin.h"

extern BabSolution *BabSol;     // global solution of B&B algorithm defined in heap.c
extern BiqBinParameters params;

void allocMemory(GlobalVariables *globals_in) {
    /* 
     * SP, SP->n, SP->L, PP, PP->n and PP->L 
     * are all allocated and defined in readData (process_input.c),
     * before this function is called
     */
    int num_vertices = globals_in->SP->n;

    /* triangle inequalities */
    alloc_vector(globals_in->Cuts, MaxTriIneqAdded, Triangle_Inequality);
    alloc_vector(globals_in->List, params.TriIneq, Triangle_Inequality);

    /* pentagonal inequalities */
    alloc_vector(globals_in->Pent_Cuts, MaxPentIneqAdded, Pentagonal_Inequality);
    alloc_vector(globals_in->Pent_List, params.PentIneq, Pentagonal_Inequality);

    /* heptagonal inequalities */
    alloc_vector(globals_in->Hepta_Cuts, MaxHeptaIneqAdded, Heptagonal_Inequality);
    alloc_vector(globals_in->Hepta_List, params.HeptaIneq, Heptagonal_Inequality);

    /* primal and dual variables */
    alloc_matrix(globals_in->X, num_vertices, double);
    alloc_matrix(globals_in->Z, num_vertices, double);
    alloc_vector(globals_in->X_bundle, num_vertices * num_vertices * MaxBundle, double);
    alloc_matrix(globals_in->X_test, num_vertices, double);
    alloc_vector(globals_in->dual_gamma, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double);
    alloc_vector(globals_in->dgamma, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double);
    alloc_vector(globals_in->gamma_test, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double);
    alloc_vector(globals_in->lambda, MaxBundle, double);
    alloc_vector(globals_in->eta, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double);
    alloc_vector(globals_in->F, MaxBundle, double);
    alloc_vector(globals_in->G, (MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded) * MaxBundle, double);   
    alloc_vector(globals_in->g, MaxTriIneqAdded + MaxPentIneqAdded + MaxHeptaIneqAdded, double); 
}


void freeMemory(GlobalVariables *globals_in) {

    if (globals_in->SP) {
        free(globals_in->SP->L);
        free(globals_in->SP);
    }
    if (globals_in->PP) {
        free(globals_in->PP->L);
        free(globals_in->PP);
    }

    free(globals_in->Cuts);
    free(globals_in->List);

    free(globals_in->Pent_Cuts);
    free(globals_in->Pent_List);

    free(globals_in->Hepta_Cuts);
    free(globals_in->Hepta_List);

    free(globals_in->X);
    free(globals_in->Z);
    free(globals_in->X_bundle);
    free(globals_in->X_test);
    free(globals_in->dual_gamma);
    free(globals_in->dgamma);
    free(globals_in->gamma_test);
    free(globals_in->lambda);
    free(globals_in->eta);
    free(globals_in->F);
    free(globals_in->G);
    free(globals_in->g);

    free(globals_in);
    if (BabSol) {
        free(BabSol);
    }
}
