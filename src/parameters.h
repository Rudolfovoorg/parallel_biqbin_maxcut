#pragma once

/* Branching strategies */
#define LEAST_FRACTIONAL 0
#define MOST_FRACTIONAL 1

// BiqBin parameters and default values
#ifndef PARAM_FIELDS
#define PARAM_FIELDS                         \
    P(int, init_bundle_iter, "%d", 3)        \
    P(int, max_bundle_iter, "%d", 15)        \
    P(int, triag_iter, "%d", 5)              \
    P(int, pent_iter, "%d", 5)               \
    P(int, hept_iter, "%d", 5)               \
    P(int, max_outer_iter, "%d", 20)         \
    P(int, extra_iter, "%d", 10)             \
    P(double, violated_TriIneq, "%lf", 1e-3) \
    P(int, TriIneq, "%d", 5000)              \
    P(int, adjust_TriIneq, "%d", 1)          \
    P(int, PentIneq, "%d", 5000)             \
    P(int, HeptaIneq, "%d", 5000)            \
    P(int, Pent_Trials, "%d", 60)            \
    P(int, Hepta_Trials, "%d", 50)           \
    P(int, include_Pent, "%d", 1)            \
    P(int, include_Hepta, "%d", 1)           \
    P(int, root, "%d", 0)                    \
    P(int, use_diff, "%d", 1)                \
    P(int, time_limit, "%d", 0)              \
    P(int, branchingStrategy, "%d", MOST_FRACTIONAL)
#endif

typedef struct BiqBinParameters
{
#define P(type, name, format, def_value) type name;
    PARAM_FIELDS
#undef P
} BiqBinParameters;
