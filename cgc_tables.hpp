/**
 * cgc_tables.hpp — Standalone CGC factor table builder.
 *
 * Provides makeup_clebsch_gordan_coefficient_tensors() as a free function,
 * independent of the dmrg_su2 class.  All CGC contractions use Dense<double>.
 *
 * Corresponds to dmrg_su2::makeup_clebsch_gordan_coefficient_tensors()
 * in SU2_real_mpi/j1j2longrange/dmrg_su2.cpp.
 */

#pragma once
#include <cstddef>

// ── CGCTables ─────────────────────────────────────────────────────────────────
// Holds all factor tables produced by the CGC assembly routine.
// All arrays are heap-allocated; the destructor frees them.

struct CGCTables {
  int physpn;       // physical spin (2*S, integer)
  int max_angm;     // maximum angular momentum index
  int physpn2;      // physpn*2+1
  int physpnsqr;    // physpn2 * physpn2

  // fac_hamilt_vec[i][j][k]
  double ***fac_hamilt_vec;

  // fac_operator_onsite_left/rght[i][j][k][l][physpn0 + physpn1*physpn2]
  double *****fac_operator_onsite_left;
  double *****fac_operator_onsite_rght;

  // fac_operator_transformation_left/rght[i][j][k][l][m][physpn0]
  double ******fac_operator_transformation_left;
  double ******fac_operator_transformation_rght;

  // fac_operator_pairup_left/rght[i][j][k][l][physpn0 + physpn1*physpn2]
  double *****fac_operator_pairup_left;
  double *****fac_operator_pairup_rght;

  // fac_permutation_left/rght[i][j][k][l][m]  (m < 8)
  double *****fac_permutation_left;
  double *****fac_permutation_rght;

  CGCTables();
  ~CGCTables();

  // Non-copyable
  CGCTables(const CGCTables&)            = delete;
  CGCTables& operator=(const CGCTables&) = delete;
};

// ── Entry point ───────────────────────────────────────────────────────────────
// Computes all CGC factor tables for spin physpn and angular-momentum cutoff
// max_angm.  On return, tables.fac_* arrays are fully populated.
void makeup_clebsch_gordan_coefficient_tensors(int physpn, int max_angm, CGCTables& tables);
