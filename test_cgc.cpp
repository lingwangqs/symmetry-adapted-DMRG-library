/**
 * test_cgc.cpp — Compare new Dense-based CGC tables with old tensor-based version.
 *
 * Compilation (from gen_mps/mps/ directory):
 *
 *   OLD_SRC = /Users/lingwang/Desktop/LingsMacPro15/Desktop/vsc/program/dmrg_new_version/SU2_real_mpi/j1j2longrange
 *
 *   g++-14 -O2 -std=c++17 -fopenmp \
 *       -I. \
 *       -I${OLD_SRC} \
 *       src/dense_inst.cpp \
 *       src/symmetry/u1.cpp \
 *       src/symmetry/su2.cpp \
 *       src/cgc_tables.cpp \
 *       ${OLD_SRC}/tensor.cpp \
 *       ${OLD_SRC}/tensor_dmrg_src.cpp \
 *       test_cgc.cpp \
 *       -o test_cgc \
 *       -lblas -llapack -lstdc++
 *
 * Note: tensor.hpp uses `using namespace std;` at global scope; this is
 *       the original file's style and does not affect compilation here.
 *
 * Global variables required by the old tensor code:
 *   int  max_dcut, myrank, psize, comm_rank
 *   double tolerance
 *   double ran_()   (random number stub)
 *   void obtain_symmetric_matrix_eigenvector*(...)  (stubs; not called here)
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>

using std::cout;
using std::endl;
using std::abs;

// ── Globals required by old tensor code ───────────────────────────────────────
// tensor.cpp defines: double tolerance=1.e-12;
//                     obtain_symmetric_matrix_eigenvector(...)
//                     obtain_symmetric_matrix_eigenvector_2(...)
// tensor.cpp/tensor_dmrg_src.cpp declare as extern:
//   int max_dcut, myrank, psize
//   int comm_rank
// We define those here:
int max_dcut  = 100;
int myrank    = 0;
int psize     = 1;
int comm_rank = 0;

// ran_() is declared extern "C" in tensor_dmrg_src.cpp
extern "C" { double ran_() { return 0.5; } }

// ── Old tensor-based CGC tables ───────────────────────────────────────────────
// Include the original tensor header.  tensor.hpp contains `using namespace std;`
// which pollutes the namespace, but is harmless for this test.
#include "tensor.hpp"   // resolves via -I${OLD_SRC}

// Free-function version of dmrg_su2::makeup_clebsch_gordan_coefficient_tensors,
// extracted verbatim from dmrg_su2.cpp with:
//   - class members physpn, comm_rank → function parameters
//   - cgc_coef_left/rght, cgc_coef_singlet, identity → local variables
//   - all fac_* tables → output parameters (allocated inside)

static void old_makeup_cgc(
    int physpn, int max_angm,
    double ***&fac_hamilt_vec,
    double *****&fac_onsite_L, double *****&fac_onsite_R,
    double ******&fac_trans_L,  double ******&fac_trans_R,
    double *****&fac_pairup_L, double *****&fac_pairup_R,
    double *****&fac_perm_L,   double *****&fac_perm_R)
{
  int i, j, k, l, m, n, physpn0, physpn1;
  int a0, a1, a2, a3;
  int physpn2   = physpn * 2 + 1;
  int physpnsqr = physpn2 * physpn2;
  tensor tmp, tmp1, tmp2;
  double nor;
  bool check1, check2;

  const int M  = max_angm;
  const int P2 = physpn2;
  const int PS = physpnsqr;

  // local CGC tensors
  tensor*** cgc_left  = new tensor**[M];
  tensor*** cgc_rght  = new tensor**[M];
  for (i = 0; i < M; ++i) {
    cgc_left[i] = new tensor*[M];
    cgc_rght[i] = new tensor*[M];
    for (j = 0; j < M; ++j) {
      cgc_left[i][j] = new tensor[M];
      cgc_rght[i][j] = new tensor[M];
    }
  }
  tensor* cgc_singlet = new tensor[M];
  tensor* ident       = new tensor[M];
  for (i = 0; i < M; ++i) {
    cgc_singlet[i].make_singlet(i);
    ident[i].make_identity(i);
  }
  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      for (k = 0; k < M; ++k)
        if (k >= abs(i - j) && k <= abs(i + j)) {
          cgc_left[i][j][k].make_cgc(i, j, k);
          cgc_rght[i][j][k].make_cgc(i, j, k);
        }

  // fac_hamilt_vec
  fac_hamilt_vec = new double**[M];
  for (i = 0; i < M; ++i) {
    fac_hamilt_vec[i] = new double*[M];
    for (j = 0; j < M; ++j) {
      fac_hamilt_vec[i][j] = new double[M];
      for (k = 0; k < M; ++k) fac_hamilt_vec[i][j][k] = 0.0;
    }
  }
  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      for (k = 0; k < M; ++k)
        if (k >= abs(i - j) && k <= abs(i + j) && (i % 2 == k % 2)) {
          tmp1.contract(cgc_left[i][j][k], 0, cgc_singlet[i], 1);
          tmp2.contract(tmp1, 0, cgc_singlet[j], 1);
          tmp1.contract(tmp2, 0, cgc_singlet[k], 1);
          fac_hamilt_vec[i][j][k] = cgc_left[i][j][k].inner_prod(tmp1) * sqrt((double)(j + 1));
        }

  // fac_onsite
  fac_onsite_L = new double****[M]; fac_onsite_R = new double****[M];
  for (i = 0; i < M; ++i) {
    fac_onsite_L[i] = new double***[M]; fac_onsite_R[i] = new double***[M];
    for (j = 0; j < M; ++j) {
      fac_onsite_L[i][j] = new double**[M]; fac_onsite_R[i][j] = new double**[M];
      for (k = 0; k < M; ++k) {
        fac_onsite_L[i][j][k] = new double*[M]; fac_onsite_R[i][j][k] = new double*[M];
        for (l = 0; l < M; ++l) {
          fac_onsite_L[i][j][k][l] = new double[PS];
          fac_onsite_R[i][j][k][l] = new double[PS];
          for (m = 0; m < PS; ++m) { fac_onsite_L[i][j][k][l][m] = 0.0; fac_onsite_R[i][j][k][l][m] = 0.0; }
        }
      }
    }
  }
  for (physpn0 = 0; physpn0 < P2; ++physpn0)
    for (physpn1 = 0; physpn1 < P2; ++physpn1)
      for (m = 0; m <= 6; m += 2)
        for (i = 0; i < M; ++i)
          for (j = 0; j < M; ++j)
            if (j >= abs(i - physpn0) && j <= abs(i + physpn0))
              for (k = 0; k < M; ++k)
                for (l = 0; l < M; ++l)
                  if (l >= abs(k - physpn1) && l <= abs(k + physpn1))
                    if (i == k && l >= abs(j - m) && l <= abs(j + m) && (j % 2 == l % 2))
                      if (physpn1 >= abs(physpn0 - m) && physpn1 <= abs(physpn0 + m)) {
                        tmp1.contract_dmrg_operator_initial(cgc_left[i][physpn0][j], cgc_left[k][physpn1][l], cgc_left[physpn0][m][physpn1], 0);
                        tmp1.is_proportional_to(cgc_left[j][m][l], fac_onsite_L[j][m][l][i][physpn0 + physpn1 * P2]);
                        tmp2.contract_dmrg_operator_initial(cgc_rght[physpn0][i][j], cgc_rght[physpn1][k][l], cgc_left[physpn0][m][physpn1], 1);
                        tmp2.is_proportional_to(cgc_left[j][m][l], fac_onsite_R[j][m][l][i][physpn0 + physpn1 * P2]);
                      }

  // fac_trans
  fac_trans_L = new double*****[M]; fac_trans_R = new double*****[M];
  for (i = 0; i < M; ++i) {
    fac_trans_L[i] = new double****[M]; fac_trans_R[i] = new double****[M];
    for (j = 0; j < M; ++j) {
      fac_trans_L[i][j] = new double***[M]; fac_trans_R[i][j] = new double***[M];
      for (k = 0; k < M; ++k) {
        fac_trans_L[i][j][k] = new double**[M]; fac_trans_R[i][j][k] = new double**[M];
        for (l = 0; l < M; ++l) {
          fac_trans_L[i][j][k][l] = new double*[M]; fac_trans_R[i][j][k][l] = new double*[M];
          for (m = 0; m < M; ++m) {
            fac_trans_L[i][j][k][l][m] = new double[P2];
            fac_trans_R[i][j][k][l][m] = new double[P2];
            for (n = 0; n < P2; ++n) { fac_trans_L[i][j][k][l][m][n] = 0.0; fac_trans_R[i][j][k][l][m][n] = 0.0; }
          }
        }
      }
    }
  }
  for (physpn0 = 0; physpn0 < P2; ++physpn0) {
    physpn1 = physpn0;
    for (m = 0; m <= 6; m += 2)
      for (i = 0; i < M; ++i)
        for (j = 0; j < M; ++j)
          if (j >= abs(i - physpn0) && j <= abs(i + physpn0))
            for (k = 0; k < M; ++k)
              for (l = 0; l < M; ++l)
                if (l >= abs(k - physpn1) && l <= abs(k + physpn1))
                  if (k >= abs(i - m) && k <= abs(i + m) && (k % 2 == i % 2) &&
                      l >= abs(j - m) && l <= (j + m) && (j % 2 == l % 2)) {
                    tmp1.contract_dmrg_operator_transformation(cgc_left[i][physpn0][j], cgc_left[k][physpn1][l], cgc_left[i][m][k], 0);
                    tmp1.is_proportional_to(cgc_left[j][m][l], fac_trans_L[j][m][l][i][k][physpn0]);
                    tmp2.contract_dmrg_operator_transformation(cgc_rght[physpn0][i][j], cgc_rght[physpn1][k][l], cgc_left[i][m][k], 1);
                    tmp2.is_proportional_to(cgc_left[j][m][l], fac_trans_R[j][m][l][i][k][physpn0]);
                  }
  }

  // fac_pairup
  fac_pairup_L = new double****[M]; fac_pairup_R = new double****[M];
  for (i = 0; i < M; ++i) {
    fac_pairup_L[i] = new double***[M]; fac_pairup_R[i] = new double***[M];
    for (j = 0; j < M; ++j) {
      fac_pairup_L[i][j] = new double**[M]; fac_pairup_R[i][j] = new double**[M];
      for (k = 0; k < M; ++k) {
        fac_pairup_L[i][j][k] = new double*[M]; fac_pairup_R[i][j][k] = new double*[M];
        for (l = 0; l < M; ++l) {
          fac_pairup_L[i][j][k][l] = new double[PS];
          fac_pairup_R[i][j][k][l] = new double[PS];
          for (m = 0; m < PS; ++m) { fac_pairup_L[i][j][k][l][m] = 0.0; fac_pairup_R[i][j][k][l][m] = 0.0; }
        }
      }
    }
  }
  for (physpn0 = 0; physpn0 < P2; ++physpn0)
    for (physpn1 = 0; physpn1 < P2; ++physpn1)
      for (m = 0; m <= 6; m += 2)
        for (i = 0; i < M; ++i)
          for (j = 0; j < M; ++j)
            if (j >= abs(i - physpn0) && j <= abs(i + physpn0))
              for (k = 0; k < M; ++k)
                for (l = 0; l < M; ++l)
                  if (l >= abs(k - physpn1) && l <= abs(k + physpn1))
                    if (k >= abs(i - m) && k <= (i + m) && j == l &&
                        physpn1 >= abs(physpn0 - m) && physpn1 <= abs(physpn0 + m)) {
                      tmp.contract(cgc_left[physpn0][m][physpn1], 1, cgc_singlet[m], 1);
                      tmp.shift(0, 2);
                      tmp *= sqrt((double)(m + 1));
                      tmp1.contract_dmrg_operator_pairup(cgc_left[i][physpn0][j], cgc_left[k][physpn1][l], cgc_left[i][m][k], tmp, 0);
                      tmp1.is_proportional_to(ident[j], fac_pairup_L[j][i][k][m][physpn0 + physpn1 * P2]);
                      tmp2.contract_dmrg_operator_pairup(cgc_rght[physpn0][i][j], cgc_rght[physpn1][k][l], cgc_left[i][m][k], tmp, 1);
                      tmp2.is_proportional_to(ident[j], fac_pairup_R[j][i][k][m][physpn0 + physpn1 * P2]);
                    }

  // fac_perm
  fac_perm_L = new double****[M]; fac_perm_R = new double****[M];
  for (i = 0; i < M; ++i) {
    fac_perm_L[i] = new double***[M]; fac_perm_R[i] = new double***[M];
    for (j = 0; j < M; ++j) {
      fac_perm_L[i][j] = new double**[M]; fac_perm_R[i][j] = new double**[M];
      for (k = 0; k < M; ++k) {
        fac_perm_L[i][j][k] = new double*[M]; fac_perm_R[i][j][k] = new double*[M];
        for (l = 0; l < M; ++l) {
          fac_perm_L[i][j][k][l] = new double[8];
          fac_perm_R[i][j][k][l] = new double[8];
          for (m = 0; m < 8; ++m) { fac_perm_L[i][j][k][l][m] = 0.0; fac_perm_R[i][j][k][l][m] = 0.0; }
        }
      }
    }
  }
  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      if (j >= abs(i - physpn) && j <= abs(i + physpn))
        for (k = 0; k < M; ++k)
          for (l = 0; l < M; ++l)
            if (l >= abs(k - physpn) && l <= abs(k + physpn))
              for (m = 0; m < 8; ++m) {
                a0 = m;
                a1 = (a0 % 2) * 2; a0 /= 2;
                a2 = (a0 % 2) * 2 + 1; a0 /= 2;
                a3 = (a0 % 2) * 2;
                if (!(a2 <= abs(a1 + physpn) && a2 >= abs(a1 - physpn) &&
                      a3 <= abs(a2 + physpn) && a3 >= abs(a2 - physpn))) continue;
                if ((i % 2 == k % 2) && (j % 2 == l % 2) &&
                    k <= abs(a1 + i) && k >= abs(a1 - i) &&
                    l <= abs(a3 + j) && l >= abs(a3 - j)) {
                  tmp1.contract_dmrg_permutation(cgc_left[i][physpn][j], cgc_left[k][physpn][l],
                    cgc_left[i][a1][k], cgc_left[physpn][a1][a2], cgc_left[physpn][a3][a2], 0);
                  if (!tmp1.is_proportional_to(cgc_left[j][a3][l], nor)) { cout << "old perm_L wrong" << endl; exit(0); }
                  fac_perm_L[i][j][k][l][m] = nor;
                }
                if ((i % 2 == k % 2) && (j % 2 == l % 2) &&
                    k <= abs(a3 + i) && k >= abs(a3 - i) &&
                    l <= abs(a1 + j) && l >= abs(a1 - j)) {
                  tmp2.contract_dmrg_permutation(cgc_rght[physpn][i][j], cgc_rght[physpn][k][l],
                    cgc_left[i][a3][k], cgc_left[a2][physpn][a1], cgc_left[a2][physpn][a3], 1);
                  if (!tmp2.is_proportional_to(cgc_left[j][a1][l], nor)) { cout << "old perm_R wrong" << endl; exit(0); }
                  fac_perm_R[i][j][k][l][m] = nor;
                }
              }

  // cleanup
  delete[] cgc_singlet; delete[] ident;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < M; ++j) { delete[] cgc_left[i][j]; delete[] cgc_rght[i][j]; }
    delete[] cgc_left[i]; delete[] cgc_rght[i];
  }
  delete[] cgc_left; delete[] cgc_rght;
}

// ── New Dense-based CGC tables ────────────────────────────────────────────────
#include "cgc_tables.hpp"

// ── Comparison utilities ──────────────────────────────────────────────────────
static int n_errors = 0;

static void compare_double(const char* name, double a, double b, double tol = 1e-10) {
  if (std::abs(a - b) > tol * (std::abs(a) + std::abs(b) + 1e-30)) {
    cout << "MISMATCH " << name << ": old=" << a << " new=" << b
         << " diff=" << std::abs(a - b) << endl;
    ++n_errors;
  }
}

static void compare_table3(const char* name, double*** A, double*** B, int M) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j)
      for (int k = 0; k < M; ++k) {
        char buf[128]; snprintf(buf, sizeof(buf), "%s[%d][%d][%d]", name, i, j, k);
        compare_double(buf, A[i][j][k], B[i][j][k]);
      }
}

static void compare_table5(const char* name, double***** A, double***** B, int M, int last) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j)
      for (int k = 0; k < M; ++k)
        for (int l = 0; l < M; ++l)
          for (int m = 0; m < last; ++m) {
            char buf[128]; snprintf(buf, sizeof(buf), "%s[%d][%d][%d][%d][%d]", name, i, j, k, l, m);
            compare_double(buf, A[i][j][k][l][m], B[i][j][k][l][m]);
          }
}

static void compare_table6(const char* name, double****** A, double****** B, int M, int last) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j)
      for (int k = 0; k < M; ++k)
        for (int l = 0; l < M; ++l)
          for (int m2 = 0; m2 < M; ++m2)
            for (int n = 0; n < last; ++n) {
              char buf[128]; snprintf(buf, sizeof(buf), "%s[%d][%d][%d][%d][%d][%d]", name, i, j, k, l, m2, n);
              compare_double(buf, A[i][j][k][l][m2][n], B[i][j][k][l][m2][n]);
            }
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
  const int physpn   = 1;   // spin-1/2: physpn=1 → physpn2=3
  // max_angm must be > 6 because the inner loops iterate m up to 6.
  const int max_angm = 8;   // angular momentum cutoff

  cout << "=== Running OLD tensor-based CGC table builder ===" << endl;
  double ***fhv_old = nullptr;
  double *****fol_old = nullptr, *****for_old = nullptr;
  double ******ftl_old = nullptr, ******ftr_old = nullptr;
  double *****fpl_old = nullptr, *****fpr_old = nullptr;
  double *****fperml_old = nullptr, *****fpermr_old = nullptr;

  old_makeup_cgc(physpn, max_angm,
    fhv_old, fol_old, for_old, ftl_old, ftr_old,
    fpl_old, fpr_old, fperml_old, fpermr_old);
  cout << "OLD done." << endl;

  cout << "=== Running NEW Dense-based CGC table builder ===" << endl;
  CGCTables nt;
  makeup_clebsch_gordan_coefficient_tensors(physpn, max_angm, nt);
  cout << "NEW done." << endl;

  const int M  = max_angm;
  const int P2 = physpn * 2 + 1;
  const int PS = P2 * P2;

  cout << "=== Comparing tables element by element ===" << endl;

  compare_table3("fac_hamilt_vec", fhv_old, nt.fac_hamilt_vec, M);
  compare_table5("fac_onsite_L", fol_old, nt.fac_operator_onsite_left, M, PS);
  compare_table5("fac_onsite_R", for_old, nt.fac_operator_onsite_rght, M, PS);
  compare_table6("fac_trans_L",  ftl_old, nt.fac_operator_transformation_left, M, P2);
  compare_table6("fac_trans_R",  ftr_old, nt.fac_operator_transformation_rght, M, P2);
  compare_table5("fac_pairup_L", fpl_old, nt.fac_operator_pairup_left, M, PS);
  compare_table5("fac_pairup_R", fpr_old, nt.fac_operator_pairup_rght, M, PS);
  compare_table5("fac_perm_L",   fperml_old, nt.fac_permutation_left,  M, 8);
  compare_table5("fac_perm_R",   fpermr_old, nt.fac_permutation_rght,  M, 8);

  if (n_errors == 0)
    cout << "ALL MATCH — new Dense-based CGC tables agree with old tensor-based tables." << endl;
  else
    cout << n_errors << " MISMATCHES found." << endl;

  return (n_errors == 0) ? 0 : 1;
}
