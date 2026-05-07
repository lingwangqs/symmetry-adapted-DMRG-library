/**
 * src/cgc_tables.cpp — Standalone CGC factor table builder.
 *
 * Ported from dmrg_su2::makeup_clebsch_gordan_coefficient_tensors()
 * in SU2_real_mpi/j1j2longrange/dmrg_su2.cpp.
 *
 * Changes from the original:
 *   - physpn is a function parameter (was a class member)
 *   - comm_rank is local and fixed to 0
 *   - tensor -> Dense<double>
 *   - cgc_coef_left/rght, cgc_coef_singlet, identity are local arrays
 *   - all fac_* tables are stored in CGCTables (output parameter)
 */

#include "../cgc_tables.hpp"
#include "../dense.hpp"
#include "../symmetry/su2.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

using std::cout;
using std::endl;
using std::abs;
using std::sqrt;

template<typename T>
double diff_norm_sq(const T* a, const T* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        auto d = a[i] - b[i];
        sum += d * d;   // use std::norm(d) if complex
    }
    return sum;
}
// ── CGCTables constructor / destructor ────────────────────────────────────────

CGCTables::CGCTables()
  : physpn(0), max_angm(0), physpn2(0), physpnsqr(0),
    fac_hamilt_vec(nullptr),
    fac_operator_onsite_left(nullptr), fac_operator_onsite_rght(nullptr),
    fac_operator_transformation_left(nullptr), fac_operator_transformation_rght(nullptr),
    fac_operator_pairup_left(nullptr), fac_operator_pairup_rght(nullptr),
    fac_permutation_left(nullptr), fac_permutation_rght(nullptr) {}

CGCTables::~CGCTables() {
  int i, j, k, l;
  int M = max_angm;

  if (fac_hamilt_vec) {
    for (i = 0; i < M; ++i) {
      for (j = 0; j < M; ++j) delete[] fac_hamilt_vec[i][j];
      delete[] fac_hamilt_vec[i];
    }
    delete[] fac_hamilt_vec;
  }

  auto free5 = [&](double*****& p) {
    if (!p) return;
    for (i = 0; i < M; ++i)
      for (j = 0; j < M; ++j)
        for (k = 0; k < M; ++k) {
          for (l = 0; l < M; ++l) delete[] p[i][j][k][l];
          delete[] p[i][j][k];
        }
    for (i = 0; i < M; ++i)
      for (j = 0; j < M; ++j) delete[] p[i][j];
    for (i = 0; i < M; ++i) delete[] p[i];
    delete[] p; p = nullptr;
  };

  auto free6 = [&](double******& p) {
    if (!p) return;
    for (i = 0; i < M; ++i)
      for (j = 0; j < M; ++j)
        for (k = 0; k < M; ++k)
          for (l = 0; l < M; ++l) {
            for (int m = 0; m < M; ++m) delete[] p[i][j][k][l][m];
            delete[] p[i][j][k][l];
          }
    for (i = 0; i < M; ++i)
      for (j = 0; j < M; ++j)
        for (k = 0; k < M; ++k) delete[] p[i][j][k];
    for (i = 0; i < M; ++i)
      for (j = 0; j < M; ++j) delete[] p[i][j];
    for (i = 0; i < M; ++i) delete[] p[i];
    delete[] p; p = nullptr;
  };

  free5(fac_operator_onsite_left);
  free5(fac_operator_onsite_rght);
  free6(fac_operator_transformation_left);
  free6(fac_operator_transformation_rght);
  free5(fac_operator_pairup_left);
  free5(fac_operator_pairup_rght);
  free5(fac_permutation_left);
  free5(fac_permutation_rght);
}

// ── Main function ─────────────────────────────────────────────────────────────

void makeup_clebsch_gordan_coefficient_tensors(int physpn, int max_angm, CGCTables& t) {
  // Store metadata
  t.physpn   = physpn;
  t.max_angm = max_angm;
  t.physpn2  = physpn * 2 + 1;
  t.physpnsqr = t.physpn2 * t.physpn2;
  const int comm_rank = 0;  // no MPI; suppress conditional output
  CGCTable cgcfunc;
  int i, j, k, l, m, n, physpn0, physpn1;
  int a0, a1, a2, a3;
  Dense<double> tmp, tmp1, tmp2;
  double nor;
  bool check1, check2;

  const int M  = max_angm;
  const int P2 = t.physpn2;
  const int PS = t.physpnsqr;

  // ── Allocate local CGC arrays ──────────────────────────────────────────────
  Dense<double>*** cgc_coef_left  = new Dense<double>**[M];
  Dense<double>*** cgc_coef_rght  = new Dense<double>**[M];
  for (i = 0; i < M; ++i) {
    cgc_coef_left[i] = new Dense<double>*[M];
    cgc_coef_rght[i] = new Dense<double>*[M];
    for (j = 0; j < M; ++j) {
      cgc_coef_left[i][j] = new Dense<double>[M];
      cgc_coef_rght[i][j] = new Dense<double>[M];
    }
  }
  Dense<double>* cgc_coef_singlet = new Dense<double>[M];
  Dense<double>* identity         = new Dense<double>[M];
  
  for (i = 0; i < M; ++i) {
    cgc_coef_singlet[i].make_singlet(i);
    identity[i].make_identity(i);
  }
  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      for (k = 0; k < M; ++k)
        if (k >= abs(i - j) && k <= abs(i + j)) {
          cgc_coef_left[i][j][k].make_cgc(i, j, k);
          cgc_coef_rght[i][j][k].make_cgc(i, j, k);
        }
  
  // ── fac_hamilt_vec ─────────────────────────────────────────────────────────
  t.fac_hamilt_vec = new double**[M];
  for (i = 0; i < M; ++i) {
    t.fac_hamilt_vec[i] = new double*[M];
    for (j = 0; j < M; ++j) {
      t.fac_hamilt_vec[i][j] = new double[M];
      for (k = 0; k < M; ++k) t.fac_hamilt_vec[i][j][k] = 0.0;
    }
  }

  for (i = 0; i < M; ++i)
    for (j = 0; j < M; ++j)
      for (k = 0; k < M; ++k)
        if (k >= abs(i - j) && k <= abs(i + j) && (i % 2 == k % 2)) {
          tmp1.contract(cgc_coef_left[i][j][k], 0, cgc_coef_singlet[i], 1);
          tmp2.contract(tmp1, 0, cgc_coef_singlet[j], 1);
          tmp1.contract(tmp2, 0, cgc_coef_singlet[k], 1);
          t.fac_hamilt_vec[i][j][k] = cgc_coef_left[i][j][k].inner_prod_u(tmp1) * sqrt((double)(j + 1));
        }

  // ── fac_operator_onsite_left / rght ───────────────────────────────────────
  t.fac_operator_onsite_left = new double****[M];
  t.fac_operator_onsite_rght = new double****[M];
  for (i = 0; i < M; ++i) {
    t.fac_operator_onsite_left[i] = new double***[M];
    t.fac_operator_onsite_rght[i] = new double***[M];
    for (j = 0; j < M; ++j) {
      t.fac_operator_onsite_left[i][j] = new double**[M];
      t.fac_operator_onsite_rght[i][j] = new double**[M];
      for (k = 0; k < M; ++k) {
        t.fac_operator_onsite_left[i][j][k] = new double*[M];
        t.fac_operator_onsite_rght[i][j][k] = new double*[M];
        for (l = 0; l < M; ++l) {
          t.fac_operator_onsite_left[i][j][k][l] = new double[PS];
          t.fac_operator_onsite_rght[i][j][k][l] = new double[PS];
          for (m = 0; m < PS; ++m) {
            t.fac_operator_onsite_left[i][j][k][l][m] = 0.0;
            t.fac_operator_onsite_rght[i][j][k][l][m] = 0.0;
          }
        }
      }
    }
  }

  for (physpn0 = 0; physpn0 < P2; physpn0 += 1)
    for (physpn1 = 0; physpn1 < P2; physpn1 += 1)
      for (m = 0; m <= 6; m += 2)
        for (i = 0; i < M; ++i)
          for (j = 0; j < M; ++j)
            if (j >= abs(i - physpn0) && j <= abs(i + physpn0))
              for (k = 0; k < M; ++k)
                for (l = 0; l < M; ++l)
                  if (l >= abs(k - physpn1) && l <= abs(k + physpn1))
                    if (i == k && l >= abs(j - m) && l <= abs(j + m) && (j % 2 == l % 2))
                      if (physpn1 >= abs(physpn0 - m) && physpn1 <= abs(physpn0 + m)) {
                        // left operator initialize
                        tmp1.contract_dmrg_operator_initial(
                          cgc_coef_left[i][physpn0][j],
                          cgc_coef_left[k][physpn1][l],
                          cgc_coef_left[physpn0][m][physpn1], 0);
                        check1 = tmp1.is_proportional_to(
                          cgc_coef_left[j][m][l],
                          t.fac_operator_onsite_left[j][m][l][i][physpn0 + physpn1 * P2]);
                        if (!check1)
                          cout << "wrong operator_initialize_left i=" << i << " j=" << j
                               << " k=" << k << " l=" << l << " m=" << m
                               << " " << physpn0 << " " << physpn1
                               << " " << t.fac_operator_onsite_left[j][m][l][i][physpn0 + physpn1 * P2] << endl;
                        // rght operator initialize
                        tmp2.contract_dmrg_operator_initial(
                          cgc_coef_rght[physpn0][i][j],
                          cgc_coef_rght[physpn1][k][l],
                          cgc_coef_left[physpn0][m][physpn1], 1);
                        check2 = tmp2.is_proportional_to(
                          cgc_coef_left[j][m][l],
                          t.fac_operator_onsite_rght[j][m][l][i][physpn0 + physpn1 * P2]);
                        if (!check2)
                          cout << "wrong operator_initialize_rght i=" << i << " j=" << j
                               << " k=" << k << " l=" << l << " m=" << m
                               << " " << physpn0 << " " << physpn1
                               << " " << t.fac_operator_onsite_rght[j][m][l][i][physpn0 + physpn1 * P2] << endl;
                      }
  
  // ── fac_operator_transformation_left / rght ───────────────────────────────
  t.fac_operator_transformation_left = new double*****[M];
  t.fac_operator_transformation_rght = new double*****[M];
  for (i = 0; i < M; ++i) {
    t.fac_operator_transformation_left[i] = new double****[M];
    t.fac_operator_transformation_rght[i] = new double****[M];
    for (j = 0; j < M; ++j) {
      t.fac_operator_transformation_left[i][j] = new double***[M];
      t.fac_operator_transformation_rght[i][j] = new double***[M];
      for (k = 0; k < M; ++k) {
        t.fac_operator_transformation_left[i][j][k] = new double**[M];
        t.fac_operator_transformation_rght[i][j][k] = new double**[M];
        for (l = 0; l < M; ++l) {
          t.fac_operator_transformation_left[i][j][k][l] = new double*[M];
          t.fac_operator_transformation_rght[i][j][k][l] = new double*[M];
          for (m = 0; m < M; ++m) {
            t.fac_operator_transformation_left[i][j][k][l][m] = new double[P2];
            t.fac_operator_transformation_rght[i][j][k][l][m] = new double[P2];
            for (n = 0; n < P2; ++n) {
              t.fac_operator_transformation_left[i][j][k][l][m][n] = 0.0;
              t.fac_operator_transformation_rght[i][j][k][l][m][n] = 0.0;
            }
          }
        }
      }
    }
  }

  for (physpn0 = 0; physpn0 < P2; physpn0 += 1) {
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
                    // left canonical form
                    tmp1.contract_dmrg_operator_transformation(
                      cgc_coef_left[i][physpn0][j],
                      cgc_coef_left[k][physpn1][l],
                      cgc_coef_left[i][m][k], 0);
                    check1 = tmp1.is_proportional_to(
                      cgc_coef_left[j][m][l],
                      t.fac_operator_transformation_left[j][m][l][i][k][physpn0]);
		    if(!check1){
		      cout << "wrong operator_transformation_left i=" << i << " j=" << j
			   << " k=" << k << " l=" << l << " m=" << m
			   << " " << physpn0 << " " << physpn1
			   << " " << t.fac_operator_transformation_left[j][m][l][i][k][physpn0] << endl;
		    }
                    // rght canonical form
                    tmp2.contract_dmrg_operator_transformation(
                      cgc_coef_rght[physpn0][i][j],
                      cgc_coef_rght[physpn1][k][l],
                      cgc_coef_left[i][m][k], 1);
                    check2 = tmp2.is_proportional_to(
                      cgc_coef_left[j][m][l],
                      t.fac_operator_transformation_rght[j][m][l][i][k][physpn0]);
		    if(!check2){
		      cout << "wrong operator_transformation_rght i=" << i << " j=" << j
			   << " k=" << k << " l=" << l << " m=" << m
			   << " " << physpn0 << " " << physpn1
			   << " " << t.fac_operator_transformation_rght[j][m][l][i][k][physpn0] << endl;
		    }
                  }
  }

  // ── fac_operator_pairup_left / rght ───────────────────────────────────────
  t.fac_operator_pairup_left = new double****[M];
  t.fac_operator_pairup_rght = new double****[M];
  for (i = 0; i < M; ++i) {
    t.fac_operator_pairup_left[i] = new double***[M];
    t.fac_operator_pairup_rght[i] = new double***[M];
    for (j = 0; j < M; ++j) {
      t.fac_operator_pairup_left[i][j] = new double**[M];
      t.fac_operator_pairup_rght[i][j] = new double**[M];
      for (k = 0; k < M; ++k) {
        t.fac_operator_pairup_left[i][j][k] = new double*[M];
        t.fac_operator_pairup_rght[i][j][k] = new double*[M];
        for (l = 0; l < M; ++l) {
          t.fac_operator_pairup_left[i][j][k][l] = new double[PS];
          t.fac_operator_pairup_rght[i][j][k][l] = new double[PS];
          for (m = 0; m < PS; ++m) {
            t.fac_operator_pairup_left[i][j][k][l][m] = 0.0;
            t.fac_operator_pairup_rght[i][j][k][l][m] = 0.0;
          }
        }
      }
    }
  }

  for (physpn0 = 0; physpn0 < P2; physpn0 += 1)
    for (physpn1 = 0; physpn1 < P2; physpn1 += 1)
      for (m = 0; m <= 6; m += 2)
        for (i = 0; i < M; ++i)
          for (j = 0; j < M; ++j)
            if (j >= abs(i - physpn0) && j <= abs(i + physpn0))
              for (k = 0; k < M; ++k)
                for (l = 0; l < M; ++l)
                  if (l >= abs(k - physpn1) && l <= abs(k + physpn1))
                    if (k >= abs(i - m) && k <= (i + m) && j == l &&
                        physpn1 >= abs(physpn0 - m) && physpn1 <= abs(physpn0 + m)) {
                      tmp.contract(cgc_coef_left[physpn0][m][physpn1], 1, cgc_coef_singlet[m], 1);
                      tmp.shift(0, 2);
                      tmp *= sqrt((double)(m + 1));
                      tmp1.contract_dmrg_operator_pairup(
                        cgc_coef_left[i][physpn0][j],
                        cgc_coef_left[k][physpn1][l],
                        cgc_coef_left[i][m][k], tmp, 0);
                      check1 = tmp1.is_proportional_to(
                        identity[j],
                        t.fac_operator_pairup_left[j][i][k][m][physpn0 + physpn1 * P2]);
		      if(!check1){
			cout<<"wrong"<<endl;
		      }
                      tmp2.contract_dmrg_operator_pairup(
                        cgc_coef_rght[physpn0][i][j],
                        cgc_coef_rght[physpn1][k][l],
                        cgc_coef_left[i][m][k], tmp, 1);
                      check2 = tmp2.is_proportional_to(
                        identity[j],
                        t.fac_operator_pairup_rght[j][i][k][m][physpn0 + physpn1 * P2]);
		      if(!check2){
			cout<<"wrong"<<endl;
		      }
                    }
  
  // ── fac_permutation_left / rght ───────────────────────────────────────────
  cout << "start permutation" << endl;
  t.fac_permutation_left = new double****[M];
  t.fac_permutation_rght = new double****[M];
  for (i = 0; i < M; ++i) {
    t.fac_permutation_left[i] = new double***[M];
    t.fac_permutation_rght[i] = new double***[M];
    for (j = 0; j < M; ++j) {
      t.fac_permutation_left[i][j] = new double**[M];
      t.fac_permutation_rght[i][j] = new double**[M];
      for (k = 0; k < M; ++k) {
        t.fac_permutation_left[i][j][k] = new double*[M];
        t.fac_permutation_rght[i][j][k] = new double*[M];
        for (l = 0; l < M; ++l) {
          t.fac_permutation_left[i][j][k][l] = new double[8];
          t.fac_permutation_rght[i][j][k][l] = new double[8];
          for (m = 0; m < 8; ++m) {
            t.fac_permutation_left[i][j][k][l][m] = 0.0;
            t.fac_permutation_rght[i][j][k][l][m] = 0.0;
          }
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
                a1 = (a0 % 2) * 2;       a0 /= 2;
                a2 = (a0 % 2) * 2 + 1;   a0 /= 2;
                a3 = (a0 % 2) * 2;
                if (!(a2 <= abs(a1 + physpn) && a2 >= abs(a1 - physpn) &&
                      a3 <= abs(a2 + physpn) && a3 >= abs(a2 - physpn))) continue;
                // permutation left
                if ((i % 2 == k % 2) && (j % 2 == l % 2) &&
                    k <= abs(a1 + i) && k >= abs(a1 - i) &&
                    l <= abs(a3 + j) && l >= abs(a3 - j)) {
                  tmp1.contract_dmrg_permutation(
                    cgc_coef_left[i][physpn][j],
                    cgc_coef_left[k][physpn][l],
                    cgc_coef_left[i][a1][k],
                    cgc_coef_left[physpn][a1][a2],
                    cgc_coef_left[physpn][a3][a2], 0);
                  if (!tmp1.is_proportional_to(cgc_coef_left[j][a3][l], nor)) {
                    cout << "contract permutation_left wrong" << endl;
                    tmp1.print();
                    cgc_coef_left[j][a3][l].print();
                    exit(0);
                  }
                  t.fac_permutation_left[i][j][k][l][m] = nor;
                }
                // permutation rght
                if ((i % 2 == k % 2) && (j % 2 == l % 2) &&
                    k <= abs(a3 + i) && k >= abs(a3 - i) &&
                    l <= abs(a1 + j) && l >= abs(a1 - j)) {
                  tmp2.contract_dmrg_permutation(
                    cgc_coef_rght[physpn][i][j],
                    cgc_coef_rght[physpn][k][l],
                    cgc_coef_left[i][a3][k],
                    cgc_coef_left[a2][physpn][a1],
                    cgc_coef_left[a2][physpn][a3], 1);
                  if (!tmp2.is_proportional_to(cgc_coef_left[j][a1][l], nor)) {
                    cout << "contract permutation_rght wrong" << endl;
                    tmp2.print();
                    cgc_coef_left[j][a1][l].print();
                    exit(0);
                  }
                  t.fac_permutation_rght[i][j][k][l][m] = nor;
                }
              }
  cout << "done permutation" << endl;
  //*/
  // ── Cleanup local CGC arrays ───────────────────────────────────────────────
  delete[] cgc_coef_singlet;
  delete[] identity;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < M; ++j) {
      delete[] cgc_coef_left[i][j];
      delete[] cgc_coef_rght[i][j];
    }
    delete[] cgc_coef_left[i];
    delete[] cgc_coef_rght[i];
  }
  delete[] cgc_coef_left;
  delete[] cgc_coef_rght;
  cout<<"physpn="<<physpn<<endl;
  cout<<"max_angm="<<max_angm<<endl;
}
