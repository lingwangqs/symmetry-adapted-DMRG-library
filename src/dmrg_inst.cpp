/**
 * src/dmrg_inst.cpp — DMRG<Scalar,Sym> and Lanczos<Scalar,Sym> implementation.
 *
 * Ported from mps_u1/dmrg_su2.cpp + dmrg_su2_omp.cpp + lanczos_su2.cpp.
 * SU2 vs U1 differences handled via `if constexpr (Sym::has_cgc)`.
 *
 * Global parameters (formerly extern in original code) are now:
 *   - bondd     : max bond dimension (was max_dcut)
 *   - omp_get_max_threads() : number of OMP threads (was psize)
 *   - comm_rank = 0 : single rank (no MPI in this version)
 */

#include "../dmrg.hpp"
#include "../lanczos.hpp"
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

extern "C" {
  // LAPACK: symmetric tridiagonal eigensolver
  void dstev_(char*, int*, double*, double*, double*, int*, double*, int*);
}

using std::cout;
using std::endl;
using std::setprecision;

// CGC table type and free function (defined in src/cgc_tables.cpp)
#include "../cgc_tables.hpp"
void makeup_clebsch_gordan_coefficient_tensors(int physpn, int max_angm, CGCTables& t);
extern CGCTables* g_cgc_tables;   // defined in main

// ── Global DMRG pointer (used by Lanczos::diag_op) ───────────────────────────
// In the original code this was a global. We keep it here as a thread-local
// workaround: the chain pointer is set before each Lanczos call.

template<typename Scalar, typename Sym>
static DMRG<Scalar,Sym>* g_chain_ptr = nullptr;

// ── DMRG Constructor ──────────────────────────────────────────────────────────
template<typename Scalar, typename Sym>
DMRG<Scalar,Sym>::DMRG(){
  opr = nullptr;
  fll = nullptr;
  frr = nullptr;
  hmap = nullptr;
  coup = nullptr;
  uu = nullptr;
  hh = nullptr;
  orth = nullptr;
  ovlp = nullptr;
  overlapvec = nullptr;
  tran = nullptr;
  ww = nullptr;
  wtmp = nullptr;
}

template<typename Scalar, typename Sym>
DMRG<Scalar,Sym>::DMRG(int xx, int yy, int bondd_, int physpn_, int totspin_, int max_exci_)
  : DMRG()
{
  int i, j, k, max_dcut_ww;
  lx     = xx;
  ly     = yy;
  ns     = lx * ly;
  bondd  = bondd_;
  physpn = physpn_;   // 2*J for SU2, or 2*Sz_max for U1
  phdim  = physpn + 1;
  totspin = totspin_;
  exci    = max_exci_;    // current excitation target
  max_exci = max_exci_;
  nfree = 2;

  max_dcut_ww = bondd;

  // ── Allocate singular value arrays ──
  ww = new double*[max_exci + 1];
  for (i = 0; i <= max_exci; ++i)
    ww[i] = new double[max_dcut_ww]();
  wtmp = new double[max_dcut_ww]();

  // ── Allocate MPS and environment tensors ──
  hh  = new TensorType[ns];
  uu  = new TensorType[ns];
  opr = new TensorType*[ns];
  fll = new int*[ns];
  frr = new int*[ns];
  hmap = new int*[ns];
  coup = new double[ns + 1]();
  for (i = 0; i < ns; ++i) {
    opr[i]  = new TensorType[ns];
    fll[i]  = new int[ns]();
    frr[i]  = new int[ns]();
    hmap[i] = new int[ns]();
  }

  // ── Excited state / overlap arrays ──
  orth       = new TensorType*[max_exci + 1];
  ovlp       = new TensorType*[max_exci + 1];
  overlapvec = new TensorType[max_exci + 1];
  tran       = new TensorType*[2];
  for (i = 0; i <= max_exci; ++i) {
    orth[i] = new TensorType[ns];
    ovlp[i] = new TensorType[ns];
  }
  for (i = 0; i < 2; ++i)
    tran[i] = new TensorType[ns];

  // ── Build Hamiltonian bond map (simple 1D nearest-neighbour chain) ──
  // hmap[i][j] = coupling index (0 = no coupling, 1 = J1, 2 = J2, ...)
  // For a simple Heisenberg chain: hmap[i][i+1] = 1 (with periodic BC handled by bdry)
  for (i = 0; i < ns; ++i)
    for (j = 0; j < ns; ++j)
      hmap[i][j] = 0;
  
  for(k = 0; k < 2; k++)
    for(i = 0; i < ns; i++){
      int x1 = i / ly, y1, x2, y2;
      if(x1 % 2 == 0)
	y1 = i % ly;
      else if(x1 % 2 == 1)
	y1 = ly - 1 - (i%ly);
      if(k == 0){
	x2 = (x1 + 1) % lx;
	y2 = y1;
      }
      else if(k == 1){
	x2 = x1;
	y2 = (y1 + 1) % ly;
      }
      if(x2 % 2 == 0)
	j = x2 * ly + y2;
      else if(x2 % 2 == 1)
	j = x2 * ly + (ly - 1 - y2);
      if (i != j){
	hmap[i][j] = 1;
	hmap[j][i] = 1;
      }
    }
  /*
  // Default: nearest-neighbour (1D) with hmap[i][j] = |i-j| (distance)
  // So coup[1] = J1, coup[2] = J2, etc.  This replicates the original.
  for (i = 0; i < ns; ++i)
    for (j = i + 1; j < ns; ++j) {
      k = j - i;
      if (k > ns / 2) k = ns - k;
      if (k < 2){ //test nearest neighbor first
	hmap[i][j] = k;
	hmap[j][i] = k;
      }
    }
  */
  // Simple uniform nearest-neighbour Heisenberg: coup[1] = 1.0
  for (i = 0; i <= ns / 2; ++i) coup[i] = 0.0;
  coup[1] = 1.0;   // J1

  // ── Build fll/frr flag maps ──
  for (i = 0; i < ns; ++i) {
    for (j = 0; j < ns; ++j) {
      fll[i][j] = 0;
      frr[i][j] = 0;
    }
    for (j = 0; j < i; ++j)
      for (k = i + 1; k < ns; ++k)
        if (hmap[j][k] != 0)
          fll[i][j] = 1;
    for (j = i + 1; j < ns; ++j)
      for (k = 0; k < i; ++k)
        if (hmap[j][k] != 0)
          frr[i][j] = 1;
  }

  // ── SU2-only: setup CGC tables ──
  if constexpr (Sym::has_cgc) {
    g_cgc_tables = new CGCTables();
    makeup_clebsch_gordan_coefficient_tensors(physpn, 20, *g_cgc_tables);
    cout<<"done make_cgc_tensors"<<endl;
  }

  // ── Initialize local spin operators ──
  initialize_local_operators();
}

// ── DMRG Destructor ───────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
DMRG<Scalar,Sym>::~DMRG() {
  int i;
  if (opr) {
    for (i = 0; i < ns; ++i) {
      delete[] opr[i];
      delete[] fll[i];
      delete[] frr[i];
      delete[] hmap[i];
    }
    delete[] opr; delete[] fll; delete[] frr; delete[] hmap;
  }
  delete[] coup;
  delete[] uu; delete[] hh;
  if (orth) {
    for (i = 0; i <= max_exci; ++i) {
      delete[] orth[i]; delete[] ovlp[i];
    }
    delete[] orth; delete[] ovlp;
  }
  delete[] overlapvec;
  if (tran) { for (i = 0; i < 2; ++i) delete[] tran[i]; delete[] tran; }
  if (ww) {
    for (i = 0; i <= max_exci; ++i) delete[] ww[i];
    delete[] ww;
  }
  delete[] wtmp;
}

// ── initialize_local_operators ────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::initialize_local_operators() {
  // sigma[0] and sigma[1] are spinor operators (T^{1/2} components).
  // For physpn=1 (spin-1/2), they represent S+ and S- components with the
  // SU2 reduced matrix element <1/2||S||1/2> = sqrt(3)/2 and sign.
  // sigma[2] = identity projected to spin-up sector (left boundary convention)
  // sigma[3] = identity for right boundary convention
  if constexpr (Sym::has_cgc) {
    sigma[0].make_spinor_start(physpn);
    sigma[1].make_spinor_start(physpn);
    //sigma[2].make_spinhalf_identity_start();
    //sigma[3].make_spinhalf_identity_end();

    // Scale the spinors by SU2 reduced matrix element <j||S||j>
    if (physpn == 1) {
      // spin-1/2: <1/2||S||1/2> = sqrt(3)/2
      sigma[0] *= std::sqrt(3.0) / 2.0;
      sigma[1] *= -std::sqrt(3.0) / 2.0;
    } else if (physpn == 2) {
      // spin-1: build from contraction of two spin-1/2 spinors
      sigma[0] *= std::sqrt(2.0);
      sigma[1] *= -std::sqrt(2.0);
      // (higher-spin construction omitted for now)
    }
  }
  else {
    sigma[0].make_spinor_start(physpn);
    sigma[1].make_spinor_start(physpn);
    TensorType tmp1, tmp2;
    BondType bb[2];
    sigma[0].get_bond(1, bb[0]);
    bb[0].invert_bonddir();
    bb[1] = bb[0];
    tmp1.fuse_to_singlet(bb[0], bb[1]);
    tmp2.contract(sigma[1], 1, tmp1, 1);
    tmp2.shift(1, 0);
    sigma[1] = tmp2;
  }
}

// ── lanczos_solve_eigenvector_idmrg ──────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::lanczos_solve_eigenvector_idmrg(int il, int ir, TensorType& vec) {
  Lanczos<Scalar,Sym> lan;
  int mlanc = 40, ndata, ncgc;
  vec.get_nelement(ndata, ncgc);
  if (mlanc > ndata) mlanc = ndata + 1;
  if (mlanc < 2)     mlanc = 2;

  // Set the chain pointer so Lanczos::diag_op can find us
  g_chain_ptr<Scalar,Sym> = this;
  lan.initialize_lanczos(vec, mlanc, 1);
  lan.lanczos1(il, ir, vec);
  gs_enr[exci] = lan.get_eigval();
  cout << "Lanczos  site=(" << il << "," << ir
       << ")  E[" << exci << "]=" << setprecision(12) << gs_enr[exci] << endl;
}

// ── hamiltonian_vector_multiplication_idmrg ───────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::hamiltonian_vector_multiplication_idmrg(int il, int ir,
                                                                 TensorType& vec1,
                                                                 TensorType& vec2) {
  int psize = omp_get_max_threads();
  TensorType *vec2arr = new TensorType[psize];
  TensorType tmp, op;
  int i, j, k, m, m0, m1, m4;
  Scalar fac, overlap;
  
  // ── Diagonal part (H_left * v * I + I * v * H_right) ──
  if (hh[il].is_null() && hh[ir].is_null()) {
    vec2 = vec1; vec2 = 0.0;
  } else if (!hh[il].is_null() && !hh[ir].is_null()) {
    // Only left env; use identity right
    vec2.contract(hh[il], 0, vec1, 0);
    // Only right env; use identity left
    tmp.contract(vec1, 1, hh[ir], 0);
    vec2 += tmp;
  } else if (hh[il].is_null() && !hh[ir].is_null()) {
    // Only right env; use identity left
    vec2.contract(vec1, 1, hh[ir], 0);
  } else if (!hh[il].is_null() && hh[ir].is_null()) {
    // Only left env; use identity right
    vec2.contract(hh[il], 0, vec1, 0);
  }
  // Determine loop structure (same as original)
  if (il < ns / 2) {
    m1 = il + 1;  //number of sites to be paired
    m4 = ns - ir; //number of op to be summed for given sites
  } else {
    m1 = ns - ir; //number of sites to be paired
    m4 = il + 1;  //number of op to be summed for given sites
  }
  int m2 = m1 + exci;
  int m3 = m2;  // no translation operators for now, otherwise m3 = m2 + 2;

#pragma omp parallel for default(shared) private(i,j,k,m,m0,overlap,fac,tmp,op) schedule(static,1)
  for (m0 = 0; m0 < m2; ++m0) {
    int tid = omp_get_thread_num();
    if (m0 < m1) {
      // Bond terms: left-site j < il paired with right site k > ir
      if (il < ns / 2) j = m0;
      else             k = ir + m0;

      op.clean();
      for (m = 0; m < m4; ++m) {
        if (il < ns / 2) k = ir + m;
        else             j = m;
        if (hmap[j][k] > 0) {
          TensorType ht;
          if (il < ns / 2) ht = opr[ir][k];
          else             ht = opr[il][j];
          ht *= coup[hmap[j][k]];
          if (op.is_null()) op = ht;
          else              op += ht;
        }
      }
      if (op.is_null()) continue;

      TensorType res;
      if (il < ns / 2)
        res.hamiltonian_vector_multiplication(vec1, opr[il][j], op);
      else
        res.hamiltonian_vector_multiplication(vec1, op, opr[ir][k]);

      if (vec2arr[tid].is_null()) vec2arr[tid] = res;
      else                         vec2arr[tid] += res;
    }
    else if (m0 >= m1 && m0 < m2) {
      // Orthogonality penalty for excited states
      if (il == ir - 1) {
        int ei = m0 - m1;
        overlap = overlapvec[ei].inner_prod_c(vec1);
        overlap = overlap * Scalar(gs_enr[ei]);
        TensorType pen = overlapvec[ei];
        pen *= -overlap;
        if (vec2arr[tid].is_null()) vec2arr[tid] = pen;
        else                         vec2arr[tid] += pen;
      }
    }
  }

  for (i = 0; i < psize; ++i)
    if (!vec2arr[i].is_null())
      vec2 += vec2arr[i];

  delete[] vec2arr;
}

// ── prepare_input_vector_initial ─────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::prepare_input_vector_initial(int il, int ir, TensorType& vec) {
  TensorType tmp1, tmp2, left, rght, tmp, tmp3;
  int j;
  BondType bb[4];
  uu[il].get_bond(0, bb[0]); uu[il].get_bond(1, bb[1]);
  uu[ir].get_bond(0, bb[2]); uu[ir].get_bond(1, bb[3]);
  tmp1.fuse(bb[0], bb[1]);
  tmp2.fuse(bb[2], bb[3]);
  left.overlap_initial(uu[il], tmp1, 0);
  rght.overlap_initial(uu[ir], tmp2, 1);

  left.get_bond(0, bb[0]);
  rght.get_bond(0, bb[1]);
  bb[0].invert_bonddir();
  bb[1].invert_bonddir();
  //fuse_to_singlet: for SU2, this generate a all-one reduced matrix multiply a singlet cgc; for u1, it generate a all-one reduced matrix
  tmp.fuse_to_singlet(bb[0], bb[1]);
  tmp3.contract(left, 0, tmp, 0);
  vec.contract(tmp3, 1, rght, 0);
  //makeup_input_vector: the function creates memory for valid blocks which has nullptr
  vec.makeup_input_vector();

  if (il + 1 != ir) return;
  for (j = 0; j < exci; ++j) {
    left.overlap_initial(ovlp[j][il], tmp1, 0);
    rght.overlap_initial(ovlp[j][ir], tmp2, 1);
    //2-index tensor, taking_conjugate shall multiply cgc by a singlet tensor
    if constexpr (Sym::has_cgc)
      left.take_conjugate(0);
    overlapvec[j].contract(left, 0, rght, 0);
    overlapvec[j].makeup_input_vector();
  }
}

// ── prepare_input_vector ─────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::prepare_input_vector(int il, int ir, TensorType& vec) {
  TensorType tmp1, tmp2, left, rght;
  int j;
  BondType bb[4];
  uu[il].get_bond(0, bb[0]); uu[il].get_bond(1, bb[1]);
  uu[ir].get_bond(0, bb[2]); uu[ir].get_bond(1, bb[3]);
  tmp1.fuse(bb[0], bb[1]);
  tmp2.fuse(bb[2], bb[3]);
  left.overlap_initial(uu[il], tmp1, 0);
  rght.overlap_initial(uu[ir], tmp2, 1);
  if constexpr (Sym::has_cgc)
    left.take_conjugate(0);
  vec.contract(left, 0, rght, 0);
  vec.makeup_input_vector();

  if (il + 1 != ir) return;
  for (j = 0; j < exci; ++j) {
    left.overlap_initial(ovlp[j][il], tmp1, 0);
    rght.overlap_initial(ovlp[j][ir], tmp2, 1);
    if constexpr (Sym::has_cgc)
      left.take_conjugate(0);
    overlapvec[j].contract(left, 0, rght, 0);
    overlapvec[j].makeup_input_vector();
  }
}
// ── prepare_site_operator_from_left (with exci update) ───────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::prepare_site_operator_from_left(int i, int flag) {
  //flag tells if overlap tensors need to be computed
  int psize = omp_get_max_threads();
  int j, k, m, m0, m1, m2, m3, m4, m5;
  TensorType htmp, *hh2, op;

  if (i == 0) {
    opr[i][i].operator_initial(uu[i], uu[i], sigma[0], 0);
    for (j = 0; j < exci; ++j)
      ovlp[j][i].overlap_initial(orth[j][i], uu[i], 0);
    hh[i].clean();
    return;
  }
  hh2 = new TensorType[psize];
  if (!hh[i-1].is_null())
    hh[i].overlap_transformation(uu[i], uu[i], hh[i-1], 0);
  else
    hh[i].clean();

  m1 = i;        // existing left operators
  m2 = m1 + 1;   // pair-up
  m3 = m2 + 1;   // init new left operator at site i
  m4 = m3 + exci;// overlap update
  m5 = m4;       // (translation operators skipped)

#pragma omp parallel for default(shared) private(j,k,m,m0,htmp,op) schedule(dynamic,1)
  for (m0 = 0; m0 < m4; ++m0) {
    int tid = omp_get_thread_num();
    if (m0 < m1) {
      k = m0;
      if (fll[i][k] == 1){
        opr[i][k].operator_transformation(uu[i], uu[i], opr[i-1][k], 0);
      }
    }
    else if (m0 >= m1 && m0 < m2) {
      op.clean();
      for (m = 0; m < i; ++m) {
        k = m;
        if (hmap[i][k] > 0) {
          htmp = opr[i-1][k];
          htmp *= coup[hmap[i][k]];
          if (op.is_null()) op = htmp;
          else              op += htmp;
        }
      }
      if (op.is_null()) continue;
      htmp.operator_pairup(uu[i], uu[i], op, sigma[1], 0);
      if (hh2[tid].is_null()) hh2[tid] = htmp;
      else                     hh2[tid] += htmp;
    }
    else if (m0 >= m2 && m0 < m3) {
      opr[i][i].operator_initial(uu[i], uu[i], sigma[0], 0);
    }
    else if (m0 >= m3 && m0 < m4) {
      j = m0 - m3;
      ovlp[j][i].overlap_transformation(orth[j][i], uu[i], ovlp[j][i-1], 0);
      if (flag == 1) {
	ovlp[j][i+1].contract(ovlp[j][i], 0, orth[j][i+1], 0);
	ovlp[j][i+2].contract(orth[j][i+2], 1, ovlp[j][i+3], 0);
	ovlp[j][i+2].shift(1,0);
      }
    }
  }

  for (j = 0; j < psize; ++j)
    if (!hh2[j].is_null()) {
      if (hh[i].is_null()) hh[i] = hh2[j];
      else                  hh[i] += hh2[j];
    }
  delete[] hh2;
}

// ── prepare_site_operator_from_right (with exci update) ──────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::prepare_site_operator_from_right(int i, int flag) {
  int psize = omp_get_max_threads();
  int j, k, m, m0, m1, m2, m3, m4;
  TensorType htmp, *hh2, op;

  if (i == ns - 1) {
    opr[i][i].operator_initial(uu[i], uu[i], sigma[1], 1);
    for (j = 0; j < exci; ++j)
      ovlp[j][i].overlap_initial(orth[j][i], uu[i], 1);
    hh[i].clean();
    return;
  }

  hh2 = new TensorType[psize];
  if (!hh[i+1].is_null())
    hh[i].overlap_transformation(uu[i], uu[i], hh[i+1], 1);
  else  hh[i].clean(); 

  m1 = ns - 1 - i;
  m2 = m1 + 1;
  m3 = m2 + 1;
  m4 = m3 + exci;

#pragma omp parallel for default(shared) private(j,k,m,m0,htmp,op) schedule(dynamic,1)
  for (m0 = 0; m0 < m4; ++m0) {
    int tid = omp_get_thread_num();
    if (m0 < m1) {
      k = i + 1 + m0;
      if (frr[i][k] == 1)
        opr[i][k].operator_transformation(uu[i], uu[i], opr[i+1][k], 1);
    }
    else if (m0 >= m1 && m0 < m2) {
      op.clean();
      for (m = 0; m < ns - 1 - i; ++m) {
        k = i + 1 + m;
        if (hmap[i][k] > 0) {
          htmp = opr[i+1][k];
          htmp *= coup[hmap[i][k]];
          if (op.is_null()) op = htmp;
          else              op += htmp;
        }
      }
      if (op.is_null()) continue;
      htmp.operator_pairup(uu[i], uu[i], op, sigma[0], 1);
      if (hh2[tid].is_null()) hh2[tid] = htmp;
      else                     hh2[tid] += htmp;
    }
    else if (m0 >= m2 && m0 < m3) {
      opr[i][i].operator_initial(uu[i], uu[i], sigma[1], 1);
    }
    else if (m0 >= m3 && m0 < m4) {
      j = m0 - m3;
      ovlp[j][i].overlap_transformation(orth[j][i], uu[i], ovlp[j][i+1], 1);
      if(flag==1){
	ovlp[j][i-2].contract(ovlp[j][i-3], 0, orth[j][i-2], 0);
	ovlp[j][i-1].contract(orth[j][i-1], 1, ovlp[j][i], 0);
	ovlp[j][i-1].shift(1,0);
      }
    }
  }

  for (j = 0; j < psize; ++j)
    if (!hh2[j].is_null()) {
      if (hh[i].is_null()) hh[i] = hh2[j];
      else                  hh[i] += hh2[j];
    }
  delete[] hh2;
}

// ── prepare_site_operator_from_left (canonical, no overlap update) ────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::prepare_site_operator_from_left(int i) {
  int psize = omp_get_max_threads();
  int j, k, m, m0, m1, m2, m3, m4;
  TensorType htmp, *hh2, op, uu1;
  BondType bb[2];

  uu[i].get_bond(0, bb[0]); uu[i].get_bond(1, bb[1]);
  uu1.fuse(bb[0], bb[1]);

  hh2 = new TensorType[psize];
  if (!hh[i-1].is_null())
    hh[i].overlap_transformation(uu1, uu1, hh[i-1], 0);
  else hh[i].clean();

  m1 = i; m2 = m1 + 1; m3 = m2 + 1; m4 = m3;

#pragma omp parallel for default(shared) private(j,k,m,m0,htmp,op) schedule(dynamic,1)
  for (m0 = 0; m0 < m4; ++m0) {
    int tid = omp_get_thread_num();
    if (m0 < m1) {
      k = m0;
      if (fll[i][k] == 1)
        opr[i][k].operator_transformation(uu1, uu1, opr[i-1][k], 0);
    }
    else if (m0 >= m1 && m0 < m2) {
      op.clean();
      for (m = 0; m < i; ++m) {
        k = m;
        if (hmap[i][k] > 0) {
          htmp = opr[i-1][k];
          htmp *= coup[hmap[i][k]];
          if (op.is_null()) op = htmp;
          else              op += htmp;
        }
      }
      if (op.is_null()) continue;
      htmp.operator_pairup(uu1, uu1, op, sigma[1], 0);
      if (hh2[tid].is_null()) hh2[tid] = htmp;
      else                     hh2[tid] += htmp;
    }
    else if (m0 >= m2 && m0 < m3) {
      opr[i][i].operator_initial(uu1, uu1, sigma[0], 0);
    }
  }

  for (j = 0; j < psize; ++j)
    if (!hh2[j].is_null()) {
      if (hh[i].is_null()) hh[i] = hh2[j];
      else                  hh[i] += hh2[j];
    }
  delete[] hh2;
}

// ── prepare_site_operator_from_right (canonical) ─────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::prepare_site_operator_from_right(int i) {
  int psize = omp_get_max_threads();
  int j, k, m, m0, m1, m2, m3;
  TensorType htmp, *hh2, op, uu1;
  BondType bb[2];

  uu[i].get_bond(0, bb[0]); uu[i].get_bond(1, bb[1]);
  uu1.fuse(bb[0], bb[1]);

  hh2 = new TensorType[psize];
  if (!hh[i+1].is_null())
    hh[i].overlap_transformation(uu1, uu1, hh[i+1], 1);
  else hh[i].clean();

  m1 = ns - 1 - i; m2 = m1 + 1; m3 = m2 + 1;

#pragma omp parallel for default(shared) private(j,k,m,m0,htmp,op) schedule(dynamic,1)
  for (m0 = 0; m0 < m3; ++m0) {
    int tid = omp_get_thread_num();
    if (m0 < m1) {
      k = i + 1 + m0;
      if (frr[i][k] == 1)
        opr[i][k].operator_transformation(uu1, uu1, opr[i+1][k], 1);
    }
    else if (m0 >= m1 && m0 < m2) {
      op.clean();
      for (m = 0; m < ns - 1 - i; ++m) {
        k = i + 1 + m;
        if (hmap[i][k] > 0) {
          htmp = opr[i+1][k];
          htmp *= coup[hmap[i][k]];
          if (op.is_null()) op = htmp;
          else              op += htmp;
        }
      }
      if (op.is_null()) continue;
      htmp.operator_pairup(uu1, uu1, op, sigma[0], 1);
      if (hh2[tid].is_null()) hh2[tid] = htmp;
      else                     hh2[tid] += htmp;
    }
    else if (m0 >= m2 && m0 < m3) {
      opr[i][i].operator_initial(uu1, uu1, sigma[1], 1);
    }
  }

  for (j = 0; j < psize; ++j)
    if (!hh2[j].is_null()) {
      if (hh[i].is_null()) hh[i] = hh2[j];
      else                  hh[i] += hh2[j];
    }
  delete[] hh2;
}

// ── do_idmrg ──────────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::do_idmrg() {
  int i;
  TensorType vec;
  BondType *bb = new BondType[4];
  int q0 = 0, d1 = 1, qp = physpn, qt = totspin;

  cout << "begin iDMRG" << endl;

  // Initialize boundary tensors.
  // Convention: incoming bonds use dir=1, outgoing bonds use dir=-1.
  // uu[0]: left boundary. Bonds (dir=1,Q=0) × (dir=1,Q=physpn) → (dir=-1,Q=physpn).
  // uu[ns-1]: right boundary. Bonds (dir=1,Q=physpn) × (dir=1,Q=totspin) → (dir=-1,Q=?).
  if constexpr (Sym::has_cgc){
    bb[0].set(1, 1, &q0, &d1);   // left vacuum bond
    bb[1].set(1, 1, &qp, &d1);   // physical spin bond (Q=physpn)
    uu[0].fuse(bb[0], bb[1]);

    bb[2].set(1, 1, &qp, &d1);   // physical spin bond at right end
    bb[3].set(1, 1, &qt, &d1);   // right target-sector bond
    uu[ns-1].fuse(bb[2], bb[3]);
  }
  else {//this is a setup for spin-1/2 U1 case
    int cval[2], bdim[2];
    cval[0] = 0;
    bdim[0] = 1;
    bb[0].set(1, 1, cval, bdim);
    cval[0] = -1;
    cval[1] = 1;
    bdim[0] = 1;
    bdim[1] = 1;
    bb[1].set(1, 2, cval, bdim);
    uu[0].fuse(bb[0],bb[1]);
    cval[0] = -1;
    cval[1] = 1;
    bdim[0] = 1;
    bdim[1] = 1;
    bb[2].set(1, 2, cval, bdim);
    cval[0] = qt;
    bdim[0] = 1;
    bb[3].set(-1, 1, cval, bdim);
    uu[ns-1].fuse(bb[2],bb[3]);
  }

  for (i = 0; i <= ns / 2 - 2; ++i) {
    cout << "iDMRG step i=" << i << endl;
    cout << "  calling prepare_site_operator_from_left(" << i << ",0)" << endl; cout.flush();
    prepare_site_operator_from_left(i, 0);
    cout << "  calling prepare_site_operator_from_right(" << ns-1-i << ",0/1)" << endl; cout.flush();
    if (i < ns / 2 - 2)
      prepare_site_operator_from_right(ns - 1 - i, 0);
    else
      prepare_site_operator_from_right(ns - 1 - i, 1);

    cout << "  fusing uu[" << i+1<< "] and uu[" << ns-2-i << "]" << endl; cout.flush();
    BondType bb0, bb1;
    uu[i].get_bond(2, bb0);   bb0.invert_bonddir();
    uu[ns-1-i].get_bond(2, bb1); bb1.invert_bonddir();
    uu[i+1].fuse(bb0, bb[1]);
    uu[ns-2-i].fuse(bb[2], bb1);
    /*
    cout<<"print uu "<<i+1<<endl;
    uu[i+1].print();
    cout<<"print uu "<<ns-2-i<<endl;
    uu[ns-2-i].print();
    */
    cout << "  condition check: (i+2)*2*physpn=" << (i+2)*2*physpn << " totspin=" << totspin << endl; cout.flush();
    if (((i + 2) * 2 * physpn >= totspin && i >= 1) || i == ns / 2 - 2) {
      cout << "  prepare_input_vector_initial(" << i+1 << "," << ns-2-i << ")" << endl; cout.flush();
      prepare_input_vector_initial(i + 1, ns - 2 - i, vec);
      cout << "  prepare_site_operator_from_left(" << i+1 << ")" << endl; cout.flush();
      prepare_site_operator_from_left(i + 1);
      cout << "  prepare_site_operator_from_right(" << ns-2-i << ")" << endl; cout.flush();
      prepare_site_operator_from_right(ns - 2 - i);
      cout<<"  done prepare_site_operator"<<endl;
      vec.initialize_input_vector();
      vec.normalize_vector();
      cout<<"  begin lanczos_solve_eigenvector"<<endl;
      lanczos_solve_eigenvector_idmrg(i + 1, ns - 2 - i, vec);
      cout<<"  done lanczos_solve_eigenvector"<<endl;
      if (i == ns / 2 - 2){
        vec.svd(uu[i+1], 1.0, uu[ns-2-i], 0.0, ww[exci]);
      }
      else{
	/*
	cout<<"uu["<<i+1<<"]"<<endl;
	uu[i+1].print();
	cout<<"uu["<<ns-2-i<<"]"<<endl;
	uu[ns-2-i].print();
	*/
        vec.svd(uu[i+1], 0.0, uu[ns-2-i], 0.0, ww[exci]);
	/*
	cout<<"uu["<<i+1<<"]"<<endl;
	uu[i+1].print();
	cout<<"uu["<<ns-2-i<<"]"<<endl;
	uu[ns-2-i].print();
	cout<<"sucessful return from svd"<<endl;
	exit(0);
	*/
      }
    }
  }

  cout << "done iDMRG  E=" << setprecision(14) << gs_enr[exci] << endl;
  delete[] bb;
}

// ── prepare_sweep ─────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::prepare_sweep() {
  int i;
  BondType bb[4];
  int q0 = 0, d1 = 1, qp = physpn, qt = totspin;
  if constexpr (Sym::has_cgc){
    bb[0].set(1, 1, &q0, &d1);   // left vacuum bond
    bb[1].set(1, 1, &qp, &d1);   // physical spin bond (Q=physpn)
    uu[0].fuse(bb[0], bb[1]);

    bb[2].set(1, 1, &qp, &d1);   // physical spin bond at right end
    bb[3].set(1, 1, &qt, &d1);   // right target-sector bond
    uu[ns-1].fuse(bb[2], bb[3]);
  }
  else {//this is a setup for spin-1/2 U1 case
    int cval[2], bdim[2];
    cval[0] = 0;
    bdim[0] = 1;
    bb[0].set(1, 1, cval, bdim);
    cval[0] = -1;
    cval[1] = 1;
    bdim[0] = 1;
    bdim[1] = 1;
    bb[1].set(1, 2, cval, bdim);
    uu[0].fuse(bb[0],bb[1]);
    cval[0] = -1;
    cval[1] = 1;
    bdim[0] = 1;
    bdim[1] = 1;
    bb[2].set(1, 2, cval, bdim);
    cval[0] = qt;
    bdim[0] = 1;
    bb[3].set(-1, 1, cval, bdim);
    uu[ns-1].fuse(bb[2],bb[3]);
  }

  for (i = 0; i < ns / 2 - 1; ++i) {
    //cout<<"i="<<i<<endl;
    if (i < ns / 2 - 2) {
      prepare_site_operator_from_left(i, 0);
      prepare_site_operator_from_right(ns - 1 - i, 0);
    } else {
      prepare_site_operator_from_left(i, 0);
      prepare_site_operator_from_right(ns - 1 - i, 1);
    }
  }
}

// ── wavefunc_transformation ───────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::wavefunc_transformation(int i, int flag) {
  int j;
  if (flag == 0) {
    uu[i].right2left_vectran();
    for (j = 0; j < max_exci; ++j)
      orth[j][i].right2left_vectran();
  } else {
    uu[i].left2right_vectran();
    for (j = 0; j < max_exci; ++j)
      orth[j][i].left2right_vectran();
  }
}

// ── sweep ─────────────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::sweep() {
  int i, j;
  TensorType vec;

  // ── Sweep left ──
  for (i = ns / 2; i >= nfree + 2; --i) {
    cout << "sweep_left i=" << i << endl; cout.flush();
    wavefunc_transformation(i - 1, 1);
    cout << "  after wavefunc_transformation(" << i-1 << ",1)" << endl; cout.flush();
    prepare_site_operator_from_right(i, 1);
    cout << "  after prepare_right(" << i << ")" << endl; cout.flush();
    prepare_input_vector(i - 2, i - 1, vec);
    cout << "  after prepare_input(" << i-2 << "," << i-1 << ")" << endl; cout.flush();
    prepare_site_operator_from_left(i - 2);
    cout << "  after prepare_left(" << i-2 << ")" << endl; cout.flush();
    prepare_site_operator_from_right(i - 1);
    cout << "  after prepare_right(" << i-1 << ")" << endl; cout.flush();
    lanczos_solve_eigenvector_idmrg(i - 2, i - 1, vec);
    cout << "  after lanczos" << endl; cout.flush();
    if (i > nfree + 2) vec.svd(uu[i-2], 1.0, uu[i-1], 0.0, ww[exci]);
    else               vec.svd(uu[i-2], 0.0, uu[i-1], 1.0, ww[exci]);
    cout << "  after svd" << endl; cout.flush();
  }

  // ── Sweep right ──
  for (i = nfree; i < ns - nfree - 2; ++i) {
    cout << "sweep_right i=" << i << endl; cout.flush();
    wavefunc_transformation(i + 1, 0);
    cout << "  after wavefunc_transformation(" << i+1 << ",0)" << endl; cout.flush();
    prepare_site_operator_from_left(i, 1);
    cout << "  after prepare_left(" << i << ")" << endl; cout.flush();
    prepare_input_vector(i + 1, i + 2, vec);
    cout << "  after prepare_input(" << i+1 << "," << i+2 << ")" << endl; cout.flush();
    prepare_site_operator_from_left(i + 1);
    cout << "  after prepare_left(" << i+1 << ")" << endl; cout.flush();
    prepare_site_operator_from_right(i + 2);
    cout << "  after prepare_right(" << i+2 << ")" << endl; cout.flush();
    lanczos_solve_eigenvector_idmrg(i + 1, i + 2, vec);
    cout << "  after lanczos" << endl; cout.flush();
    if (i < ns - nfree - 3) vec.svd(uu[i+1], 0.0, uu[i+2], 1.0, ww[exci]);
    else                     vec.svd(uu[i+1], 1.0, uu[i+2], 0.0, ww[exci]);
    cout << "  after svd" << endl; cout.flush();
  }

  // ── Return sweep ──
  for (i = ns - nfree - 1; i >= ns / 2 + 1; --i) {
    cout << "sweep_return i=" << i << endl; cout.flush();
    wavefunc_transformation(i - 1, 1);
    cout << "  after wavefunc_transformation(" << i-1 << ",1)" << endl; cout.flush();
    prepare_site_operator_from_right(i, 1);
    cout << "  after prepare_right(" << i << ")" << endl; cout.flush();
    prepare_input_vector(i - 2, i - 1, vec);
    cout << "  after prepare_input(" << i-2 << "," << i-1 << ")" << endl; cout.flush();
    prepare_site_operator_from_left(i - 2);
    cout << "  after prepare_left(" << i-2 << ")" << endl; cout.flush();
    prepare_site_operator_from_right(i - 1);
    cout << "  after prepare_right(" << i-1 << ")" << endl; cout.flush();
    lanczos_solve_eigenvector_idmrg(i - 2, i - 1, vec);
    cout << "  after lanczos" << endl; cout.flush();
    for (j = 0; j < bondd; ++j) ww[exci][j] = 0.0;
    vec.svd(uu[i-2], 1.0, uu[i-1], 0.0, ww[exci]);
    cout << "  after svd" << endl; cout.flush();
  }

  cout << "before save_mps2" << endl; cout.flush();
  save_mps2(exci);
  save_ww();
  save_enr();
}

// ── I/O: save_mps2 ────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::save_mps2(int exci_idx) {
  int i, j, k, ndata, ncgc;
  char name[256], base[200], len[16], dim[16], sec[16], pos[16], exc[16];
  std::ofstream fout;
  Scalar* ptr; double* ptr1;

  sprintf(len, "%d", ns);
  sprintf(sec, "%d", totspin);
  sprintf(dim, "%d", bondd);
  sprintf(exc, "%d", exci_idx);
  snprintf(base, sizeof(base), "uu-%s-%s-%s-%s-", len, dim, sec, exc);

  for (i = 0; i < ns; ++i) {
    TensorType* tptr = &uu[i];
    sprintf(pos, "%d", i);
    snprintf(name, sizeof(name), "%s%s.dat", base, pos);
    fout.open(name);
    if (fout.is_open()) {
      fout << tptr->get_nbond() << "\t" << tptr->get_locspin() << "\n";
      int nb = tptr->get_nbond();
      for (j = 0; j < nb; ++j) {
        fout << tptr->get_bonddir(j) << "\t" << tptr->get_nmoment(j) << "\n";
        int nc = tptr->get_nmoment(j);
        for (k = 0; k < nc; ++k) fout << tptr->get_angularmoment(j, k) << "\t";
        fout << "\n";
        for (k = 0; k < nc; ++k) fout << tptr->get_bonddim(j, k) << "\t";
        fout << "\n";
      }
      fout.close();
    }
    snprintf(name, sizeof(name), "%s%s.bin", base, pos);
    fout.open(name, std::ios::binary);
    if (fout.is_open()) {
      tptr->get_nelement(ndata, ncgc);
      ptr  = new Scalar[ndata + 1]();
      ptr1 = new double[ncgc  + 1]();
      tptr->get_elems(ptr, ptr1);
      fout.write(reinterpret_cast<char*>(ptr),  ndata * sizeof(Scalar));
      fout.write(reinterpret_cast<char*>(ptr1), ncgc  * sizeof(double));
      delete[] ptr; delete[] ptr1;
      fout.close();
    }
  }
}

// ── I/O: save_enr ────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::save_enr() {
  char name[256];
  sprintf(name, "enr-%d-%d-%d-%d.dat", ns, bondd, totspin, exci);
  std::ofstream fout(name, std::ios::app);
  if (fout.is_open()) {
    fout << setprecision(16) << gs_enr[exci] << "\n";
  }
}

// ── I/O: read_enr ────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::read_enr(int read1, int read2) {
  char name[256];
  sprintf(name, "enr-%d-%d-%d-%d.dat", ns, read1, totspin, exci);
  std::ifstream fin(name);
  if (fin.is_open()) {
    double e;
    while (fin >> e) gs_enr[exci] = e;
    cout << "read_enr: gs_enr[" << exci << "]=" << gs_enr[exci] << endl;
  }
  if (exci > 0)
    for (int idx = 0; idx < exci; idx++){
      sprintf(name, "enr-%d-%d-%d-%d.dat", ns, read2, totspin, idx);
      std::ifstream fin(name);
      if (fin.is_open()) {
	double e;
	while (fin >> e) gs_enr[idx] = e;
	cout << "read_enr: gs_enr[" << idx << "]=" << gs_enr[idx] << endl;
      }
    }
}

// ── I/O: read_mps ────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
bool DMRG<Scalar,Sym>::read_mps(int read1) {
  int i, j, k, nb, nc, lc, dir, ndata, ncgc;
  char name[256], base[200], len[16], dim[16], sec[16], pos[16], exc[16];
  std::ifstream fin;
  bool pass = true;
  typename Sym::StructType strct;

  sprintf(len, "%d", ns);
  sprintf(sec, "%d", totspin);
  sprintf(dim, "%d", read1);
  sprintf(exc, "%d", exci);
  snprintf(base, sizeof(base), "uu-%s-%s-%s-%s-", len, dim, sec, exc);

  for (i = 0; i < ns; ++i) {
    sprintf(pos, "%d", i);
    snprintf(name, sizeof(name), "%s%s.dat", base, pos);
    fin.open(name);
    if (!fin.is_open()) { pass = false; continue; }
    BondType* bonds;
    fin >> nb >> lc;
    bonds = new BondType[nb];
    for (j = 0; j < nb; ++j) {
      fin >> dir >> nc;
      int* angm = new int[nc]; int* bdim = new int[nc];
      for (k = 0; k < nc; ++k) fin >> angm[k];
      for (k = 0; k < nc; ++k) fin >> bdim[k];
      bonds[j].set(dir, nc, angm, bdim);
      delete[] angm; delete[] bdim;
    }
    strct.set_struct(nb, lc, bonds);
    delete[] bonds;
    fin.close();

    snprintf(name, sizeof(name), "%s%s.bin", base, pos);
    fin.open(name, std::ios::binary);
    if (fin.is_open()) {
      strct.get_nelement(ndata, ncgc);
      Scalar* ptr  = new Scalar[ndata + 1]();
      double* ptr1 = new double[ncgc  + 1]();
      fin.read(reinterpret_cast<char*>(ptr),  ndata * sizeof(Scalar));
      fin.read(reinterpret_cast<char*>(ptr1), ncgc  * sizeof(double));
      uu[i].set(strct, ptr, ptr1);
      delete[] ptr; delete[] ptr1;
      fin.close();
    } else { pass = false; }
  }
  return pass;
}

template<typename Scalar, typename Sym>
bool DMRG<Scalar,Sym>::read_mps(int read1, int read2) {
  bool pass = read_mps(read1);

  if (exci > 0) {
    int i, j, k, nb, nc, lc, dir, ndata, ncgc;
    char name[256], base[200], len[16], dim[16], sec[16], pos[16], exc[16];
    std::ifstream fin;
    typename Sym::StructType strct;

    sprintf(len, "%d", ns);
    sprintf(sec, "%d", totspin);
    sprintf(dim, "%d", read2);

    for (int idx = 0; idx < exci; ++idx) {
      bool idx_pass = true;
      sprintf(exc, "%d", idx);
      snprintf(base, sizeof(base), "uu-%s-%s-%s-%s-", len, dim, sec, exc);

      for (i = 0; i < ns; ++i) {
        sprintf(pos, "%d", i);
        snprintf(name, sizeof(name), "%s%s.dat", base, pos);
        fin.open(name);
        if (!fin.is_open()) { idx_pass = false; continue; }
        BondType* bonds;
        fin >> nb >> lc;
        bonds = new BondType[nb];
        for (j = 0; j < nb; ++j) {
          fin >> dir >> nc;
          int* angm = new int[nc]; int* bdim = new int[nc];
          for (k = 0; k < nc; ++k) fin >> angm[k];
          for (k = 0; k < nc; ++k) fin >> bdim[k];
          bonds[j].set(dir, nc, angm, bdim);
          delete[] angm; delete[] bdim;
        }
        strct.set_struct(nb, lc, bonds);
        delete[] bonds;
        fin.close();

        snprintf(name, sizeof(name), "%s%s.bin", base, pos);
        fin.open(name, std::ios::binary);
        if (fin.is_open()) {
          strct.get_nelement(ndata, ncgc);
          Scalar* ptr  = new Scalar[ndata + 1]();
          double* ptr1 = new double[ncgc  + 1]();
          fin.read(reinterpret_cast<char*>(ptr),  ndata * sizeof(Scalar));
          fin.read(reinterpret_cast<char*>(ptr1), ncgc  * sizeof(double));
          orth[idx][i].set(strct, ptr, ptr1);
          delete[] ptr; delete[] ptr1;
          fin.close();
        } else { idx_pass = false; }
      }
      if (idx_pass)
        cout << "read_mps: orth[" << idx << "] read successfully" << endl;
    }
  }
  read_ww(read1, read2);
  read_enr(read1, read2);
  return pass;
}

// ── I/O: save_ww / read_ww ───────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::save_ww() {
  char name[256];
  sprintf(name, "ww-%d-%d-%d-%d.bin", ns, bondd, totspin, exci);
  std::ofstream fout(name, std::ios::binary);
  if (fout.is_open()) {
    fout.write(reinterpret_cast<char*>(ww[exci]),
               bondd * phdim * sizeof(double));
  }
}

template<typename Scalar, typename Sym>
void DMRG<Scalar,Sym>::read_ww(int read1, int read2) {
  char name[256];
  sprintf(name, "ww-%d-%d-%d-%d.bin", ns, read1, totspin, exci);
  std::ifstream fin(name, std::ios::binary);
  if (fin.is_open()) {
    fin.read(reinterpret_cast<char*>(ww[exci]),
             read1 * sizeof(double));
  } else {
    for (int i = 0; i < read1; ++i) ww[exci][i] = 1.0;
  }
  if (exci > 0)
    for (int idx = 0; idx < exci; idx++){
      sprintf(name, "ww-%d-%d-%d-%d.bin", ns, read2, totspin, idx);
      std::ifstream fin(name, std::ios::binary);
      if (fin.is_open()) {
	fin.read(reinterpret_cast<char*>(ww[idx]),
		 read2 * sizeof(double));
      } else {
	for (int i = 0; i < read1; ++i) ww[idx][i] = 1.0;
      }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Lanczos<Scalar, Sym>
// ═════════════════════════════════════════════════════════════════════════════

template<typename Scalar, typename Sym>
Lanczos<Scalar,Sym>::Lanczos()
  : nrep(0), mlanc(0), neig(0),
    aal(nullptr), nnl(nullptr), eig(nullptr), vec(nullptr),
    enrexp(0), ff(nullptr)
{}

template<typename Scalar, typename Sym>
Lanczos<Scalar,Sym>::~Lanczos() {
  delete[] eig;
  delete[] vec;
  delete[] nnl;
  delete[] aal;
  delete[] ff;
}

// ── initialize_lanczos ────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void Lanczos<Scalar,Sym>::initialize_lanczos(TensorType& vin, int ml, int ng) {
  mlanc = ml;
  neig  = ng;
  aal = new double[mlanc]();
  nnl = new double[mlanc + 1]();
  ff  = new TensorType[mlanc + 1];
  eig = new double[mlanc * 2]();
  vec = new double[mlanc * mlanc]();
  ff[0] = vin;
  ff[0].normalize_vector();
}

// ── diatridiag (LAPACK dstev) ─────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void Lanczos<Scalar,Sym>::diatridiag(int n) {
  double *d, *e, *work;
  int info, nn = n;
  char jobz = 'V';
  d    = new double[n];
  e    = new double[n];
  work = new double[2 * n + 1];
  for (int i = 0; i < n; ++i) {
    d[i] = aal[i];
    if (i < n - 1) e[i] = nnl[i + 1];
  }
  dstev_(&jobz, &nn, d, e, vec, &nn, work, &info);
  for (int i = 0; i < n; ++i) eig[i] = d[i];
  delete[] d; delete[] e; delete[] work;
}

// ── diag_op (calls DMRG H*v) ─────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void Lanczos<Scalar,Sym>::diag_op(int il, int ir, TensorType& v_in, TensorType& v_out, int /*nsite*/) {
  DMRG<Scalar,Sym>* chain = g_chain_ptr<Scalar,Sym>;
  chain->hamiltonian_vector_multiplication_idmrg(il, ir, v_in, v_out);
}

// ── check_eigenvector ─────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void Lanczos<Scalar,Sym>::check_eigenvector(int il, int ir, TensorType& vout,
                                              double& eigval, double& olap, int nsite) {
  TensorType hvec;
  double nor = std::sqrt(std::real(vout.inner_prod_c(vout)));
  vout /= nor;
  diag_op(il, ir, vout, hvec, nsite);
  double prod    = std::sqrt(std::real(hvec.inner_prod_c(hvec)));
  dcmplex overlap = vout.inner_prod_c(hvec);
  eigval = prod;
  olap   = std::abs(overlap) / prod;
}

// ── compute_eigenvector ───────────────────────────────────────────────────────
// Reconstruct Ritz vector from Krylov basis using eigencoefficients.
// mlanc_curr: number of Lanczos steps taken; which_coeff: eigenvector index (0=GS).

template<typename Scalar, typename Sym>
void Lanczos<Scalar,Sym>::compute_eigenvector(TensorType& vout, int state_idx, int which_coeff) {
  TensorType tmp;
  int mc = (which_coeff > 0) ? which_coeff : mlanc;
  for (int j = 0; j < mc; ++j) {
    double coeff = vec[j + state_idx * mc];   // eigenvector for state_idx
    if (j == 0) {
      vout = ff[j];
      vout *= coeff;
    } else {
      tmp = ff[j];
      tmp *= coeff;
      vout += tmp;
    }
  }
}

template<typename Scalar, typename Sym>
void Lanczos<Scalar,Sym>::compute_eigenvector(int il, int ir, TensorType& vout,
                                               int mlanc_curr, int& stp, int nsite) {
  compute_eigenvector(vout, 0, mlanc_curr);
  double eigval, olap;
  check_eigenvector(il, ir, vout, eigval, olap, nsite);
  stp = (olap > 1.0 - 1e-10) ? 1 : 0;
}

template<typename Scalar, typename Sym>
void Lanczos<Scalar,Sym>::lanczos1(int il, int ir, TensorType& vout) {
  int m;
  const double tol = 1.e-6;
  double ee, ov;
  TensorType tmp, v2;
  const int nsite = 2;

  for (int i = 0; i < mlanc;     ++i) aal[i] = 0.0;
  for (int i = 0; i < mlanc + 1; ++i) nnl[i] = 0.0;

  diag_op(il, ir, ff[0], ff[1], nsite);
  aal[0] = std::real(ff[1].inner_prod_c(ff[0]));
  tmp = ff[0];  tmp *= aal[0];
  ff[1] -= tmp;
  nnl[1] = std::sqrt(std::real(ff[1].inner_prod_c(ff[1])));
  ff[1] /= nnl[1];

  if (mlanc == 2) {
    if (nnl[1] < tol) { diatridiag(mlanc - 1); compute_eigenvector(vout, 0, mlanc-1); return; }
    diag_op(il, ir, ff[1], ff[2], nsite);
    aal[1] = std::real(ff[2].inner_prod_c(ff[1]));
    diatridiag(mlanc);
    compute_eigenvector(vout, 0, mlanc);
    return;
  }

  for (m = 2; m < mlanc; ++m) {
    diag_op(il, ir, ff[m-1], ff[m], nsite);
    aal[m-1] = std::real(ff[m].inner_prod_c(ff[m-1]));
    tmp = ff[m-1]; tmp *= aal[m-1];  ff[m] -= tmp;
    tmp = ff[m-2]; tmp *= nnl[m-1];  ff[m] -= tmp;

    // Re-orthogonalise against previous Krylov vectors
    for (int j = 0; j < m; ++j) {
      Scalar q1 = ff[j].inner_prod_c(ff[m]);
      tmp = ff[j]; tmp *= q1; ff[m] -= tmp;
    }

    nnl[m] = std::sqrt(std::real(ff[m].inner_prod_c(ff[m])));
    ff[m]  /= nnl[m];
    diatridiag(m);
    if (nnl[m] < tol) {
      compute_eigenvector(vout, 0, m);
      check_eigenvector(il, ir, vout, ee, ov, nsite);
      cout << "lanczos1 early exit m=" << m << " E=" << ee << " ov=" << ov << endl;
      return;
    }
    if (m % 4 == 0) {
      compute_eigenvector(vout, 0, m);
      check_eigenvector(il, ir, vout, ee, ov, nsite);
      cout << m << "\tE=" << setprecision(12) << ee << "\tov=" << ov << endl;
      if (std::abs(ov - 1.0) < 1e-8) return;
    }
  }

  diag_op(il, ir, ff[mlanc-1], ff[mlanc], nsite);
  aal[mlanc-1] = std::real(ff[mlanc].inner_prod_c(ff[mlanc-1]));
  diatridiag(mlanc);
  compute_eigenvector(vout, 0, mlanc);
  check_eigenvector(il, ir, vout, ee, ov, nsite);
  cout << mlanc << "\tE=" << setprecision(12) << ee << "\tov=" << ov << endl;
}

// ── lanczos2 (re-orthogonalisation for excited states) ───────────────────────

template<typename Scalar, typename Sym>
void Lanczos<Scalar,Sym>::lanczos2(int il, int ir, TensorType& vout, int nsite) {
  int m;
  const double tol = 1.e-6;
  TensorType tmp, v2;

  for (int i = 0; i < mlanc;     ++i) aal[i] = 0.0;
  for (int i = 0; i < mlanc + 1; ++i) nnl[i] = 0.0;

  diag_op(il, ir, ff[0], ff[1], nsite);
  aal[0] = std::real(ff[1].inner_prod_c(ff[0]));
  tmp = ff[0]; tmp *= aal[0]; ff[1] -= tmp;
  nnl[1] = std::sqrt(std::real(ff[1].inner_prod_c(ff[1])));
  ff[1] /= nnl[1];

  if (mlanc == 2) {
    if (nnl[1] < tol) { diatridiag(mlanc-1); compute_eigenvector(vout, 0, mlanc-1); return; }
    diag_op(il, ir, ff[1], ff[2], nsite);
    aal[1] = std::real(ff[2].inner_prod_c(ff[1]));
    diatridiag(mlanc);
    compute_eigenvector(vout, 0, mlanc);
    return;
  }

  for (m = 2; m < mlanc; ++m) {
    diag_op(il, ir, ff[m-1], ff[m], nsite);
    aal[m-1] = std::real(ff[m].inner_prod_c(ff[m-1]));
    tmp = ff[m-1]; tmp *= aal[m-1]; ff[m] -= tmp;
    tmp = ff[m-2]; tmp *= nnl[m-1]; ff[m] -= tmp;
    for (int j = 0; j < m; ++j) {
      Scalar q1 = ff[j].inner_prod_c(ff[m]);
      tmp = ff[j]; tmp *= q1; ff[m] -= tmp;
    }
    nnl[m] = std::sqrt(std::real(ff[m].inner_prod_c(ff[m])));
    ff[m] /= nnl[m];
    if (nnl[m] < tol) return;
    diatridiag(m);
    compute_eigenvector(vout, 0, m);
    vout.normalize_vector();
    v2 = vout;
  }
  diag_op(il, ir, ff[mlanc-1], ff[mlanc], nsite);
  aal[mlanc-1] = std::real(ff[mlanc].inner_prod_c(ff[mlanc-1]));
  diatridiag(mlanc);
  compute_eigenvector(vout, 0, mlanc);
}

// ── Explicit instantiations ───────────────────────────────────────────────────

template class DMRG<double,  SU2Symmetry>;
template class DMRG<dcmplex, SU2Symmetry>;
template class DMRG<double,  U1Symmetry>; 
template class DMRG<dcmplex, U1Symmetry>; 

template class Lanczos<double,  SU2Symmetry>;
template class Lanczos<dcmplex, SU2Symmetry>;
template class Lanczos<double,  U1Symmetry>;
template class Lanczos<dcmplex, U1Symmetry>;
