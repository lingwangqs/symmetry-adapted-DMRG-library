/**
 * src/block_tensor_inst.cpp — BlockTensor<Scalar, Sym> implementation.
 * Ported from mps_u1/tensor_u1.cpp + tensor_u1_dmrg_src.cpp (U1)
 * and tensor_su2.cpp + tensor_su2_dmrg_src.cpp (SU2).
 *
 * The U1 and SU2 paths are unified via `if constexpr (Sym::has_cgc)`.
 */

#include "../block_tensor.hpp"
#include "../cgc_tables.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <vector>

// Pointer to the pre-built CGC factor tables (defined in main, set before use).
extern CGCTables* g_cgc_tables;

extern int max_dcut;
extern int psize;

// Forward declarations for free-function templates defined in dense_inst.cpp
// tb parameters are always Dense<double> (CGC blocks are real-valued)
template<typename Scalar>
bool sum_direct_product(Dense<Scalar>& ta1, Dense<double>& tb1,
                        Dense<Scalar>& ta2, Dense<double>& tb2);
template<typename Scalar>
void sum_direct_product(Dense<Scalar>& ta1, Dense<double>& tb1,
                        Dense<Scalar>& ta2, Dense<double>& tb2,
                        Dense<Scalar>& ta3, Dense<double>& tb3);

using std::cout;
using std::endl;

void sort2(int n, double* key, int* val) {
    std::vector<std::pair<double,int>> tmp(n);
    for (int i = 0; i < n; ++i)
        tmp[i] = {key[i], val[i]};

    std::sort(tmp.begin(), tmp.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    for (int i = 0; i < n; ++i) {
        key[i] = tmp[i].first;
        val[i] = tmp[i].second;
    }
}
// ── Helper: uniform get_tensor_argument ───────────────────────────────────────
// For U1, cdims is unused.  For SU2, cdims = 2J+1.
template<typename Sym>
static bool get_ta(const typename Sym::StructType& s, int i, int* angm, int* bdim, int* cdim) {
  if constexpr (Sym::has_cgc)
    return s.get_tensor_argument(i, angm, bdim, cdim);
  else {
    bool r = s.get_tensor_argument(i, angm, bdim);
    for (int k = 0; k < s.get_nbond(); ++k) cdim[k] = 1;
    return r;
  }
}

// ── Constructor / destructor / clean ─────────────────────────────────────────

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>::BlockTensor()
  : nbond(0), nten(0), locspin(0), data_blocks(nullptr), cgc_blocks(nullptr) {}

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>::~BlockTensor() { clean(); }

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>::BlockTensor(const BlockTensor& o)
  : nbond(0), nten(0), locspin(0), data_blocks(nullptr), cgc_blocks(nullptr)
{ *this = o; }

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::clean() {
  delete[] data_blocks; data_blocks = nullptr;
  delete[] cgc_blocks;  cgc_blocks  = nullptr;
  cgc.clean();
  nbond = nten = locspin = 0;
}

// ── set(): initialise from bond array ────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::set(int nb, int ls, const BondType* bonds, Dense<Scalar>** blocks) {
  //this function is used at building operators, and building uu vv from BlockTensor::svd
  clean();
  nbond   = nb;
  locspin = ls;
  cgc.set_struct(nb, ls, bonds);
  nten = cgc.get_nten();
  data_blocks = new Dense<Scalar>[nten];
  if constexpr (Sym::has_cgc)
    cgc_blocks = new Dense<double>[nten];
  if (blocks) {
    int *angm, *bdim;
    angm = new int[nbond]; bdim = new int[nbond];
    int j = 0;
    for (int i = 0; i < nten; ++i){
      bool check = cgc.get_tensor_argument(i, angm, bdim);
      if (check) {
	data_blocks[i].copy(*blocks[j]);
	if constexpr (Sym::has_cgc)
	  if (nbond == 3)
	    cgc_blocks[i].make_cgc(angm[0], angm[1], angm[2]);
	  else if (nbond == 2)
	    cgc_blocks[i].make_identity(angm[0]);
	  else {cout<<"BlockTensor<Scalar, Sym>::set: wrong BlockTensor Struct"<<endl; exit(1);}
	j++;
      }
    }
    delete []angm; delete []bdim;
  }
}

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::set(const StructType& s, const Scalar* data_in, const double* cgc_in){
  //this function is used at building wavefunction from stored disk data
  clean();
  cgc     = s;
  nbond   = s.get_nbond();
  locspin = s.get_locspin();
  nten    = s.get_nten();
  data_blocks = new Dense<Scalar>[nten];
  if constexpr (Sym::has_cgc) cgc_blocks = new Dense<double>[nten];
  // Populate blocks from flat arrays
  int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];
  int offset_d = 0, offset_c = 0;
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    if (data_in) {
      int sz = 1;
      for (int j = 0; j < nbond; ++j) sz *= bdim[j];
      data_blocks[i].copy(nbond, bdim, data_in + offset_d);
      offset_d += sz;
    }
    if constexpr (Sym::has_cgc) {
      if (cgc_in) { 
	// sz_c = product of cgcdim
	int sz_c = 1;
	for (int j = 0; j < nbond; ++j) sz_c *= cdim[j];
        cgc_blocks[i].copy(nbond, cdim, cgc_in + offset_c);
	offset_c += sz_c;
      }
      else{
	if (nbond == 3)
	  cgc_blocks[i].make_cgc(angm[0], angm[1], angm[2]);
	else if (nbond == 2)
	  cgc_blocks[i].make_identity(angm[0]);
	else {cout<<"BlockTensor<Scalar, Sym>::set: wrong BlockTensor Struct"<<endl; exit(1);}
      }
    }
  }
  delete[] angm; delete[] bdim; delete[] cdim;
}

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::set(const StructType& s, const Scalar* data_in) {
  set(s, data_in, nullptr);
}

// ── get_tensor_argument (uniform 3-array interface) ──────────────────────────

template<typename Scalar, typename Sym>
bool BlockTensor<Scalar, Sym>::get_tensor_argument(int b, int* moment, int* bdim, int* cdim) const {
  //this function check momentum consistency of blocktensor index b, return its corresponding moment, bdim, and cdim
  return cgc.get_tensor_argument(b, moment, bdim, cdim);
}

template<typename Scalar, typename Sym>
bool BlockTensor<Scalar, Sym>::get_tensor_argument(int b, int* moment, int* bdim) const {
  //this function check momentum consistency of blocktensor index b, return its corresponding moment, bdim, and cdim
  return cgc.get_tensor_argument(b, moment, bdim);
}

// ── is_null ───────────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
bool BlockTensor<Scalar, Sym>::is_null(int b) const {
  return data_blocks[b].is_null();
}

template<typename Scalar, typename Sym>
bool BlockTensor<Scalar, Sym>::is_null() const { return (nbond == 0); }

template<typename Scalar, typename Sym>
bool BlockTensor<Scalar, Sym>::check_angularmoments(const int* angm) const {
  return cgc.check_angularmoments(angm);
}

// ── Arithmetic operators ──────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::operator=(double s) {
  for (int i = 0; i < nten; ++i)
    if (!data_blocks[i].check_null()) data_blocks[i] = Scalar(s);
  return *this;
}

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::operator=(const BlockTensor& o) {
  if (this == &o) return *this;
  clean();
  cgc     = o.cgc;
  nbond   = o.nbond;
  nten    = o.nten;
  locspin = o.locspin;
  data_blocks = new Dense<Scalar>[nten];
  for (int i = 0; i < nten; ++i)
    if (!o.data_blocks[i].check_null())
      data_blocks[i] = o.data_blocks[i];
  if constexpr (Sym::has_cgc) {
    cgc_blocks = new Dense<double>[nten];
    for (int i = 0; i < nten; ++i)
      if (!o.cgc_blocks[i].check_null())
	cgc_blocks[i] = o.cgc_blocks[i];
  }
  return *this;
}

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::operator+=(const BlockTensor& o) {
  int* angm = new int[nbond], *bdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    if (data_blocks[i].check_null() && !o.data_blocks[i].check_null())
      data_blocks[i] = o.data_blocks[i];
    else if (!data_blocks[i].check_null() && !o.data_blocks[i].check_null()) {
      if constexpr (Sym::has_cgc) {
	double rat;
	bool check;
	check = o.cgc_blocks[i].is_proportional_to(cgc_blocks[i], rat);
	if (!check || std::fabs(rat - 1.) > 1e-12) {
	  cgc_blocks[i].print();
	  cout<<"=========="<<endl;
	  o.cgc_blocks[i].print();
	  cout<<"check="<<check<<"\t rat="<<rat<<endl;
	  exit(0);
	  cout<<"BlockTensor::operator+= input tensors wrong"<<endl;
	  exit(1);
	}
      }
      data_blocks[i] += o.data_blocks[i];
    }
  }
  delete[] angm; delete[] bdim;
  return *this;
}

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::operator-=(const BlockTensor& o) {
  int* angm = new int[nbond], *bdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    if (data_blocks[i].check_null() && !o.data_blocks[i].check_null()) {
      data_blocks[i] = o.data_blocks[i];
      data_blocks[i] *= -1.0;
    }
    else if (!data_blocks[i].check_null() && !o.data_blocks[i].check_null()) {
      if constexpr (Sym::has_cgc) {
	double rat;
	bool check;
	check = o.cgc_blocks[i].is_proportional_to(cgc_blocks[i], rat);
	if (!check || std::fabs(rat - 1.) > 1e-12) {
	  cout<<"BlockTensor::operator-= input tensors wrong"<<endl;
	  exit(1);
	}
      }
      data_blocks[i] -= o.data_blocks[i];
    }
  }
  delete[] angm; delete[] bdim;
  return *this;
}

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::operator*=(double s) {
  for (int i = 0; i < nten; ++i)
    if (!data_blocks[i].check_null()) data_blocks[i] *= s;
  return *this;
}

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::operator*=(dcmplex s) {
  if constexpr (ScalarTraits<Scalar>::is_complex){
    for (int i = 0; i < nten; ++i)
      if (!data_blocks[i].check_null()) data_blocks[i] *= s;
  }
  else {
    cout<<"BlockTensor::operator*= can not multiply complex number"<<endl;
    exit(1);
  }
  return *this;
}

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::operator/=(double s) {
  for (int i = 0; i < nten; ++i)
    if (!data_blocks[i].check_null()) data_blocks[i] /= s;
  return *this;
}

template<typename Scalar, typename Sym>
bool BlockTensor<Scalar, Sym>::operator==(const BlockTensor& o) const { return !(*this != o); }

template<typename Scalar, typename Sym>
bool BlockTensor<Scalar, Sym>::operator!=(const BlockTensor& o) const {
  if (cgc != o.cgc || nbond != o.nbond || nten != o.nten) return true;
  for (int i = 0; i < nten; ++i)
    if (data_blocks[i] != o.data_blocks[i]) return true;
  return false;
}

// ── Inner product / normalization ─────────────────────────────────────────────

template<typename Scalar, typename Sym>
Scalar BlockTensor<Scalar, Sym>::inner_prod_c(const BlockTensor& o) const {
  //note that inner_prod_c call Dense.inner_prod_c, then call ScalarTraits.dotc, which gives x^{dagger} * y, double check to make sure it is correct when used
  Scalar prod = 0;
  int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    if (!data_blocks[i].check_null() && !o.data_blocks[i].check_null())
      prod += data_blocks[i].inner_prod_c(o.data_blocks[i]);
  }
  delete[] angm; delete[] bdim; delete[] cdim;
  return prod;
}

template<typename Scalar, typename Sym>
Scalar BlockTensor<Scalar, Sym>::inner_prod_u(const BlockTensor& o) const {
  //note that inner_prod_u call Dense.inner_prod_u, then call ScalarTraits.dotu, which gives x * y, double check to make sure it is correct when used
  Scalar prod = 0;
  int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    if (!data_blocks[i].check_null() && !o.data_blocks[i].check_null())
      prod += data_blocks[i].inner_prod_u(o.data_blocks[i]);
  }
  delete[] angm; delete[] bdim; delete[] cdim;
  return prod;
}

template<typename Scalar, typename Sym>
Scalar BlockTensor<Scalar, Sym>::take_trace() const {
  Scalar tr = 0;
  int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    if (!data_blocks[i].check_null() && nbond == 2 && angm[0] == angm[1] && bdim[0] == bdim[1])
      tr += data_blocks[i].take_trace();
  }
  delete[] angm; delete[] bdim; delete[] cdim;
  return tr;
}

template<typename Scalar, typename Sym>
double BlockTensor<Scalar, Sym>::normalize_vector() {
  double nor = 0;
  int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    if (!data_blocks[i].check_null())
      nor += std::real(data_blocks[i].inner_prod_c(data_blocks[i]));
  }
  nor = std::sqrt(std::abs(nor));
  if (nor > 1e-12)
    for (int i = 0; i < nten; ++i)
      if (!data_blocks[i].check_null()) data_blocks[i] /= nor;
  delete[] angm; delete[] bdim; delete[] cdim;
  return nor;
}

// ── Singular value operations ─────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::multiply_singular_value(int leg, double* w) {
  int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];
  int nmom = cgc.get_nmoment(leg);
  double** ww = new double*[nmom];
  int j = 0;
  for (int i = 0; i < nmom; ++i) {
    ww[i] = &w[j];
    j += cgc.get_bonddim(leg, i);
  }
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    if (data_blocks[i].check_null()) continue;
    //find which index gives moment angm[leg]
    int k = cgc.get_angularmoment_index(leg, angm[leg]);
    if (k == -1){
      cout<<"BlockTensor::multiply_singular_value wrong angm index"<<endl;
      exit(1);
    }
    data_blocks[i].multiply_singular_value(leg, ww[k]);
  }
  delete[] ww; delete[] angm; delete[] bdim; delete[] cdim;
}

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::devide_singular_value(int leg, double* w) {
  int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];
  int nmom = cgc.get_nmoment(leg);
  double** ww = new double*[nmom];
  int j = 0;
  for (int i = 0; i < nmom; ++i) {
    ww[i] = &w[j];
    j += cgc.get_bonddim(leg, i);
  }
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    if (data_blocks[i].check_null()) continue;
    int k = cgc.get_angularmoment_index(leg, angm[leg]);
    if (k == -1){
      cout<<"BlockTensor::multiply_singular_value wrong angm index"<<endl;
      exit(1);
    }
    data_blocks[i].devide_singular_value(leg, ww[k]);
  }
  delete[] ww; delete[] angm; delete[] bdim; delete[] cdim;
}

// ── Canonical form transformations ────────────────────────────────────────────

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::left2right_vectran() {
  int ishift = 2;
  StructType cgc1 = cgc;
  cgc1.shift(1, 0);
  Dense<Scalar>* new_blocks = new Dense<Scalar>[nten];
  int* angm  = new int[nbond];
  int* angm1 = new int[nbond];
  int* bdim  = new int[nbond];
  int* cdim  = new int[nbond];
  if constexpr (Sym::has_cgc)
    for (int i = 0; i < nten; ++i) 
      cgc_blocks[i].clean();
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    for (int k = 0; k < nbond; ++k) angm1[(k + ishift) % nbond] = angm[k];
    int j = cgc1.get_tensor_index(angm1);
    if (j < nten && !data_blocks[i].check_null()) {
      new_blocks[j] = data_blocks[i];
      new_blocks[j].shift(1, 0);
      if constexpr (Sym::has_cgc)
	cgc_blocks[j].make_cgc(angm1[0], angm1[1], angm1[2]);
    }
  }
  delete[] data_blocks; data_blocks = new_blocks;
  cgc = cgc1;
  if constexpr (Sym::has_cgc) {
    cgc.set_bonddir(0, 1);
    cgc.set_bonddir(1, 1);
    cgc.set_bonddir(2, -1);
  }
  delete[] angm; delete[] angm1; delete[] bdim; delete[] cdim;
  return *this;
}

template<typename Scalar, typename Sym>
BlockTensor<Scalar, Sym>& BlockTensor<Scalar, Sym>::right2left_vectran() {
  int ishift = 1;
  StructType cgc1 = cgc;
  cgc1.shift(2, 0);
  Dense<Scalar>* new_blocks = new Dense<Scalar>[nten];
  int* angm  = new int[nbond];
  int* angm1 = new int[nbond];
  int* bdim  = new int[nbond];
  int* cdim  = new int[nbond];
  if constexpr (Sym::has_cgc)
    for (int i = 0; i < nten; ++i) 
      cgc_blocks[i].clean();
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim, cdim)) continue;
    for (int k = 0; k < nbond; ++k) angm1[(k + ishift) % nbond] = angm[k];
    int j = cgc1.get_tensor_index(angm1);
    if (j < nten && !data_blocks[i].check_null()) {
      new_blocks[j] = data_blocks[i];
      new_blocks[j].shift(2, 0);
      if constexpr (Sym::has_cgc)
	cgc_blocks[j].make_cgc(angm1[0], angm1[1], angm1[2]);
    }
  }
  delete[] data_blocks; data_blocks = new_blocks;
  cgc = cgc1;
  if constexpr (Sym::has_cgc) {
    cgc.set_bonddir(0, 1);
    cgc.set_bonddir(1, 1);
    cgc.set_bonddir(2, -1);
  }
  delete[] angm; delete[] angm1; delete[] bdim; delete[] cdim;
  return *this;
}

// ── Block-diagonal SVD ────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::svd(BlockTensor& tu, double p1, BlockTensor& tv, double p2, double* wout) {
  int i,j,k,m,n,n0,a1,a2,m1,m2,nmom1,nmom2,nmom3,*dc,*dim,*bdim,*isort,*amom;
  int nm1, dcut;
  Dense<Scalar> tmp, **puarr, **pvarr, *uu, *vv;
  BlockTensor proju, projv, left, rght;
  double **ww, *wsort;
  BondType bb[2], bb2[4], bb3[2];
  tu.get_bond(0, bb2[0]);
  tu.get_bond(1, bb2[1]);
  tv.get_bond(0, bb2[2]);
  tv.get_bond(1, bb2[3]);

  bb[0].fuse(bb2[0], bb2[1]);
  bb[1].fuse(bb2[2], bb2[3]);

  nmom1=bb[0].get_nmoment();
  nmom2=bb[1].get_nmoment();
  uu=new Dense<Scalar>[nmom1];
  vv=new Dense<Scalar>[nmom1];
  dc=new int[nmom1];
  dim=new int[nmom1];
  bdim=new int[nmom1+2];
  amom=new int[nmom1];
  ww=new double*[nmom1];
  nm1=0;
  for(i=0;i<nmom1;i++){
    ww[i]=new double[bb[0].get_bonddim(i)];
    nm1+=bb[0].get_bonddim(i);
  }
  wsort=new double[nm1];
  isort=new int[nm1];
  n=0; //total number of singular values
  n0=0; //number of nonzero block elements
  for(i=0;i<nmom1;i++){
    dc[i] = 0;
    a1 = bb[0].get_angularmoment(i);
    m1 = bb[0].get_bonddim(i);
    for(j=0;j<nmom2;j++){
      a2 = bb[1].get_angularmoment(j);
      m2 = bb[1].get_bonddim(j);
      if (a1 == a2 && !data_blocks[i+j*nmom1].is_null()) {
	bdim[0]=m1;
	bdim[1]=m2;
	tmp.copy(2, bdim, data_blocks[i+j*nmom1].getptr());
	tmp.svd(uu[i], p1, vv[i], p2, ww[i], dc[i], max_dcut, 1);
	//for (int k = 0; k < dc[i]; k++)
	//cout<<i<<"\t"<<ww[i][k]<<endl;
	for(k = 0; k < dc[i]; k++){
	  wsort[n + k] = ww[i][k];
	  isort[n + k] = i;
	}
	n0 += m1 * m2;
	n  += dc[i];
      }
    }
  }
  sort2(n, wsort, isort);  
  for(i = 0; i < nmom1; i++)
    dim[i] = 0;
  dcut = (n > max_dcut) ? max_dcut : n;
  for(i = 0; i < dcut; i++)
    for(j = 0; j < nmom1; j++)
      if (isort[i] == j) {
	dim[j]++;
	break;
      }
  
  nmom3 = 0;
  for(i = 0; i < nmom1; i++)
    if(dim[i]!=0)
      nmom3++;
  puarr = new Dense<Scalar>*[nmom3];
  pvarr = new Dense<Scalar>*[nmom3];

  j = 0; //index for nmom3
  m = 0; //kept index
  m1 = dcut; //discarded index
  for(i = 0; i < nmom1; i++){
    if(dim[i] > 0){
      if (dim[i] < dc[i]) {
	uu[i].direct_subtract(1, dim[i]);
	vv[i].direct_subtract(1, dim[i]);
      }
      puarr[j] = &(uu[i]);
      pvarr[j] = &(vv[i]);
      bdim[j]  = dim[i];
      amom[j]  = bb[0].get_angularmoment(i);
      for(k = 0; k < dim[i]; k++)
	wout[m + k] = ww[i][k];
      for(k = dim[i]; k < dc[i]; k++)
	wout[m1 + k - dim[i]] = ww[i][k];
      m  += dim[i];
      m1 += dc[i] - dim[i];
      j++;
    }
    else{
      for(k = 0; k < dc[i]; k++)
	wout[m1 + k] = ww[i][k];
      m1 += dc[i];
    }
  }
  //set proju block tensor
  bb3[1].set(bb[0].get_bonddir(), nmom3, amom, bdim);
  bb3[0]=bb[0];
  bb3[0].invert_bonddir();
  proju.set(2, 0, bb3, puarr);
  //set projv block tensor
  bb3[0]=bb[1];
  bb3[0].invert_bonddir();
  if constexpr (!Sym::has_cgc)
    bb3[1].invert_bonddir();
  projv.set(2, 0, bb3, pvarr);
  left.fuse(bb2[0], bb2[1]);
  rght.fuse(bb2[2], bb2[3]);
  //left = tu;
  //rght = tv;
  tu.contract(left, 2, proju, 0);
  tv.contract(rght, 2, projv, 0);
  delete []uu;
  delete []vv;
  delete []dc;
  delete []dim;
  delete []bdim;
  delete []amom;
  for(i=0;i<nmom1;i++)
    delete []ww[i];
  delete []ww;
  delete []wsort;
  delete []isort;
  delete []puarr;
  delete []pvarr;
  //print();
  //tu.print();
  //tv.print();
  //exit(0);
}


// ── Flat element I/O ──────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::get_elems(Scalar* data_out, double* cgc_out) const {
  int off_d = 0, off_c = 0;
  int* angm = new int[nbond], *bdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    if (!data_blocks[i].check_null()) {
      if (data_out) data_blocks[i].get_elems(data_out + off_d);
      off_d += data_blocks[i].get_nelem();
    }
    if constexpr (Sym::has_cgc) {
      if (!cgc_blocks[i].check_null()) {
        if (cgc_out) cgc_blocks[i].get_elems(cgc_out + off_c);
        off_c += cgc_blocks[i].get_nelem();
      }
    }
  }
  delete[] angm; delete[] bdim;
}

// ── Printing ─────────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::print() const {
  cout << "BlockTensor nbond=" << nbond << " nten=" << nten << " locspin=" << locspin << endl;
  cgc.print();
  int* angm = new int[nbond], *bdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    if (!data_blocks[i].check_null()) {
      cout << "  block[" << i << "]:";
      for (int j = 0; j < nbond; ++j)
	cout << " q=" << angm[j] << " d=" << bdim[j];
      cout << " nelem=" << data_blocks[i].get_nelem() << endl;
      //data_blocks[i].print();
    }
  }
  cout<<"====================="<<endl;
  delete[] angm; delete[] bdim;
}

// ── Conjugate ─────────────────────────────────────────────────────────────────
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::take_conjugate(int bond_idx) {
  cgc.take_conjugate(bond_idx);
  if constexpr (Sym::has_cgc){
    int* angm = new int[nbond], *bdim = new int[nbond];
    for (int i = 0; i < nten; ++i) {
      if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
      if (!data_blocks[i].check_null()){
	Dense<double> tmp, tmpcgc;
	tmp.make_singlet(angm[bond_idx]);
	tmpcgc.contract(tmp, 0, cgc_blocks[i], bond_idx);
	tmpcgc.shift(0, bond_idx);
	cgc_blocks[i] = tmpcgc;
      }
    }
    delete[] angm; delete[] bdim;
  }
}

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::take_conjugate() { cgc.take_conjugate(); }

// ── Fuse ─────────────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::fuse(const BondType& b1, const BondType& b2) {
  clean();
  BondType bb[3];
  bb[0] = b1; bb[1] = b2;
  bb[2].fuse(b1, b2);
  nbond = 3; locspin = 0;
  cgc.set_struct(3, 0, bb);
  nten = cgc.get_nten();
  data_blocks = new Dense<Scalar>[nten];
  if constexpr (Sym::has_cgc) cgc_blocks = new Dense<double>[nten];

  int* angm = new int[nbond], *bdim = new int[nbond];
  int n3 = bb[2].get_nmoment();
  int* pos = new int[n3]();
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    int k = cgc.get_angularmoment_index(2, angm[2]);
    data_blocks[i].alloc(3, bdim);
    // Identity block: data[a, b, pos[k] + b*bdim[0] + a] = 1
    for (int a = 0; a < bdim[0]; ++a)
      for (int b = 0; b < bdim[1]; ++b) {
        int col = pos[k] + b * bdim[0] + a;
        if (col < bdim[2])
          data_blocks[i].set_elem(a, b, col, Scalar(1));
	else {
	  cout<<"BlockTensor::fuse(bondtype, bondtype) wrong"<<endl;
	  exit(1);
	}
      }
    pos[k] += bdim[0] * bdim[1];
    if constexpr (Sym::has_cgc)
      cgc_blocks[i].make_cgc(angm[0], angm[1], angm[2]);
  }
  delete[] pos; delete[] angm; delete[] bdim;
}

/*
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::fuse_to_multiplet(const BondType& b1, const BondType& b2, int t) {
  clean();
  BondType bb[3];
  bb[0] = b1; bb[1] = b2;
  bb[2].fuse_to_multiplet(b1, b2, t);
  nbond = 3; locspin = 0;
  cgc.set_struct(3, 0, bb);
  nten = cgc.get_nten();
  data_blocks = new Dense<Scalar>[nten];
  if constexpr (Sym::has_cgc) cgc_blocks = new Dense<double>[nten];

  int* angm = new int[nbond], *bdim = new int[nbond];
  int n3 = bb[2].get_nmoment();
  int* pos = new int[n3]();
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    int k = cgc.get_angularmoment_index(2, angm[2]);
    data_blocks[i].alloc(3, bdim);
    // Identity block: data[a, b, pos[k] + b*bdim[0] + a] = 1
    for (int a = 0; a < bdim[0]; ++a)
      for (int b = 0; b < bdim[1]; ++b) {
        int col = pos[k] + b * bdim[0] + a;
        if (col < bdim[2])
          data_blocks[i].set_elem(a, b, col, Scalar(1));
      }
    pos[k] += bdim[0] * bdim[1];
    if constexpr (Sym::has_cgc)
      cgc_blocks[i].make_cgc(angm[0], angm[1], angm[2]);
  }
  delete[] pos; delete[] angm; delete[] bdim;
}
*/

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::fuse_to_singlet(const BondType& b1, const BondType& b2) {
  clean();
  BondType bb[2];
  int bdim[2];
  bb[0] = b1; bb[1] = b2;
  nbond = 2; locspin = 0;
  cgc.set_struct(2, 0, bb);
  nten = cgc.get_nten();
  data_blocks = new Dense<Scalar>[nten];
  if constexpr (Sym::has_cgc) cgc_blocks = new Dense<double>[nten];
  int n1 = bb[0].get_nmoment();
  int n2 = bb[1].get_nmoment();
  for (int j = 0; j < n2; j++) {
    int m2 = bb[1].get_angularmoment(j);
    bdim[1] = bb[1].get_bonddim(j);
    for (int i = 0; i < n1; i++) {
      int m1 = bb[0].get_angularmoment(i);
      bdim[0] = bb[0].get_bonddim(i);
      if constexpr (Sym::has_cgc) {
	if (m1 == m2) {
	  int l = i + j * n1;
	  data_blocks[l].alloc(2, bdim);
	  data_blocks[l] = 1;
	  cgc_blocks[l].make_singlet(m1);
	}
      }
      else {
	if (m1 * b1.get_bonddir() + m2 * b2.get_bonddir() == 0) {
	  int l = i + j * n1;
	  data_blocks[l].alloc(2, bdim);
	  data_blocks[l] = 1;
	}
      }
    }
  }
}

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::fuse(int ind1, int ind2) {
  if(ind2 != ind1 + 1 || cgc.get_bonddir(ind1) != cgc.get_bonddir(ind2) || ind2 >= nbond) {
    cout<<"BlockTensor::fuse(int,int) has wrong parameters"<<endl;
    exit(1);
  }
  BondType *bb;
  StructType cgc1;
  //fused blocktensor has one less bond
  bb = new BondType[nbond - 1];
  cgc.get_bond(ind1, bb[0]);
  cgc.get_bond(ind2, bb[1]);
  BlockTensor<Scalar, Sym> tmp;
  bb[0].invert_bonddir();
  bb[1].invert_bonddir();
  tmp.fuse(bb[0], bb[1]);
  for(int i = 0; i < ind1; i++)
    get_bond(i, bb[i]);
  for(int i = ind2; i < nbond - 1; i++)
    get_bond(i+1, bb[i]);
  //bb[ind1] has fused info of ind1 and ind2
  tmp.get_bond(2, bb[ind1]);
  cgc1.set_struct(nbond - 1, locspin, bb);
  int nten1 = cgc1.get_nten();
  //nmom is for looping into fused number of moments
  int nmom = bb[ind1].get_nmoment();
  int* angm1 = new int[nbond - 1];
  int* angm2 = new int[3];
  int k, l;
  Dense<Scalar> te, ta, tc;
  Dense<Scalar>* new_data_blocks = new Dense<Scalar>[nten1];
  Dense<double>* new_cgc_blocks = nullptr;
  if constexpr (Sym::has_cgc) new_cgc_blocks = new Dense<double>[nten1];

  int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];

  for(int i = 0; i < nten; i++){
    if (get_tensor_argument(i, angm, bdim, cdim) == false) continue;
    //record all moments except ind1 and ind2
    for(int j = 0; j < ind1; j++)
      angm1[j] = angm[j];
    for(int j = ind2; j < nbond - 1; j++)
      angm1[j] = angm[j+1];
    //record moments of ind1 and ind2
    angm2[0] = angm[ind1];
    angm2[1] = angm[ind2];
    for(int j = 0; j < nmom; j++){
      //determine fused moment given ind1 and ind2
      angm1[ind1] = bb[ind1].get_angularmoment(j);
      angm2[2] = angm1[ind1];
      k = cgc1.get_tensor_index(angm1);
      l = tmp.get_tensor_index(angm2);
      if (!tmp.check_angularmoments(angm2) || !cgc1.check_angularmoments(angm1)) continue;
      if (data_blocks[i].is_null() || tmp.data_blocks[l].is_null()) continue;
      te = data_blocks[i];
      te.mergeindex(ind1, ind2);
      ta = tmp.data_blocks[l];
      ta.mergeindex(0, 1);
      tc.contract(ta, 0, te, ind1);
      tc.shift(0, ind1);
      if (new_data_blocks[k].is_null()) {
	new_data_blocks[k] = tc;
	if constexpr (Sym::has_cgc) {
	  if (nbond - 1 == 2)
	    //take advantage of the fact that \sum_{j1,j2} cgc(j1,j2,j3) * cgc(j1,j2,j3') = \delta(j3,j3'); fuse in the su2 case only defines for 3-index tensor -> 2-index tensor
	    new_cgc_blocks[k].make_identity(angm1[ind1]);
	  else {
	    cout<<"BlockTensor::fuse(int,int) has wrong argument"<<endl;
	    exit(1);
	  }
	}
      }
      else
	new_data_blocks[k] += tc;
    }
  }
  nbond = nbond - 1;
  nten  = nten1;
  cgc   = cgc1;
  delete []data_blocks;
  data_blocks = new_data_blocks;
  if constexpr (Sym::has_cgc) {
    delete []cgc_blocks;
    cgc_blocks = new_cgc_blocks;
  }
  delete []angm;
  delete []angm1;
  delete []angm2;
  delete []bdim;
  delete []cdim;
  delete []bb;
}

// ── shift / exchangeindex ─────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::shift(int i0, int i1) {
  if (i0 == i1) return;
  int ishift = (i1 > i0) ? (i1 - i0) : (nbond - (i0 - i1));
  StructType cgc1 = cgc;
  cgc1.shift(i0, i1);
  Dense<Scalar>* nb_blocks = new Dense<Scalar>[nten];
  Dense<double>* nc_blocks = nullptr;
  if constexpr (Sym::has_cgc)
    nc_blocks = new Dense<double>[nten];

  int* angm  = new int[nbond];
  int* angm1 = new int[nbond];
  int* bdim  = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    for (int k = 0; k < nbond; ++k) angm1[(k + ishift) % nbond] = angm[k];
    int j = cgc1.get_tensor_index(angm1);
    if (j >= 0 && j < nten && !data_blocks[i].check_null()) {
      nb_blocks[j] = data_blocks[i];
      nb_blocks[j].shift(i0, i1);
      if constexpr (Sym::has_cgc) {
	//this will be used at contract ovlp matrix, simply transpose back
	nc_blocks[j] = cgc_blocks[i];
	nc_blocks[j].shift(i0, i1);
      }
    }
  }
  delete[] data_blocks; data_blocks = nb_blocks;
  cgc = cgc1;
  if constexpr (Sym::has_cgc) {
    delete[] cgc_blocks;
    cgc_blocks = nc_blocks;
  }
  delete[] angm; delete[] angm1; delete[] bdim;
}

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::exchangeindex(int i0, int i1) {
  if (i0 == i1) return;
  StructType cgc1 = cgc;
  cgc1.exchangeindex(i0, i1);
  Dense<Scalar>* nb_blocks = new Dense<Scalar>[nten];
  Dense<double>* nc_blocks = nullptr;
  if constexpr (Sym::has_cgc)
    nc_blocks = new Dense<double>[nten];
  int* angm  = new int[nbond];
  int* angm1 = new int[nbond];
  int* bdim  = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    std::memcpy(angm1, angm, nbond * sizeof(int));
    std::swap(angm1[i0], angm1[i1]);
    int j = cgc1.get_tensor_index(angm1);
    if (j >= 0 && j < nten && !data_blocks[i].check_null()) {
      nb_blocks[j] = data_blocks[i];
      nb_blocks[j].exchangeindex(i0, i1);
      if constexpr (Sym::has_cgc) {
	//this func seems not used anywhere, simply switch indices
	nc_blocks[j] = cgc_blocks[i];
	nc_blocks[j].exchangeindex(i0, i1);
      }
    }
  }
  cgc = cgc1;
  delete[] data_blocks; data_blocks = nb_blocks;
  if constexpr (Sym::has_cgc) {
    delete[] cgc_blocks;
    cgc_blocks = nc_blocks;
  }
  delete[] angm; delete[] angm1; delete[] bdim;
}

// ── Direct sum ────────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::direct_sum(int ind, const BlockTensor& t1, const BlockTensor& t2) {
  StructType cgc1, cgc2;
  int i,j,k,l,*angm,*bdim,nten1,nten2;  
  bool check;
  clean();
  cgc1 = t1.get_struct();
  cgc2 = t2.get_struct();
  cgc.direct_sum(ind, cgc1, cgc2);
  nbond = cgc.get_nbond();
  nten  = cgc.get_nten();
  locspin = cgc.get_locspin();
  data_blocks = new Dense<Scalar>[nten];
  if constexpr (Sym::has_cgc) 
    cgc_blocks = new Dense<double>[nten];
  angm=new int[nbond];
  bdim=new int[nbond];
  nten1=t1.get_nten();
  nten2=t2.get_nten();
  for(i = 0;i < nten; i++) {
    if (get_tensor_argument(i, angm, bdim) == false) continue;
    j = t1.get_tensor_index(angm);
    k = t2.get_tensor_index(angm);
    if(j == nten1 && k != nten2){
      data_blocks[i] = t2.data_blocks[k];
      if constexpr (Sym::has_cgc) 
	cgc_blocks[i] = t2.cgc_blocks[k];
    }
    else if (j != nten1 && k == nten2){
      data_blocks[i] = t1.data_blocks[j];
      if constexpr (Sym::has_cgc) 
	cgc_blocks[i] = t1.cgc_blocks[j];
    }
    else if(j != nten1 && k != nten2){
      double rat;
      bool check;
      if constexpr (Sym::has_cgc) {
	check = t1.cgc_blocks[j].is_proportional_to(t2.cgc_blocks[k], rat);
	if (!check || std::fabs(rat - 1.) > 1e-12) {
	  cout<<"BlockTensor::direct_sum input tensors wrong"<<endl;
	  exit(1);
	}
      }
      data_blocks[i].direct_sum(ind, t1.data_blocks[j], t2.data_blocks[k]);
      if constexpr (Sym::has_cgc) 
	cgc_blocks[i]  = t1.cgc_blocks[j];
    }
    else if(j == nten1 && k == nten2){
      cout<<"BlockTensor::direct_sum wrong with argument"<<endl;
      exit(1);
    }
  }
  delete []angm;
  delete []bdim;
}

// ── Generic contraction ───────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::contract(const BlockTensor& t1, int i1, const BlockTensor& t2, int i2) {
  BondType *bb, *bb2;
  int i,j,k,l,nbond1,nbond2,nmom;
  int *angm, *bdim, *angm1, *angm2;
  Dense<Scalar> tmp1;
  Dense<double> tmp2;

  bb2 = new BondType[2];
  clean();
  t1.get_bond(i1, bb2[0]);
  t2.get_bond(i2, bb2[1]);
  bb2[1].invert_bonddir();
  if(bb2[0] != bb2[1]){
    cout<<"BlockTensor::contract, two bonds can not be contracted, check bond parameters"<<endl;
    bb2[0].print();
    bb2[1].print();
    exit(1);
  }
  locspin = t1.get_locspin() + t2.get_locspin();
  nbond1 = t1.get_nbond();
  nbond2 = t2.get_nbond();
  nbond = nbond1 + nbond2 - 2;
  bb = new BondType[nbond];
  for(i = 0; i < nbond1 - 1; i++)
    t1.get_bond((i1 + 1 + i) % nbond1, bb[i]);
  for(i = 0; i < nbond2 - 1; i++)
    t2.get_bond((i2 + 1 + i) % nbond2, bb[i + nbond1 - 1]);
  cgc.set_struct(nbond, locspin, bb);
  nten = cgc.get_nten();
  angm=new int[nbond];
  bdim=new int[nbond];
  angm1=new int[nbond1];
  angm2=new int[nbond2];
  data_blocks = new Dense<Scalar>[nten];
  if constexpr (Sym::has_cgc) 
    cgc_blocks = new Dense<double>[nten];

  nmom=bb2[0].get_nmoment();
  for(i = 0; i < nten; i++){
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    for(k = 0; k < nbond1 - 1; k++)
      angm1[(i1 + 1 + k) % nbond1] = angm[k];
    for(k = 0; k < nbond2 - 1; k++)
      angm2[(i2 + 1 + k) % nbond2] = angm[k + nbond1 - 1];
    for(j = 0; j < nmom; j++){
      angm1[i1] = bb2[0].get_angularmoment(j);
      angm2[i2] = angm1[i1];
      if (!t1.check_angularmoments(angm1)) continue;
      if (!t2.check_angularmoments(angm2)) continue;
      k = t1.get_tensor_index(angm1);
      l = t2.get_tensor_index(angm2);
      if (t1.data_blocks[k].is_null() || t2.data_blocks[l].is_null()) continue;
      if (data_blocks[i].is_null()) {
	data_blocks[i].contract(t1.data_blocks[k], i1, t2.data_blocks[l], i2);
	if constexpr (Sym::has_cgc)
	  cgc_blocks[i].contract(t1.cgc_blocks[k], i1,  t2.cgc_blocks[l], i2);
      }
      else{
	if constexpr (Sym::has_cgc) {
	  tmp1.contract(t1.data_blocks[k], i1, t2.data_blocks[l], i2);
	  tmp2.contract(t1.cgc_blocks[k],  i1, t2.cgc_blocks[l],  i2);
	  if(!sum_direct_product(data_blocks[i], cgc_blocks[i], tmp1, tmp2)){
	    cout<<"BlockTensor::contract, su2 tensor contraction: sum_direct_product wrong"<<endl;
	    data_blocks[i].print();
	    cgc_blocks[i].print();
	    tmp1.print();
	    tmp2.print();
	    exit(1);
	  }
	}
      }
    }
  }
  delete []angm;
  delete []angm1;
  delete []angm2;
  delete []bdim;
  delete []bb;
  delete []bb2;
}


// ── Local operator construction ───────────────────────────────────────────────

// make_spinor_start(ls):  Construct the spin-ls/2 spinor operator (m=+ls/2 component).
//
// Convention (from original mps_u1 code):
//   Bond 0: dir=-1, Q=+ls  (physical "bra" sector: spin-up, using 2*m_j units)
//   Bond 1: dir=-1, Q=+2ls (operator channel: the charge carried by this spinor
//                            component; in 2*m_j units the spinor carries +2*ls)
//   Bond 2: dir=+1, Q=+ls  (physical "ket" sector)
//
// totchar = -ls - 2*ls + ls = -2*ls, so cgc locspin is set to -2*ls.
// The single block (ls, 2*ls, ls) has element 1.0.
//
// For spin-1/2 (ls=1): sigma[0] = this spinor × (+sqrt(3)/2)
//                       sigma[1] = this spinor × (-sqrt(3)/2)
// These represent the reduced matrix element <1/2||S||1/2>=sqrt(3)/2 for S+/S-.
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::make_spinor_start(int physpn) {
  BondType *bb;
  int *angm,*bdim;
  Scalar *tele;
  //direction -1 out going, direction 1 in going
  //order: down, horizontal, up
  clean();
  nbond=3;
  locspin=0;
  bb   = new BondType[nbond];
  tele = new Scalar[1];
  if constexpr (Sym::has_cgc) {
    angm = new int[1];
    bdim = new int[nbond];
    angm[0] = physpn;
    bdim[0] = 1;
    bb[0].set(-1, 1, angm, bdim);
    bb[2].set( 1, 1, angm, bdim);
    angm[0] = 2;
    bdim[0] = 1;
    bb[1].set(-1, 1, angm, bdim);
    
    cgc.set_struct(nbond,locspin,bb);
    nten=cgc.get_nten();
    data_blocks=new Dense<Scalar>[1];
    cgc_blocks=new Dense<double>[1];
    for (int i = 0; i < nbond; i++)
      bdim[i]=1;
    tele[0]=1;
    data_blocks[0].copy(nbond, bdim, tele);
    cgc_blocks[0].make_cgc(physpn, 2, physpn);
    delete []angm;
    delete []bdim;
  }
  else {
    angm = new int[physpn + 1 > nbond? physpn + 1: nbond];
    bdim = new int[physpn + 1 > nbond? physpn + 1: nbond];
    for (int i = 0; i < physpn + 1; i++){
      if (i == 0)  angm[i] = 0 - physpn;
      else angm[i] = angm[i - 1] + 2;
      bdim[i] = 1;
    }
    bb[0].set(-1, physpn + 1, angm, bdim);
    bb[2].set( 1, physpn + 1, angm, bdim);
    for (int i = 0; i < 3; i++){
      if (i == 0)  angm[i] = -2;
      else angm[i] = angm[i - 1] + 2;
      bdim[i] = 1;
    }
    bb[1].set(-1, 3, angm, bdim);

    cgc.set_struct(nbond, locspin, bb);
    nten = cgc.get_nten();
    data_blocks = new Dense<Scalar>[nten];
    
    for (int i = 0; i < nten; i++) {
      if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
      if (angm[0] == angm[2] && angm[1] == 0){
	tele[0] = (double)(angm[0]) / 2.; //for spin-1/2 and 1
	data_blocks[i].copy(nbond, bdim, tele);
      }
      else if (angm[0] != angm[2]){
	tele[0] = 1./std::sqrt(2.); //for spin-1/2 and 1
	data_blocks[i].copy(nbond, bdim, tele);
      }
    }
    delete []angm;
    delete []bdim;
  }
  delete []bb;
  delete []tele;
}

// ── Random initialization ──────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::random_initialize_tensor() {
  int* angm = new int[nbond], *bdim = new int[nbond];
  for (int i = 0; i < nten; ++i) {
    if (!cgc.get_tensor_argument(i, angm, bdim)) continue;
    if (!data_blocks[i].check_null()) data_blocks[i].random_init();
  }
  delete[] angm; delete[] bdim;
}

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::makeup_input_vector() {
  int i,j,k,nmom0,nmom1,a0,a1,bdim[2];
  if (nbond != 2) {
    cout<<"BlockTensor::makeup_input_vector called by wrong tensor"<<endl;
    exit(1);
  }
  nmom0 = cgc.get_nmoment(0);
  nmom1 = cgc.get_nmoment(1);
  for(i=0;i<nmom0;i++){
    a0 = cgc.get_angularmoment(0,i);
    bdim[0] = cgc.get_bonddim(0,i);
    for(j=0;j<nmom1;j++){
      a1 = cgc.get_angularmoment(1,j);
      bdim[1] = cgc.get_bonddim(1,j);
      if(a0 == a1 && data_blocks[i+j*nmom0].check_null()){
	data_blocks[i+j*nmom0].alloc(2, bdim);
	if constexpr (Sym::has_cgc) {
	  bdim[0] = a0 + 1;
	  bdim[1] = a1 + 1;
	  cgc_blocks[i+j*nmom0].alloc(2, bdim);
	  cgc_blocks[i+j*nmom0].make_singlet(a0);
	}
      }
    }
  }
}

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::initialize_input_vector() { random_initialize_tensor(); }

// ── CGC helpers ───────────────────────────────────────────────────────────────

template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::make_standard_cgc() {
  if constexpr (Sym::has_cgc) {
    if (nbond != 3) {
      cout<<"BlockTensor::make_standard_cgc called by wrong tensor"<<endl;
      exit(1);
    }
    int* angm = new int[nbond], *bdim = new int[nbond], *cdim = new int[nbond];
    for (int i = 0; i < nten; ++i) {
      if (!get_ta<Sym>(cgc, i, angm, bdim, cdim)) continue;
      int cgc_dims[3] = { cdim[0], cdim[1], cdim[2] };
      cgc_blocks[i].alloc(3, cgc_dims);
      CGCTable::fill_block(angm[0], angm[1], angm[2], cgc_blocks[i].getptr());
    }
    delete[] angm; delete[] bdim; delete[] cdim;
  }
}

//--------------------------------------------------------------------------------------
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::operator_initial(BlockTensor&  uu, BlockTensor&  vv, BlockTensor& op, int flag){
//--------------------------------------------------------------------------------------
  BondType bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,nr,nc,m1,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;//used for SU2 cgc multiplication factor
  Dense<Scalar> tmp,*data_blocks_tmp,step1,tmp1;
  Dense<double> *cgc_blocks_tmp;
  Scalar *aa;
  Scalar alpha=1,beta=0,ctmp;
  int psize = omp_get_max_threads();
  clean();
  //check bond consistancy
  uu.get_bond(1-flag,bb[0]);
  op.get_bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"operator initial wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_bond(1-flag,bb[0]);
  op.get_bond(2,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"operator initial wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  //set block tensor struct
  nbond=3;
  locspin=0;
  uu.get_bond(2,bb[0]);
  op.get_bond(1,bb[1]);
  vv.get_bond(2,bb[2]);
  bb[2].invert_bonddir();
  cgc.set_struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  //alloc reduced dense tensors
  data_blocks = new Dense<Scalar>[nten];
  data_blocks_tmp = new Dense<Scalar>[psize*nten];
  //alloc cgc dense tensors
  if constexpr (Sym::has_cgc) {
    cgc_blocks = new Dense<double>[nten];
    cgc_blocks_tmp=new Dense<double>[psize*nten];
  }
  
  nb=3;
  bc1=new int*[psize];
  bc2=new int*[psize];
  bc3=new int*[psize];
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
  }

  nt1=op.get_nten();
  nt2=uu.get_nten();
  nt3=vv.get_nten();
  nt=nt1*nt2;
  nb=3;

#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,n,p,nr,nc,m1,fac,ctmp,step1,tmp,tmp1,aa) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][1-flag])continue;
    if(op.get_parr(j)->is_null()||uu.get_parr(k)->is_null())continue;
    if(op.get_parr(j)->get_nelem()==1){
      ctmp = op.get_parr(j)->get_elem(0);
      step1=(*uu.get_parr(k));
      step1*=ctmp;
      step1.shift(2,0);
      step1.separateindex(0,bd2[myrk][2],1);
    }
    else{
      step1.contract((*uu.get_parr(k)),1-flag,(*op.get_parr(j)),0);
      if(flag==0)step1.exchangeindex(1,2);
      else if(flag==1)step1.shift(1,0);
    }
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(vv.get_parr(l)->is_null())continue;
      if(angm1[myrk][2]!=angm3[myrk][1-flag])continue;
      if(angm2[myrk][flag]!=angm3[myrk][flag])continue;
      bdim[myrk][0]=angm2[myrk][2];
      bdim[myrk][1]=angm1[myrk][1];
      bdim[myrk][2]=angm3[myrk][2];
      if(cgc.check_angularmoments(bdim[myrk])==false)continue;
      m=cgc.get_tensor_index(bdim[myrk]);
      bdim[myrk][0]=bd2[myrk][2];
      bdim[myrk][1]=bd1[myrk][1];
      bdim[myrk][2]=bd3[myrk][2];
      nr=bdim[myrk][0]*bdim[myrk][1];
      nc=bdim[myrk][2];
      m1=bd3[myrk][0]*bd3[myrk][1];
      tmp1=(*vv.get_parr(l));
      tmp1.conjugate();
      aa=new Scalar[nr*nc];
      ScalarTraits<Scalar>::gemm('N', 'N', nr, nc, m1, alpha, step1.getptr(), nr, tmp1.getptr(), m1, beta, aa, nr);
      //dgemm_("N","N",&nr,&nc,&m1,&alpha,step1.getptr(),&nr,tmp1.getptr(),&m1,&beta,aa,&nr);
      tmp.copy(nb,bdim[myrk],aa);
      delete []aa;
      if constexpr (Sym::has_cgc) {
	if(flag==0)
	  fac=g_cgc_tables->fac_operator_onsite_left[angm2[myrk][2]][angm1[myrk][1]][angm3[myrk][2]][angm2[myrk][flag]][angm1[myrk][0]+angm1[myrk][2]*g_cgc_tables->physpn2];
	else if(flag==1)
	  fac=g_cgc_tables->fac_operator_onsite_rght[angm2[myrk][2]][angm1[myrk][1]][angm3[myrk][2]][angm2[myrk][flag]][angm1[myrk][0]+angm1[myrk][2]*g_cgc_tables->physpn2];

	if(fabs(fac)<1.e-8)
	  continue;
	tmp*=fac;
      } 
      if(data_blocks_tmp[nten*myrk+m].is_null()){
	data_blocks_tmp[nten*myrk+m]=tmp;
	if constexpr (Sym::has_cgc)
	  cgc_blocks_tmp[nten*myrk+m].make_cgc(angm2[myrk][2],angm1[myrk][1],angm3[myrk][2]);
      }
      else
	data_blocks_tmp[nten*myrk+m]+=tmp;
    } 
  }
  for(i=0;i<nten;i++){
    for(j=0;j<psize;j++){
      if(!data_blocks_tmp[nten*j+i].is_null()){
	if(data_blocks[i].is_null()){
	  data_blocks[i]=data_blocks_tmp[nten*j+i];
	  if constexpr (Sym::has_cgc) cgc_blocks[i]=cgc_blocks_tmp[nten*j+i];
	}
	else
	  data_blocks[i]+=data_blocks_tmp[nten*j+i];
      }
    }
  }
  delete []bc1[0];
  delete []bc1;
  delete []bc2[0];
  delete []bc2;
  delete []bc3[0];
  delete []bc3;
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []data_blocks_tmp;
  if constexpr (Sym::has_cgc)
    delete []cgc_blocks_tmp;
}

//--------------------------------------------------------------------------------------
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::operator_transformation(BlockTensor& uu, BlockTensor& vv, BlockTensor& op, int flag){
//--------------------------------------------------------------------------------------
  BondType bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  Dense<Scalar> tmp,*data_blocks_tmp,step1;
  Dense<double> *cgc_blocks_tmp;
  clean();
  //check bond consistancy
  uu.get_bond(flag,bb[0]);
  op.get_bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"operator transformation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_bond(flag,bb[0]);
  op.get_bond(2,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"operator transformation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  //set block tensor struct
  nbond=3;
  locspin=0;
  uu.get_bond(2,bb[0]);
  op.get_bond(1,bb[1]);
  vv.get_bond(2,bb[2]);
  bb[2].invert_bonddir();
  cgc.set_struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  data_blocks=new Dense<Scalar>[nten];
  data_blocks_tmp=new Dense<Scalar>[psize*nten];
  if constexpr (Sym::has_cgc) {
    cgc_blocks=new Dense<double>[nten];
    cgc_blocks_tmp=new Dense<double>[psize*nten];
  }
  nb=3;
  bc1=new int*[psize];
  bc2=new int*[psize];
  bc3=new int*[psize];
  bc4=new int*[psize];
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bd4=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  angm4=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bc4[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bd4[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  angm4[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bc4[i]=&(bc4[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bd4[i]=&(bd4[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
    angm4[i]=&(angm4[0][i*nb]);
  }

  nt1=op.get_nten();
  nt2=uu.get_nten();
  nt3=vv.get_nten();
  nt=nt1*nt2;
  nb=5;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,n,p,fac,step1,tmp) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][flag])continue;
    if(op.get_parr(j)->is_null()||uu.get_parr(k)->is_null())continue;
    step1.contract((*uu.get_parr(k)),flag,(*op.get_parr(j)),0);
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(vv.get_parr(l)->is_null())continue;
      if(angm1[myrk][2]!=angm3[myrk][flag])continue;
      if(angm2[myrk][1-flag]!=angm3[myrk][1-flag])continue;
      bdim[myrk][0]=angm2[myrk][2];
      bdim[myrk][1]=angm1[myrk][1];
      bdim[myrk][2]=angm3[myrk][2];
      if(cgc.check_angularmoments(bdim[myrk])==false)continue;
      m=cgc.get_tensor_index(bdim[myrk]);

      bdim[myrk][0]=bd2[myrk][(flag+1)%3];
      bdim[myrk][1]=bd2[myrk][(flag+2)%3];
      bdim[myrk][2]=bd1[myrk][1];
      bdim[myrk][3]=bd3[myrk][(flag+1)%3];
      bdim[myrk][4]=bd3[myrk][(flag+2)%3];
      Dense<Scalar> tmp1=(*vv.get_parr(l));
      tmp1.conjugate();
      tmp.contract_dmrg_operator_transformation_step5(step1,tmp1,nb,bdim[myrk],flag);
      if(bd2[myrk][1-flag]==1&&bd3[myrk][1-flag]==1){
	tmp.mergeindex(3,4);
	tmp.mergeindex(0,1);
      }
      else{
	if(flag==0)tmp.contractindex(0,3);
	else if(flag==1)tmp.contractindex(1,4);
      }
      if constexpr (Sym::has_cgc) {
	if(flag==0)
	  fac=g_cgc_tables->fac_operator_transformation_left[angm2[myrk][2]][angm1[myrk][1]][angm3[myrk][2]][angm2[myrk][flag]][angm3[myrk][flag]][angm2[myrk][1-flag]];
	else if(flag==1)
	  fac=g_cgc_tables->fac_operator_transformation_rght[angm2[myrk][2]][angm1[myrk][1]][angm3[myrk][2]][angm2[myrk][flag]][angm3[myrk][flag]][angm2[myrk][1-flag]];
	if(fabs(fac)<1.e-8)continue;
	tmp*=fac;
      }
      if(data_blocks_tmp[nten*myrk+m].is_null()){
	data_blocks_tmp[nten*myrk+m]=tmp;
	if constexpr (Sym::has_cgc)
	  cgc_blocks_tmp[nten*myrk+m].make_cgc(angm2[myrk][2],angm1[myrk][1],angm3[myrk][2]);
      }
      else
	data_blocks_tmp[nten*myrk+m]+=tmp;
    }
  }
  for(i=0;i<nten;i++){
    for(j=0;j<psize;j++){
      if(!data_blocks_tmp[nten*j+i].is_null()){
	if(data_blocks[i].is_null()){
	  data_blocks[i]=data_blocks_tmp[nten*j+i];
	  if constexpr (Sym::has_cgc)
	    cgc_blocks[i]=cgc_blocks_tmp[nten*j+i];
	}
	else
	  data_blocks[i]+=data_blocks_tmp[nten*j+i];
      }
    }
  }
  delete []bc1[0];
  delete []bc1;
  delete []bc2[0];
  delete []bc2;
  delete []bc3[0];
  delete []bc3;
  delete []bc4[0];
  delete []bc4;
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bd4[0];
  delete []bd4;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []angm4[0];
  delete []angm4;
  delete []data_blocks_tmp;
  if constexpr (Sym::has_cgc)
    delete []cgc_blocks_tmp;
}


//--------------------------------------------------------------------------------------
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::overlap_initial(BlockTensor& uu, BlockTensor& vv, int flag){
//--------------------------------------------------------------------------------------
  BondType bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  Dense<Scalar> tmp,*data_blocks_tmp,step1;
  Dense<double> *cgc_blocks_tmp;
  clean();
  uu.get_bond(flag,bb[0]);
  vv.get_bond(flag,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"overlap initial wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  uu.get_bond(1-flag,bb[0]);
  vv.get_bond(1-flag,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"overlap initial wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  //set block tensor struct
  nbond=2;
  locspin=0;
  uu.get_bond(2,bb[0]);
  vv.get_bond(2,bb[1]);
  bb[1].invert_bonddir();
  cgc.set_struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  //alloc reduced dense tensors
  data_blocks=new Dense<Scalar>[nten];
  data_blocks_tmp=new Dense<Scalar>[psize*nten];
  //alloc cgc dense tensors
  if constexpr (Sym::has_cgc) {
    cgc_blocks=new Dense<double>[nten];
    cgc_blocks_tmp=new Dense<double>[psize*nten];
  }
  
  nb=3;
  bc1=new int*[psize];
  bc2=new int*[psize];
  bd1=new int*[psize];
  bd2=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
  }

  nt1=uu.get_nten();
  nt2=vv.get_nten();
  nt=nt1;
#pragma omp parallel for default(shared) private(myrk,i,j,k,m,tmp) schedule(dynamic,1)
  for(i=0;i<nt1;i++){
    myrk=omp_get_thread_num();
    j=i;
    if(uu.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_parr(j)->is_null())continue;
    for(k=0;k<nt2;k++){
      if(vv.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
      if(vv.get_parr(k)->is_null())continue;
      if(angm1[myrk][0]!=angm2[myrk][0])continue;
      if(angm1[myrk][1]!=angm2[myrk][1])continue;
      bdim[myrk][0]=angm1[myrk][2];
      bdim[myrk][1]=angm2[myrk][2];
      m=cgc.get_tensor_index(bdim[myrk]);
      Dense<Scalar> tmp1 = (*vv.get_parr(k));
      tmp1.conjugate();
      tmp.contract_dmrg_overlap_initial((*uu.get_parr(j)),tmp1,flag);
      if(data_blocks_tmp[nten*myrk+m].is_null()){
	data_blocks_tmp[nten*myrk+m]=tmp;
	if constexpr (Sym::has_cgc)
	  cgc_blocks_tmp[nten*myrk+m].make_identity(angm1[myrk][2]);
      }
      else
	data_blocks_tmp[nten*myrk+m]+=tmp;
    }
  }
  for(i=0;i<nten;i++){
    for(j=0;j<psize;j++){
      if(!data_blocks_tmp[nten*j+i].is_null()){
	if(data_blocks[i].is_null()){
	  data_blocks[i]=data_blocks_tmp[nten*j+i];
	  if constexpr (Sym::has_cgc)
	    cgc_blocks[i]=cgc_blocks_tmp[nten*j+i];
	}
	else
	  data_blocks[i]+=data_blocks_tmp[nten*j+i];
      }
    }
  }
  delete []bc1[0];
  delete []bc1;
  delete []bc2[0];
  delete []bc2;
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []data_blocks_tmp;
  if constexpr (Sym::has_cgc) 
    delete []cgc_blocks_tmp;
}

//--------------------------------------------------------------------------------------
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::overlap_transformation(BlockTensor& uu, BlockTensor& vv, BlockTensor& op, int flag){
//--------------------------------------------------------------------------------------
  BondType bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  Dense<Scalar> tmp,*data_blocks_tmp,step1;
  Dense<double> *cgc_blocks_tmp;
  clean();
  //check bond consistancy
  uu.get_bond(flag,bb[0]);
  op.get_bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"overlap transformation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_bond(flag,bb[0]);
  op.get_bond(1,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"overlap transformation wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  //set block tensor struct
  nbond=2;
  locspin=0;
  uu.get_bond(2,bb[0]);
  vv.get_bond(2,bb[1]);
  bb[1].invert_bonddir();
  cgc.set_struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  data_blocks=new Dense<Scalar>[nten];
  data_blocks_tmp=new Dense<Scalar>[psize*nten];
  if constexpr (Sym::has_cgc) {
    cgc_blocks=new Dense<double>[nten];
    cgc_blocks_tmp=new Dense<double>[psize*nten];
  }

  nb=3;
  bc1=new int*[psize];
  bc2=new int*[psize];
  bc3=new int*[psize];
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
  }

  nt1=op.get_nten();
  nt2=uu.get_nten();
  nt3=vv.get_nten();
  nt=nt1*nt2;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,step1,tmp) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][flag])continue;
    if(op.get_parr(j)->is_null()||uu.get_parr(k)->is_null())continue;
    if(flag==0)
      step1.contract((*op.get_parr(j)),0,(*uu.get_parr(k)),flag);
    else{
      if(bd2[myrk][1-flag]==1){
	tmp=(*uu.get_parr(k));
	tmp.mergeindex(0,1);
	step1.contract((*op.get_parr(j)),0,tmp,0);
	step1.separateindex(0,1,bd1[myrk][1]);
      }
      else{
	step1.contract((*op.get_parr(j)),0,(*uu.get_parr(k)),flag);
	step1.shift(2,0);
      }
    }
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(vv.get_parr(l)->is_null())continue;
      if(angm1[myrk][1]!=angm3[myrk][flag])continue;
      if(angm2[myrk][1-flag]!=angm3[myrk][1-flag])continue;
      bdim[myrk][0]=angm2[myrk][2];
      bdim[myrk][1]=angm3[myrk][2];
      if(cgc.check_angularmoments(bdim[myrk])==false)continue;
      m=cgc.get_tensor_index(bdim[myrk]);
      Dense<Scalar> tmp1 = (*vv.get_parr(l));
      tmp1.conjugate();
      tmp.contract_dmrg_overlap_initial(step1,tmp1,flag);
      if(data_blocks_tmp[nten*myrk+m].is_null()){
	data_blocks_tmp[nten*myrk+m]=tmp;
	if constexpr (Sym::has_cgc)
	  cgc_blocks_tmp[nten*myrk+m].make_identity(angm2[myrk][2]);
      }
      else
	data_blocks_tmp[nten*myrk+m]+=tmp;
    }
  }
  for(i=0;i<nten;i++){
    for(j=0;j<psize;j++){
      if(!data_blocks_tmp[nten*j+i].is_null()){
	if(data_blocks[i].is_null()){
	  data_blocks[i]=data_blocks_tmp[nten*j+i];
	  if constexpr (Sym::has_cgc)
	    cgc_blocks[i]=cgc_blocks_tmp[nten*j+i];
	}
	else
	  data_blocks[i]+=data_blocks_tmp[nten*j+i];
      }
    }
  }
  delete []bc1[0];
  delete []bc1;
  delete []bc2[0];
  delete []bc2;
  delete []bc3[0];
  delete []bc3;
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []data_blocks_tmp;
  if constexpr (Sym::has_cgc)
    delete []cgc_blocks_tmp;
}

//--------------------------------------------------------------------------------------
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::operator_pairup(BlockTensor& uu, BlockTensor& vv, BlockTensor& op1, BlockTensor& op2, int flag){
//--------------------------------------------------------------------------------------
  BondType bb[3];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  Dense<Scalar> tmp,*data_blocks_tmp,step1,step2,tmp1;
  Dense<double> *cgc_blocks_tmp;
  clean();
  //check bond consistancy
  uu.get_bond(flag,bb[0]);
  op1.get_bond(0,bb[1]);
  bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"operator pairup wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  vv.get_bond(flag,bb[0]);
  op1.get_bond(2,bb[1]);
  if(bb[0]!=bb[1]){
    cout<<"operator pairup wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  op1.get_bond(1,bb[0]);
  op2.get_bond(1,bb[1]);
  if constexpr (!Sym::has_cgc)
    bb[1].invert_bonddir();
  if(bb[0]!=bb[1]){
    cout<<"operator pairup wrong bb"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  //set block tensor struct
  nbond=2;
  locspin=0;
  uu.get_bond(2,bb[0]);
  vv.get_bond(2,bb[1]);
  bb[1].invert_bonddir();
  cgc.set_struct(nbond,locspin,bb);
  nten = cgc.get_nten();
  data_blocks=new Dense<Scalar>[nten];
  data_blocks_tmp=new Dense<Scalar>[psize*nten];
  if constexpr (Sym::has_cgc) {
    cgc_blocks=new Dense<double>[nten];
    cgc_blocks_tmp=new Dense<double>[psize*nten];
  }

  nb=3;
  bc1=new int*[psize];
  bc2=new int*[psize];
  bc3=new int*[psize];
  bc4=new int*[psize];
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bd4=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  angm4=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bc4[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bd4[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  angm4[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bc4[i]=&(bc4[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bd4[i]=&(bd4[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
    angm4[i]=&(angm4[0][i*nb]);
  }

  nt1=op1.get_nten();
  nt2=uu.get_nten();
  nt3=vv.get_nten();
  nt4=op2.get_nten();
  nt=nt1*nt2;
  nb=2;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,n,p,fac,step1,step2,tmp,tmp1) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    //cout<<"i=\t"<<i<<endl;
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op1.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(uu.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][flag])continue;
    if(op1.get_parr(j)->is_null()||uu.get_parr(k)->is_null())continue;
    if(flag==0){
      tmp1=(*uu.get_parr(k));
      tmp1.exchangeindex(1,2);
      step1.contract(tmp1,0,(*op1.get_parr(j)),0);
      //cout<<"check i="<<i<<" step1"<<endl;
    }
    else if(flag==1)
      step1.contract((*uu.get_parr(k)),flag,(*op1.get_parr(j)),0);
    for(l=0;l<nt3;l++){
      if(vv.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(vv.get_parr(l)->is_null())continue;
      if(angm1[myrk][2]!=angm3[myrk][flag])continue;
      tmp1=(*vv.get_parr(l));
      tmp1.conjugate();
      if(flag==1)
	tmp1.exchangeindex(0,1);
      step2.contract(step1,3,tmp1,0);
      step2.mergeindex(2,3);
      step2.mergeindex(1,2);
      step2.exchangeindex(0,1);
      //cout<<"check i="<<i<<" step2"<<endl;
      for(n=0;n<nt4;n++){
	if(op2.get_tensor_argument(n,angm4[myrk],bd4[myrk],bc4[myrk])==false)continue;	
	if(angm4[myrk][0]!=angm2[myrk][1-flag]||angm4[myrk][2]!=angm3[myrk][1-flag]||angm4[myrk][1]!=angm1[myrk][1])continue;
	if(op2.get_parr(n)->is_null())continue;

	bdim[myrk][0]=angm2[myrk][2];
	bdim[myrk][1]=angm3[myrk][2];
	if(cgc.check_angularmoments(bdim[myrk])==false)continue;
	m=cgc.get_tensor_index(bdim[myrk]);
	tmp1=(*op2.get_parr(n));
	tmp1.mergeindex(1,2);
	tmp1.mergeindex(0,1);
	bdim[myrk][0]=bd2[myrk][2];
	bdim[myrk][1]=bd3[myrk][2];
	tmp.contract_dmrg_operator_transformation_step1(step2,tmp1,nb,bdim[myrk],flag);
	//cout<<"check i="<<i<<" step3"<<endl;
	if constexpr (Sym::has_cgc){
	  if(flag==0)
	    fac=g_cgc_tables->fac_operator_pairup_left[angm2[myrk][2]][angm2[myrk][0]][angm3[myrk][0]][angm1[myrk][1]][angm4[myrk][0]+angm4[myrk][2]*g_cgc_tables->physpn2];
	  else if(flag==1)
	    fac=g_cgc_tables->fac_operator_pairup_rght[angm2[myrk][2]][angm2[myrk][1]][angm3[myrk][1]][angm1[myrk][1]][angm4[myrk][0]+angm4[myrk][2]*g_cgc_tables->physpn2];
	  if(fabs(fac)<1.e-8)continue;
	  //cout<<"check i="<<i<<" step4"<<endl;
	  tmp*=fac;
	  //tmp.print();
	  //cout<<"check i="<<i<<" step5 fac="<<fac<<endl;
	}
	if(data_blocks_tmp[nten*myrk+m].is_null()){
	  data_blocks_tmp[nten*myrk+m]=tmp;
	  if constexpr (Sym::has_cgc)
	    cgc_blocks_tmp[nten*myrk+m].make_identity(angm2[myrk][2]);
	}
	else
	  data_blocks_tmp[nten*myrk+m]+=tmp;
	//cout<<"check i="<<i<<" step done"<<endl;
      }
    }
  }
  
  for(i=0;i<nten;i++){
    for(j=0;j<psize;j++){
      if(!data_blocks_tmp[nten*j+i].is_null()){
	if(data_blocks[i].is_null()){
	  data_blocks[i]=data_blocks_tmp[nten*j+i];
	  if constexpr (Sym::has_cgc)
	    cgc_blocks[i]=cgc_blocks_tmp[nten*j+i];
	}
	else
	  data_blocks[i]+=data_blocks_tmp[nten*j+i];
      }
    }
  }
  delete []bc1[0];
  delete []bc1;
  delete []bc2[0];
  delete []bc2;
  delete []bc3[0];
  delete []bc3;
  delete []bc4[0];
  delete []bc4;
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bd4[0];
  delete []bd4;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []angm4[0];
  delete []angm4;
  delete []data_blocks_tmp;
  if constexpr (Sym::has_cgc)
    delete []cgc_blocks_tmp;
}

//--------------------------------------------------------------------------------------
template<typename Scalar, typename Sym>
void BlockTensor<Scalar, Sym>::hamiltonian_vector_multiplication(BlockTensor& vec, BlockTensor& op1, BlockTensor& op2){
//--------------------------------------------------------------------------------------
  BondType bb[6];
  int myrk,i,j,k,l,m,n,p,q,nb,nt,nt1,nt2,nt3,nt4,**angm1,**angm2,**angm3,**angm4,**bc1,**bc2,**bc3,**bc4,**bd1,**bd2,**bd3,**bd4,**bdim;
  double fac;
  Dense<Scalar> tmp,*data_blocks_tmp,step1;
  Dense<double> *cgc_blocks_tmp;
  clean();
  //check bond consistancy
  if(op1.get_nbond()==0||op2.get_nbond()==0)return;
  op1.get_bond(0,bb[0]);
  vec.get_bond(0,bb[1]);
  op2.get_bond(0,bb[2]);
  vec.get_bond(1,bb[3]);
  op1.get_bond(1,bb[4]);
  op2.get_bond(1,bb[5]);
  bb[1].invert_bonddir();
  bb[3].invert_bonddir();
  if constexpr (!Sym::has_cgc)
    bb[5].invert_bonddir();
  if(bb[0]!=bb[1]||bb[2]!=bb[3]||bb[4]!=bb[5]){
    cout<<"hamiltonian_vector_multiplication bb wrong"<<endl;
    bb[0].print();
    bb[1].print();
    exit(0);
  }
  //set block tensor struct
  nbond=2;
  locspin=0;
  op1.get_bond(2,bb[0]);
  op2.get_bond(2,bb[1]);
  cgc.set_struct(nbond,locspin,bb);
  nten=cgc.get_nten();
  data_blocks=new Dense<Scalar>[nten];
  data_blocks_tmp=new Dense<Scalar>[psize*nten];
  if constexpr (Sym::has_cgc) {
    cgc_blocks=new Dense<double>[nten];
    cgc_blocks_tmp=new Dense<double>[psize*nten];
  }

  nb=3;
  bc1=new int*[psize];
  bc2=new int*[psize];
  bc3=new int*[psize];
  bd1=new int*[psize];
  bd2=new int*[psize];
  bd3=new int*[psize];
  bdim=new int*[psize];
  angm1=new int*[psize];
  angm2=new int*[psize];
  angm3=new int*[psize];
  bc1[0]=new int[psize*nb];
  bc2[0]=new int[psize*nb];
  bc3[0]=new int[psize*nb];
  bd1[0]=new int[psize*nb];
  bd2[0]=new int[psize*nb];
  bd3[0]=new int[psize*nb];
  bdim[0]=new int[psize*(nb+3)];
  angm1[0]=new int[psize*nb];
  angm2[0]=new int[psize*nb];
  angm3[0]=new int[psize*nb];
  for(i=1;i<psize;i++){
    bc1[i]=&(bc1[0][i*nb]);
    bc2[i]=&(bc2[0][i*nb]);
    bc3[i]=&(bc3[0][i*nb]);
    bd1[i]=&(bd1[0][i*nb]);
    bd2[i]=&(bd2[0][i*nb]);
    bd3[i]=&(bd3[0][i*nb]);
    bdim[i]=&(bdim[0][i*(nb+3)]);
    angm1[i]=&(angm1[0][i*nb]);
    angm2[i]=&(angm2[0][i*nb]);
    angm3[i]=&(angm3[0][i*nb]);
  }

  nt1=op1.get_nten();
  nt2=vec.get_nten();
  nt3=op2.get_nten();
  nt=nt1*nt2;
#pragma omp parallel for default(shared) private(myrk,i,j,k,l,m,fac,tmp,step1) schedule(dynamic,1)
  for(i=0;i<nt;i++){
    myrk=omp_get_thread_num();
    j=i%nt1;
    k=(i/nt1)%nt2;
    if(op1.get_tensor_argument(j,angm1[myrk],bd1[myrk],bc1[myrk])==false)continue;
    if(vec.get_tensor_argument(k,angm2[myrk],bd2[myrk],bc2[myrk])==false)continue;
    if(angm1[myrk][0]!=angm2[myrk][0])continue;
    if(op1.get_parr(j)->is_null()||vec.get_parr(k)->is_null())continue;
    step1.contract((*vec.get_parr(k)),0,(*op1.get_parr(j)),0);
    for(l=0;l<nt3;l++){
      if(op2.get_tensor_argument(l,angm3[myrk],bd3[myrk],bc3[myrk])==false)continue;
      if(angm3[myrk][0]!=angm2[myrk][1]||angm1[myrk][1]!=angm3[myrk][1])continue;
      bdim[myrk][0]=angm1[myrk][2];
      bdim[myrk][1]=angm3[myrk][2];
      if(cgc.check_angularmoments(bdim[myrk])==false)continue;
      m=cgc.get_tensor_index(bdim[myrk]);
      if(op2.get_parr(l)->is_null())continue;
      tmp.contract_dmrg_overlap_initial(step1,(*op2.get_parr(l)),0);
      if constexpr (Sym::has_cgc){
	fac=g_cgc_tables->fac_hamilt_vec[angm1[myrk][0]][angm1[myrk][1]][angm1[myrk][2]];
	if(fabs(fac)<1.e-12)continue;
	tmp*=fac;
      }
      if(data_blocks_tmp[nten*myrk+m].is_null()){
	data_blocks_tmp[nten*myrk+m]=tmp;
	if constexpr (Sym::has_cgc)
	  cgc_blocks_tmp[nten*myrk+m].make_singlet(angm1[myrk][2]);
      }
      else
	data_blocks_tmp[nten*myrk+m]+=tmp;
    }
  }
  for(i=0;i<nten;i++){
    for(j=0;j<psize;j++){
      if(!data_blocks_tmp[nten*j+i].is_null()){
	if(data_blocks[i].is_null()){
	  data_blocks[i]=data_blocks_tmp[nten*j+i];
	  if constexpr (Sym::has_cgc)
	    cgc_blocks[i]=cgc_blocks_tmp[nten*j+i];
	}
	else
	  data_blocks[i]+=data_blocks_tmp[nten*j+i];
      }
    }
  }
  delete []bc1[0];
  delete []bc1;
  delete []bc2[0];
  delete []bc2;
  delete []bc3[0];
  delete []bc3;
  delete []bd1[0];
  delete []bd1;
  delete []bd2[0];
  delete []bd2;
  delete []bd3[0];
  delete []bd3;
  delete []bdim[0];
  delete []bdim;
  delete []angm1[0];
  delete []angm1;
  delete []angm2[0];
  delete []angm2;
  delete []angm3[0];
  delete []angm3;
  delete []data_blocks_tmp;
  if constexpr (Sym::has_cgc)
    delete []cgc_blocks_tmp;
}

// ── Explicit instantiations ────────────────────────────────────────────────────
template class BlockTensor<double,  U1Symmetry>; 
template class BlockTensor<dcmplex, U1Symmetry>; 
template class BlockTensor<double,  SU2Symmetry>;
template class BlockTensor<dcmplex, SU2Symmetry>;
