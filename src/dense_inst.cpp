/**
 * this file implement the backend of dense tensor manipulations, its head file is in the parent folder dense.hpp
 * src/dense_inst.cpp — Dense<Scalar> template implementation.
 * Ported from mps_u1/tensor.cpp (double) and tensor_dcmplex.cpp (dcmplex).
 *
 * ScalarTraits<Scalar> dispatches all BLAS/LAPACK calls so the same
 * method body compiles correctly for both real and complex.
 */

#include "../dense.hpp"
#include "../symmetry/su2.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <random>

using std::cout;
using std::endl;

// ── Global RNG (thread-safe for single-threaded use) ──────────────────────────
static std::mt19937_64 g_rng(42);
static std::uniform_real_distribution<double> g_dist(-1.0, 1.0);

// ── Index utilities ───────────────────────────────────────────────────────────

// Encode multi-index aa[0..nb-1] → flat index, given dims bdim[0..nb-1].
// Convention: index 0 varies fastest (Fortran / column-major in last sense,
// but matching the original code which uses bonddim[0] as leading dim).
static inline void get_tensor_index(int& i0, int nb, int* bdim, int* aa) {
  i0 = 0;
  for (int i = nb - 1; i >= 0; --i) {
    i0 *= bdim[i];
    i0 += aa[i];
  }
}

// Decode flat index i0 → multi-index aa[0..nb-1]
static inline void get_bond_index(int i0, int nb, int* bdim, int* aa) {
  for (int i = 0; i < nb; ++i) {
    aa[i] = i0 % bdim[i];
    i0   /= bdim[i];
  }
}

// ── Dense<Scalar> ─────────────────────────────────────────────────────────────

template<typename Scalar>
Dense<Scalar>::Dense()
  : nbond(0), nelem(0), bonddim(nullptr), data(nullptr) {}

template<typename Scalar>
Dense<Scalar>::Dense(int nb, const int* dims)
  : nbond(0), nelem(0), bonddim(nullptr), data(nullptr)
{
  alloc(nb, dims);
  random_init();
}

template<typename Scalar>
Dense<Scalar>::Dense(int nb, const int* dims, const Scalar* init)
  : nbond(0), nelem(0), bonddim(nullptr), data(nullptr)
{
  alloc(nb, dims);
  std::memcpy(data, init, nelem * sizeof(Scalar));
}

template<typename Scalar>
Dense<Scalar>::Dense(const Dense& o)
  : nbond(0), nelem(0), bonddim(nullptr), data(nullptr)
{
  copy(o);
}

template<typename Scalar>
Dense<Scalar>::~Dense() { clean(); }

template<typename Scalar>
void Dense<Scalar>::clean() {
  delete[] data;    data    = nullptr;
  delete[] bonddim; bonddim = nullptr;
  nbond = 0; nelem = 0;
}

template<typename Scalar>
void Dense<Scalar>::alloc(int nb, const int* dims) {
  clean();
  nbond = nb;
  bonddim = new int[nb];
  nelem = 1;
  for (int i = 0; i < nb; ++i) { bonddim[i] = dims[i]; nelem *= dims[i]; }
  data = new Scalar[nelem];
  std::fill(data, data + nelem, Scalar(0));
}

template<typename Scalar>
void Dense<Scalar>::copy(int nb, const int* dims, const Scalar* src) {
  alloc(nb, dims);
  std::memcpy(data, src, nelem * sizeof(Scalar));
}

template<typename Scalar>
void Dense<Scalar>::copy(const Dense& o) {
  if (&o == this) return;
  copy(o.nbond, o.bonddim, o.data);
}

template<typename Scalar>
void Dense<Scalar>::random_init() {
  for (int i = 0; i < nelem; ++i) {
    if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
      data[i] = Scalar(g_dist(g_rng), g_dist(g_rng));
    } else {
      data[i] = g_dist(g_rng);
    }
  }
}

template<typename Scalar>
void Dense<Scalar>::zero() { std::fill(data, data + nelem, Scalar(0)); }

template<typename Scalar>
void Dense<Scalar>::print() const {
  int *idx =  new int[nbond]();
  cout << "Dense<Scalar> nbond=" << nbond << " nelem=" << nelem << endl;
  for (int i = 0; i < nbond; ++i) cout << "  bonddim[" << i << "]=" << bonddim[i] << endl;
  /*
  for (int i = 0; i < nelem; i++)
    if (std::norm(data[i]) > 1e-12){
      get_bond_index(i, nbond, bonddim, idx);
      cout<<"elem[";
      for (int j = 0; j < nbond; j++)
	cout<<idx[j]<<",";
      cout<<"]="<<data[i]<<endl;
    }
  */
  delete []idx;
}

template<typename Scalar>
void Dense<Scalar>::set_elem(int i, int j, int k, Scalar val) {
  data[i + j * bonddim[0] + k * bonddim[0] * bonddim[1]] = val;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::conjugate() {
  if constexpr (std::is_same_v<Scalar, std::complex<double>> ||
		std::is_same_v<Scalar, std::complex<float>>) {
    for (int i = 0; i < nelem ; i++)
      data[i] = std::conj(data[i]);
  }
  return *this;
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::operator=(Scalar s) {
  std::fill(data, data + nelem, s); return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::operator=(const Dense& o) {
  if (this == &o) return *this;
  copy(o); return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::operator+=(const Dense& o) {
  for (int i = 0; i < nelem; ++i) data[i] += o.data[i];
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::operator-=(const Dense& o) {
  for (int i = 0; i < nelem; ++i) data[i] -= o.data[i];
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::operator*=(double s) {
  for (int i = 0; i < nelem; ++i) data[i] *= s; return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::operator*=(dcmplex s) {
  Scalar ss;
  if constexpr (std::is_same_v<Scalar, double>) ss = s.real();
  else ss = s;
  for (int i = 0; i < nelem; ++i) data[i] *= ss;
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::operator/=(double s) {
  for (int i = 0; i < nelem; ++i) data[i] /= s; return *this;
}

template<typename Scalar>
Dense<Scalar> Dense<Scalar>::operator+(const Dense& o) const {
  Dense res(*this); res += o; return res;
}

template<typename Scalar>
Dense<Scalar> Dense<Scalar>::operator-(const Dense& o) const {
  Dense res(*this); res -= o; return res;
}

template<typename Scalar>
bool Dense<Scalar>::operator==(const Dense& o) const { return !(*this != o); }

template<typename Scalar>
bool Dense<Scalar>::operator!=(const Dense& o) const {
  if (nbond != o.nbond || nelem != o.nelem) return true;
  const double tol = 1e-12;
  for (int i = 0; i < nelem; ++i) {
    double a = ScalarTraits<Scalar>::norm(data[i]);
    double b = ScalarTraits<Scalar>::norm(o.data[i]);
    if ((a + b) > tol && ScalarTraits<Scalar>::norm(data[i] - o.data[i]) / (a + b) > tol)
      return true;
  }
  return false;
}

template<typename Scalar>
double Dense<Scalar>::get_norm() const {
  double n = 0;
  for (int i = 0; i < nelem; ++i) {
    double a = ScalarTraits<Scalar>::norm(data[i]);
    n += a * a;
  }
  return n;
}

template<typename Scalar>
double Dense<Scalar>::rescale() {
  double nor = std::sqrt(get_norm());
  if (nor > 1e-24) for (int i = 0; i < nelem; ++i) data[i] /= nor;
  return nor;
}

template<typename Scalar>
void Dense<Scalar>::rescale(double s) {
  if (std::abs(s) > 1e-24) for (int i = 0; i < nelem; ++i) data[i] /= s;
}

// ── Index manipulation ─────────────────────────────────────────────────────────

template<typename Scalar>
void Dense<Scalar>::mergeindex(int ind1, int ind2) {
  if (ind2 != ind1 + 1) { cout << "Dense::mergeindex: only adjacent indices!" << endl; exit(1); }
  bonddim[ind1] *= bonddim[ind2];
  --nbond;
  for (int i = ind2; i < nbond; ++i) bonddim[i] = bonddim[i + 1];
}

template<typename Scalar>
void Dense<Scalar>::separateindex(int ind, int d0, int d1) {
  if (d0 * d1 != bonddim[ind]) { cout << "Dense::separateindex: dims inconsistent" << endl; exit(1); }
  int* bdim = new int[nbond + 1];
  for (int i = 0; i < ind;   ++i) bdim[i]     = bonddim[i];
  bdim[ind]   = d0;
  bdim[ind+1] = d1;
  for (int i = ind + 1; i < nbond; ++i) bdim[i + 1] = bonddim[i];
  ++nbond;
  delete[] bonddim;
  bonddim = bdim;
}

template<typename Scalar>
void Dense<Scalar>::shift(int i0, int i1) {
  if (i0 == i1) return;

  const int ishift = (i1 > i0) ? (i1 - i0) : (nbond - (i0 - i1));
  if (ishift == 0) return;

  int* bdim = new int[nbond];
  for (int i = 0; i < nbond; ++i) {
    bdim[(i + ishift) % nbond] = bonddim[i];
  }

  // Same shortcut logic you already had
  const bool trivial =
      (bonddim[0] == 1 && ishift == nbond - 1) ||
      (bonddim[nbond - 1] == 1 && ishift == 1);

  if (trivial) {
    delete[] bonddim;
    bonddim = bdim;
    return;
  }

  const int p = nbond - ishift;

  int B = 1;  // product of old dims [0..p-1]
  int A = 1;  // product of old dims [p..nbond-1]
  for (int i = 0; i < p; ++i) B *= bonddim[i];
  for (int i = p; i < nbond; ++i) A *= bonddim[i];

  Scalar* tele = new Scalar[nelem];

  // Column-major transpose:
  // new linear index  = a + A*b
  // old linear index  = b + B*a
  for (int b = 0; b < B; ++b) {
    for (int a = 0; a < A; ++a) {
      tele[a + A * b] = data[b + B * a];
    }
  }

  delete[] data;
  data = tele;

  delete[] bonddim;
  bonddim = bdim;
}

template<typename Scalar>
void Dense<Scalar>::exchangeindex(int ind1, int ind2) {
  if (ind1 == ind2) return;

  if (std::abs(ind1 - ind2) == 1 &&
      (bonddim[ind1] == 1 || bonddim[ind2] == 1)) {
    std::swap(bonddim[ind1], bonddim[ind2]);
    return;
  }

  if (ind1 > ind2) std::swap(ind1, ind2);

  int* newdim = new int[nbond];
  std::memcpy(newdim, bonddim, nbond * sizeof(int));
  std::swap(newdim[ind1], newdim[ind2]);

  Scalar* tele = new Scalar[nelem];

  int A = 1, C = 1, E = 1;
  const int B = bonddim[ind1];
  const int D = bonddim[ind2];

  for (int i = 0; i < ind1; ++i) A *= bonddim[i];
  for (int i = ind1 + 1; i < ind2; ++i) C *= bonddim[i];
  for (int i = ind2 + 1; i < nbond; ++i) E *= bonddim[i];

  for (int e = 0; e < E; ++e) {
    for (int d = 0; d < D; ++d) {
      for (int c = 0; c < C; ++c) {
        for (int b = 0; b < B; ++b) {
          const int src_block = A * (b + B * (c + C * (d + D * e)));
          const int dst_block = A * (d + D * (c + C * (b + B * e)));

          std::memcpy(tele + dst_block, data + src_block, A * sizeof(Scalar));
        }
      }
    }
  }

  delete[] data;
  data = tele;

  delete[] bonddim;
  bonddim = newdim;
}

// ── Contraction ────────────────────────────────────────────────────────────────

// *this = A contracted on A.bond[ia] with B.bond[ib]
// Output bond ordering: remaining bonds of A (starting from ia+1) then remaining of B (from ib+1)
template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract(const Dense& A, int ia, const Dense& B, int ib) {
  if (A.bonddim[ia] != B.bonddim[ib]) {
    cout << "Dense::contract: bond dims mismatch " << A.bonddim[ia] << "!=" << B.bonddim[ib] << endl;
    exit(1);
  }
  Dense tp1, tp2;
  tp1.copy(A); tp2.copy(B);

  char transa, transb;
  int lda, ldb;
  if (ia == 0) { transa = 'T'; lda = A.bonddim[ia]; }
  else if (ia == A.nbond - 1) { transa = 'N'; lda = A.nelem / A.bonddim[ia]; }
  else { tp1.shift(ia, A.nbond - 1); transa = 'N'; lda = A.nelem / A.bonddim[ia]; }

  if (ib == 0) { transb = 'N'; ldb = B.bonddim[ib]; }
  else if (ib == B.nbond - 1) { transb = 'T'; ldb = B.nelem / B.bonddim[ib]; }
  else { tp2.shift(ib, 0); transb = 'N'; ldb = B.bonddim[ib]; }

  int m = A.nelem / A.bonddim[ia];
  int n = B.nelem / B.bonddim[ib];
  int k = A.bonddim[ia];

  Scalar* tele = new Scalar[m * n];
  Scalar alpha(1), beta(0);
  ScalarTraits<Scalar>::gemm(transa, transb, m, n, k, alpha, tp1.data, lda, tp2.data, ldb, beta, tele, m);

  int nb1 = A.nbond + B.nbond - 2;
  int* bdim = new int[nb1];
  int jj = A.nbond - 1;
  for (int i = 0; i < jj; ++i) bdim[i] = A.bonddim[(ia + 1 + i) % A.nbond];
  int kk = B.nbond - 1;
  for (int i = 0; i < kk; ++i) bdim[jj + i] = B.bonddim[(ib + 1 + i) % B.nbond];

  clean();
  data = tele;
  bonddim = bdim;
  nbond = nb1; nelem = m * n;
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contractindex(int i0, int i1) {
  if (bonddim[i0] != bonddim[i1]) {
    cout << "Dense::contractindex: dims differ" << endl; exit(1);
  }
  if (i0 > i1) std::swap(i0, i1);
  int m  = bonddim[i0];
  int ni1 = 1; for (int i = 0; i < i0; ++i) ni1 *= bonddim[i];
  int ni2 = ni1; for (int i = i0 + 1; i < i1; ++i) ni2 *= bonddim[i];
  int nj1 = ni1 * m;
  int nj2 = ni2 * m;
  int nj3 = ni2 * m * m;
  int nele = nelem / (m * m);
  Scalar* tele = new Scalar[nele];
  for (int i = 0; i < nele; ++i) {
    int s0 = i % ni1;
    int s1 = (i % ni2) / ni1;
    int s2 = i / ni2;
    Scalar tmp = 0;
    for (int j = 0; j < m; ++j)
      tmp += data[s0 + j * ni1 + s1 * nj1 + j * nj2 + s2 * nj3];
    tele[i] = tmp;
  }
  for (int i = i0; i < i1 - 1; ++i) bonddim[i] = bonddim[i + 1];
  for (int i = i1 - 1; i < nbond - 2; ++i) bonddim[i] = bonddim[i + 2];
  nbond -= 2;
  delete[] data; data = tele;
  nelem = nele;
  return *this;
}

// ── SVD ───────────────────────────────────────────────────────────────────────

// Split *this into U * S * V.
// Bonds 0..n0-1 go into U, bonds n0..nbond-1 go into V.
// p1=1 → U gets S (V is pure right singular vectors)
// p2=1 → V gets S (U is pure left singular vectors)
template<typename Scalar>
void Dense<Scalar>::svd(Dense& U, double p1, Dense& V, double p2,
			double* svals, int& kept, int max_kept, int n0) {
  int m = 1, n = 1;
  // n0 is given; caller sets p1/p2 to determine which side absorbs S
  // We determine cut_bond: n0 = number of bonds in left factor
  // Convention from original: p1 is left weight, p2 is right weight, both in [0,1].
  // We infer n0 from the product of first half of bonds.
  // To match original: we expose the cut_bond explicitly via the dims.
  // For simplicity, use LAPACK gesvd directly.

  // Treat data as an m×n matrix (m = product of first k bonds, n = rest).
  // Determine k: the cut is passed implicitly — we search for the natural cut
  // that makes m <= n (or vice versa depending on p1/p2).
  // Actually the original code uses p1=1 or p2=1 as a flag.
  // We detect which side we're on by looking at p1, p2.
  // If p1 > 0.5: left side absorbs (p1-1) power of S; right absorbs p2 power.
  // For DMRG: p1=0,p2=1 → U is isometry (S absorbed into V)
  //            p1=1,p2=0 → V is isometry (S absorbed into U)

  // Determine split: if p1 > p2, left side is heavier → split to keep left minimal.

  for (int i = 0;  i < n0; ++i) m *= bonddim[i];
  for (int i = n0; i < nbond; ++i) n *= bonddim[i];

  // LAPACK gesvd on an m×n matrix.
  int mn = std::min(m, n);
  Scalar* A    = new Scalar[m * n];
  std::memcpy(A, data, m * n * sizeof(Scalar));
  double* S    = new double[mn];
  Scalar* Umat = new Scalar[m * mn];
  Scalar* VT   = new Scalar[mn * n];

  // Work size query
  int lwork_q = ScalarTraits<Scalar>::gesvd_lwork(m, n);
  Scalar* work = new Scalar[lwork_q];

  int info;
  if constexpr (std::is_same_v<Scalar, double>) {
    info = ScalarTraits<Scalar>::gesvd('S', 'S', m, n, A, m, S, Umat, m, VT, mn, work, lwork_q);
  } else {
    double* rwork = new double[5 * mn];
    info = ScalarTraits<Scalar>::gesvd('S', 'S', m, n, A, m, S, Umat, m, VT, mn, work, lwork_q, rwork);
    delete[] rwork;
  }
  delete[] work;
  delete[] A;
  if (info != 0) { cout << "Dense::svd: gesvd info=" << info << endl; exit(1); }

  // Determine how many singular values to keep
  double smax = (S[0] > 1e-16 ? S[0] : 1.0);
  kept = 0;
  for (int i = 0; i < mn; ++i)
    if (S[i] / smax > 1e-10) ++kept;
  if (kept == 0) kept = 1;
  if (max_kept > 0 && kept > max_kept) kept = max_kept;

  // Copy singular values out
  for (int i = 0; i < kept; ++i) svals[i] = S[i];

  // Build U: m × kept matrix
  {
    int nb_u = n0 + 1;
    int* d_u = new int[nb_u];
    for (int i = 0; i < n0; ++i) d_u[i] = bonddim[i];
    d_u[n0] = kept;
    U.alloc(nb_u, d_u);
    delete[] d_u;
    // Umat is m×mn column major; we want first 'kept' columns
    for (int j = 0; j < kept; ++j) {
      double sv_pow = (p1 != 0.0) ? std::pow(S[j], p1) : 1.0;
      for (int i = 0; i < m; ++i)
	U.data[i + j * m] = Umat[i + j * m] * sv_pow;
    }
  }

  // Build V: kept × n matrix (transposed from VT)
  {
    int nb_v = 1 + (nbond - n0);
    int* d_v = new int[nb_v];
    d_v[nb_v - 1] = kept;
    for (int i = n0; i < nbond; ++i) d_v[i - n0] = bonddim[i];
    V.alloc(nb_v, d_v);
    delete[] d_v;
    for (int i = 0; i < kept; ++i) {
      double sv_pow = (p2 != 0.0) ? std::pow(S[i], p2) : 1.0;
      for (int j = 0; j < n; ++j)
        V.data[j + i * n] = VT[i + j * mn] * sv_pow;
    }
  }

  delete[] S;
  delete[] Umat;
  delete[] VT;
}

template<typename Scalar>
void Dense<Scalar>::multiply_singular_value(int leg, const double* svals) {
    const int dim = bonddim[leg];

    int inner = 1;
    for (int k = leg + 1; k < nbond; ++k) inner *= bonddim[k];

    const int outer = nelem / (dim * inner);

    for (int o = 0; o < outer; ++o) {
        const int base_o = o * dim * inner;
        for (int j = 0; j < dim; ++j) {
            const Scalar scale = static_cast<Scalar>(svals[j]);
            Scalar* p = data + base_o + j * inner;
            for (int t = 0; t < inner; ++t) {
                p[t] *= scale;
            }
        }
    }
}

template<typename Scalar>
void Dense<Scalar>::devide_singular_value(int leg, const double* svals) {
    const int dim = bonddim[leg];

    int inner = 1;
    for (int k = leg + 1; k < nbond; ++k) inner *= bonddim[k];

    const int outer = nelem / (dim * inner);

    for (int o = 0; o < outer; ++o) {
        const int base_o = o * dim * inner;
        for (int j = 0; j < dim; ++j) {
            const Scalar scale = static_cast<Scalar>(svals[j]);
	    if (std::norm(scale) < 1e-16) continue;
            Scalar* p = data + base_o + j * inner;
            for (int t = 0; t < inner; ++t) {
                p[t] /= scale;
            }
        }
    }
}

// ── Eigenvalue ────────────────────────────────────────────────────────────────

template<typename Scalar>
void Dense<Scalar>::hermitian_eig(double* evals) {
  // Assumes *this is a square (n×n) Hermitian matrix stored flat.
  int n = bonddim[0];  // must be 2-bond with equal dims
  int lwork = ScalarTraits<Scalar>::heev_lwork(n);
  if constexpr (std::is_same_v<Scalar, double>) {
    Scalar* work = new Scalar[lwork];
    int info = ScalarTraits<Scalar>::heev('V', 'U', n, data, n, evals, work, lwork);
    delete[] work;
    if (info != 0) { cout << "Dense::hermitian_eig: dsyev info=" << info << endl; exit(1); }
  } else {
    Scalar* work = new Scalar[lwork];
    double* rwork = new double[ScalarTraits<Scalar>::heev_lrwork(n)];
    int info = ScalarTraits<Scalar>::heev('V', 'U', n, data, n, evals, work, lwork, rwork);
    delete[] work; delete[] rwork;
    if (info != 0) { cout << "Dense::hermitian_eig: zheev info=" << info << endl; exit(1); }
  }
}

template<typename Scalar>
void Dense<Scalar>::hermitian_eig_inv() {
  int n = bonddim[0];
  double* evals = new double[n];
  hermitian_eig(evals);  // data → eigenvectors
  // Invert eigenvalues
  Scalar* inv = new Scalar[n * n];
  std::fill(inv, inv + n * n, Scalar(0));
  for (int i = 0; i < n; ++i)
    if (std::abs(evals[i]) > 1e-16)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
          inv[j + k * n] += data[j + i * n] * ScalarTraits<Scalar>::conj(data[k + i * n]) / evals[i];
  std::memcpy(data, inv, n * n * sizeof(Scalar));
  delete[] inv; delete[] evals;
}

// ── Inner products ────────────────────────────────────────────────────────────

template<typename Scalar>
Scalar Dense<Scalar>::inner_prod_u(const Dense& o) const {
  if (nelem != o.nelem) { cout << "Dense::inner_prod_u: size mismatch (" << nelem << " vs " << o.nelem << ")" << endl; exit(1); }
  if constexpr (ScalarTraits<Scalar>::is_complex)
    return ScalarTraits<Scalar>::dotu(nelem, data, o.data);
  else
    return ScalarTraits<Scalar>::dot(nelem, data, o.data);
}

template<typename Scalar>
Scalar Dense<Scalar>::inner_prod_c(const Dense& o) const {
  if (nelem != o.nelem) { cout << "Dense::inner_prod_c: size mismatch (" << nelem << " vs " << o.nelem << ")" << endl; exit(1); }
  if constexpr (ScalarTraits<Scalar>::is_complex)
    return ScalarTraits<Scalar>::dotc(nelem, data, o.data);
  else
    return ScalarTraits<Scalar>::dot(nelem, data, o.data);
}

template<typename Scalar>
Scalar Dense<Scalar>::take_trace() const {
  if (nbond != 2 || bonddim[0] != bonddim[1]) {
    cout << "Dense::take_trace: requires square 2-bond tensor" << endl; exit(1);
  }
  int m = bonddim[0];
  Scalar tr = 0;
  for (int i = 0; i < m; ++i) tr += data[i + i * m];
  return tr;
}

template<typename Scalar>
void Dense<Scalar>::calculate_difference(const Dense& o, double& abs_diff, double& rel_diff) const {
  abs_diff = 0;
  rel_diff = 0;
  for (int i = 0; i < nelem; ++i) {
    double d = ScalarTraits<Scalar>::norm(data[i] - o.data[i]);
    double a = ScalarTraits<Scalar>::norm(data[i]);
    abs_diff += d * d;
    rel_diff += a * a;
  }
}

// ── DMRG environment contractions ─────────────────────────────────────────────
// These match contract_dmrg_overlap_initial etc. in the original tensor.cpp.


// ── CGC / operator helpers ────────────────────────────────────────────────────

template<typename Scalar>
void Dense<Scalar>::make_cgc(int j1, int j2, int j3) {
  // fill using CGCTable::fill_block (defined in su2.cpp)
  int dims[3] = { j1 + 1, j2 + 1, j3 + 1 };
  alloc(3, dims);
  CGCTable cgcfunc;
  if constexpr (std::is_same_v<Scalar, double>)
    cgcfunc.fill_block(j1,j2,j3,data);
  else
    cout<<"No implementation for complex cgc"<<endl;
}

template<typename Scalar>
void Dense<Scalar>::make_singlet(int jj) {
  // Singlet operator: identity in J-space, delta in m-space
  int dims[2] = { jj + 1, jj + 1 }, sgn;
  alloc(2, dims);
  for(int i=0;i<jj+1;i++){
    int m1=2*i-jj;
    for(int j=0;j<jj+1;j++){
      int m2=2*j-jj;
      if(m1+m2==0){
	if(((jj-m1)/2)%2==0)sgn=1;
	else sgn=-1;
	data[i+j*(jj+1)]=sgn/sqrt((double)(jj+1));
      }
    }
  }
}

template<typename Scalar>
void Dense<Scalar>::make_identity(int jj) {
  int dims[2] = { jj + 1, jj + 1 };
  alloc(2, dims);
  for (int i = 0; i < jj + 1; ++i) data[i + i * (jj + 1)] = Scalar(1);
}

template<typename Scalar>
void Dense<Scalar>::multiply_cgc(int j1, int j2, int j3, int which_bond, int dir) {
  mergeindex(which_bond, which_bond + 1);
  Dense tmp; tmp.make_cgc(j1, j2, j3);
  tmp.mergeindex(0, 1);
  Dense res; res.contract(tmp, 0, *this, which_bond);
  res.shift(0, which_bond);
  *this = res;
}

template<typename Scalar>
void Dense<Scalar>::shift_set_identity(int dim, int nshift, const int* dims) {
  // Fill a block at offset nshift along bond 0 with identity
  // Used in block_tensor construction
  for (int i = 0; i < dim; ++i)
    data[nshift + i + (nshift + i) * dims[0]] = Scalar(1);
}

template<typename Scalar>
void Dense<Scalar>::shift_copy(int i0, int i1, int i2, const Dense& src) {
  int d0 = src.bonddim[0], d1 = src.bonddim[1], d2 = src.bonddim[2];
  for (int a = 0; a < d0; ++a)
    for (int b = 0; b < d1; ++b)
      for (int c = 0; c < d2; ++c) {
        int src_idx = a + b * d0 + c * d0 * d1;
        int dst_idx = (i0+a) + (i1+b)*bonddim[0] + (i2+c)*bonddim[0]*bonddim[1];
        data[dst_idx] = src.data[src_idx];
      }
}

template<typename Scalar>
void Dense<Scalar>::shift_copy(int i0, int i1, int i2, int j0, int j1, int j2, const Dense& src) {
  int d0 = bonddim[0] - i0, d1 = bonddim[1] - i1;
  int s0 = src.bonddim[0];
  for (int a = 0; a < d0; ++a)
    for (int b = 0; b < d1; ++b) {
      int dst_idx = (i0 + a) + (i1 + b) * bonddim[0];
      int src_idx = (j0 + a) + (j1 + b) * s0;
      data[dst_idx] = src.data[src_idx];
    }
}

// ── Predicates ────────────────────────────────────────────────────────────────

template<typename Scalar>
bool Dense<Scalar>::check_null()  const { return (nbond == 0 || nelem == 0); }
template<typename Scalar>
bool Dense<Scalar>::is_null()     const { return check_null(); }

template<typename Scalar>
bool Dense<Scalar>::is_zero() const {
  for (int i = 0; i < nelem; ++i)
    if (ScalarTraits<Scalar>::norm(data[i]) > 1e-12) return false;
  return true;
}

template<typename Scalar>
bool Dense<Scalar>::is_identity() const {
  if (nbond != 2 || bonddim[0] != bonddim[1]) return false;
  int n = bonddim[0];
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      Scalar expected = (i == j) ? Scalar(1) : Scalar(0);
      if (ScalarTraits<Scalar>::norm(data[i + j * n] - expected) > 1e-10) return false;
    }
  return true;
}

template<typename Scalar>
bool Dense<Scalar>::is_minus_identity() const {
  if (nbond != 2 || bonddim[0] != bonddim[1]) return false;
  int n = bonddim[0];
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      Scalar expected = (i == j) ? Scalar(-1) : Scalar(0);
      if (ScalarTraits<Scalar>::norm(data[i + j * n] - expected) > 1e-10) return false;
    }
  return true;
}

template<typename Scalar>
bool Dense<Scalar>::is_one() const {
  return (nelem == 1 && ScalarTraits<Scalar>::norm(data[0] - Scalar(1)) < 1e-10);
}

template<typename Scalar>
bool Dense<Scalar>::is_minus_one() const {
  return (nelem == 1 && ScalarTraits<Scalar>::norm(data[0] - Scalar(-1)) < 1e-10);
}

template<typename Scalar>
bool Dense<Scalar>::is_proportional_to(const Dense& o, Scalar& ratio){
  if (nelem != o.nelem) return false;
  if (nbond != o.get_nbond())return false;
  int pos=0;
  double max=0, tol=1e-12;
  for(int i=0;i<nbond;i++)
    if(bonddim[i]!=o.get_bonddim(i))return false;
  if(is_zero()){
    ratio=0;
    return true;
  }
  for(int i=0;i<nelem;i++){
    if(std::abs(o.get_elem(i))>max){
      pos=i;
      max=std::abs(o.get_elem(i));
    }
  }
  if(max>tol)
    ratio=data[pos]/o.get_elem(pos);
  else{
    cout<<"is_proportional_to comparing to zero tensor"<<endl;
    exit(0);
  }
  
  for(int i=0;i<nelem;i++)
    if(std::abs(data[i])<=tol&&std::abs(o.get_elem(i))<=tol)
      continue;
    else if(std::abs(data[i])<=tol&&std::abs(o.get_elem(i))>tol)
      return false;
    else if(std::abs(data[i])>tol&&std::abs(o.get_elem(i))<=tol)
      return false;
    else if(std::abs(data[i]/o.get_elem(i)-ratio)>tol)
      return false;
  
  /*
  for (int i = 0; i < nelem; ++i)
    if (ScalarTraits<Scalar>::norm(data[i] - ratio * o.data[i]) > 1e-10 *
	(ScalarTraits<Scalar>::norm(data[i]) + ScalarTraits<Scalar>::norm(o.data[i]) + 1e-6))
      return false;
  */
  for(int i=0;i<nelem;i++)
    data[i]=o.get_elem(i);
  return true;
}

// ── Direct sum / product ───────────────────────────────────────────────────────

template<typename Scalar>
void Dense<Scalar>::direct_sum(int ind, const Dense& t1, const Dense& t2) {
  if (t1.nbond != t2.nbond) { cout << "Dense::direct_sum: nbond mismatch" << endl; exit(1); }
  for (int i = 0; i < t1.nbond; ++i)
    if (i != ind && t1.bonddim[i] != t2.bonddim[i]) {
      cout << "Dense::direct_sum: bonddim mismatch at bond " << i << endl; exit(1);
    }
  int* bdim = new int[t1.nbond];
  for (int i = 0; i < t1.nbond; ++i) bdim[i] = t1.bonddim[i];
  bdim[ind] += t2.bonddim[ind];
  alloc(t1.nbond, bdim);
  delete[] bdim;

  // Copy t1 into first part, t2 into second part along ind
  // For simplicity (and matching original behavior), use element-by-element copy
  int* aa = new int[nbond];
  // Copy t1
  for (int k = 0; k < t1.nelem; ++k) {
    get_bond_index(k, t1.nbond, t1.bonddim, aa);
    int j; get_tensor_index(j, nbond, bonddim, aa);
    data[j] = t1.data[k];
  }
  // Copy t2 with offset along ind
  int* aa2 = new int[nbond];
  for (int k = 0; k < t2.nelem; ++k) {
    get_bond_index(k, t2.nbond, t2.bonddim, aa);
    std::memcpy(aa2, aa, nbond * sizeof(int));
    aa2[ind] += t1.bonddim[ind];
    int j; get_tensor_index(j, nbond, bonddim, aa2);
    data[j] = t2.data[k];
  }
  delete[] aa; delete[] aa2;
}

template<typename Scalar>
void Dense<Scalar>::direct_subtract(int ind, int dim_kept) {
  //trim the bonddim of specified index ind to dim_kept
  shift(ind, nbond - 1);
  int nelem1 = (nelem / bonddim[nbond-1]) * dim_kept;
  Scalar* data_tmp = new Scalar[nelem1];
  std::memcpy(data_tmp, data, nelem1 * sizeof(Scalar));
  bonddim[nbond - 1] = dim_kept;
  nelem = nelem1;
  delete []data; data = data_tmp;
  shift(nbond - 1, ind);
}

template<typename Scalar>
void Dense<Scalar>::direct_product(const Dense& t1, const Dense& t2) {
  int nb = t1.nbond + t2.nbond;
  int* bdim = new int[nb];
  for (int i = 0; i < t1.nbond; ++i) bdim[i]          = t1.bonddim[i];
  for (int i = 0; i < t2.nbond; ++i) bdim[t1.nbond+i] = t2.bonddim[i];
  alloc(nb, bdim);
  delete[] bdim;
  int* aa  = new int[nb];
  int* aa1 = new int[t1.nbond];
  int* aa2 = new int[t2.nbond];
  for (int k = 0; k < nelem; ++k) {
    get_bond_index(k, nb, bonddim, aa);
    int j1, j2;
    get_tensor_index(j1, t1.nbond, t1.bonddim, aa);
    get_tensor_index(j2, t2.nbond, t2.bonddim, aa + t1.nbond);
    data[k] = t1.data[j1] * t2.data[j2];
  }
  delete[] aa; delete[] aa1; delete[] aa2;
}

template<typename Scalar>
void Dense<Scalar>::tensor_product(const Dense& t1, const Dense& t2) {
  direct_product(t1, t2);
}

// ── CGC tensor contractions (ported from tensor_dmrg_src.cpp) ─────────────────
// Direct ports of the five DMRG contraction helpers, plus the overlap helper.
// Translation key:
//   tensor::nelement   → Dense::nelem
//   tensor::telement   → Dense::data
//   tensor::bonddim[i] → Dense::bonddim[i]  (both private; accessible inside member)
//   dgemm_(...)        → ScalarTraits<Scalar>::gemm(...)

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract_dmrg_overlap_initial(Dense& t1, Dense& t2, int /*flag*/) {
  if (t1.nbond != 3 || t2.nbond != 3) {
    cout << "Dense::contract_dmrg_overlap_initial: nbond must be 3" << endl;
    t1.print(); t2.print(); exit(0);
  }
  if (t1.bonddim[0] != t2.bonddim[0] || t1.bonddim[1] != t2.bonddim[1]) {
    cout << "Dense::contract_dmrg_overlap_initial: bond dimensions not consistent" << endl;
    t1.print(); t2.print(); exit(0);
  }
  int k  = t1.bonddim[0] * t1.bonddim[1];
  int m  = t1.nelem / k;
  int n  = t2.nelem / k;
  int ne = m * n;
  Scalar* tele = new Scalar[ne];
  ScalarTraits<Scalar>::gemm('T', 'N', m, n, k,
    Scalar(1), t1.data, k, t2.data, k, Scalar(0), tele, m);
  int* bdim = new int[2];
  bdim[0] = t1.bonddim[2];
  bdim[1] = t2.bonddim[2];
  delete[] data;    data    = tele;
  delete[] bonddim; bonddim = bdim;
  nbond = 2; nelem = ne;
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract_dmrg_hamiltonian_vector_multiplication(Dense& t1, Dense& t2, int /*flag*/) {
  if (t1.nbond != 3 || t2.nbond != 3) {
    cout << "Dense::contract_dmrg_hamiltonian_vector_multiplication: nbond must be 3" << endl;
    t1.print(); t2.print(); exit(0);
  }
  if (t1.bonddim[0] != t2.bonddim[0] || t1.bonddim[1] != t2.bonddim[1]) {
    cout << "Dense::contract_dmrg_hamiltonian_vector_multiplication: bond dims not consistent" << endl;
    t1.print(); t2.print(); exit(0);
  }
  int k  = t1.bonddim[0] * t1.bonddim[1];
  int m  = t1.nelem / k;
  int n  = t2.nelem / k;
  int ne = m * n;
  Scalar* tele = new Scalar[ne];
  ScalarTraits<Scalar>::gemm('T', 'N', m, n, k,
    Scalar(1), t1.data, k, t2.data, k, Scalar(0), tele, m);
  int* bdim = new int[2];
  bdim[0] = t1.bonddim[2];
  bdim[1] = t2.bonddim[2];
  delete[] data;    data    = tele;
  delete[] bonddim; bonddim = bdim;
  nbond = 2; nelem = ne;
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract_dmrg_operator_initial(Dense& t1, Dense& t2, Dense& t3, int flag) {
  if (flag == 0 && (t1.bonddim[1] != t3.bonddim[0] || t2.bonddim[1] != t3.bonddim[2])) {
    cout << "Dense::contract_dmrg_operator_initial bonddim not consistent" << endl; exit(0);
  }
  if (flag == 1 && (t1.bonddim[0] != t3.bonddim[0] || t2.bonddim[0] != t3.bonddim[2])) {
    cout << "Dense::contract_dmrg_operator_initial bonddim not consistent" << endl; exit(0);
  }
  Dense tmp1, tmp2;
  if (t3.bonddim[0] == 1 && t3.bonddim[1] == 1 && t3.bonddim[2] == 1) {
    contract_dmrg_overlap_initial(t1, t2, flag);
    (*this) *= t3.data[0];
    separateindex(0, bonddim[0], 1);
  } else if (flag == 0) {
    tmp1.contract(t1, 1, t3, 0);
    tmp1.exchangeindex(1, 2);
    tmp1.mergeindex(2, 3);
    tmp2 = t2; tmp2.mergeindex(0, 1);
    contract(tmp1, 2, tmp2, 0);
  } else {
    tmp1.contract(t1, 0, t3, 0);
    tmp1.shift(3, 0);
    tmp1.mergeindex(0, 1);
    tmp2 = t2; tmp2.mergeindex(0, 1);
    contract(tmp1, 0, tmp2, 0);
  }
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract_dmrg_operator_transformation(Dense& t1, Dense& t2, Dense& t3, int flag) {
  if (flag == 0 && (t1.bonddim[0] != t3.bonddim[0] || t2.bonddim[0] != t3.bonddim[2])) {
    cout << "Dense::contract_dmrg_operator_transformation bonddim not consistent" << endl; exit(0);
  }
  if (flag == 1 && (t1.bonddim[1] != t3.bonddim[0] || t2.bonddim[1] != t3.bonddim[2])) {
    cout << "Dense::contract_dmrg_operator_transformation bonddim not consistent" << endl; exit(0);
  }
  Dense tmp1, tmp2;
  // Fast path: t3 has no physical index (bonddim[1]==1) and t1 is rank-1 along the contracted direction
  if (t3.bonddim[1] == 1 && ((flag == 0 && t1.bonddim[1] == 1) || (flag == 1 && t1.bonddim[0] == 1))) {
    int k1 = t3.bonddim[0];
    int m1 = t1.nelem / k1;
    int n1 = t3.nelem / k1;
    int ne1 = m1 * n1;
    Scalar* tele1 = new Scalar[ne1];
    ScalarTraits<Scalar>::gemm('T', 'N', m1, n1, k1,
      Scalar(1), t1.data, k1, t3.data, k1, Scalar(0), tele1, m1);
    int k2 = t3.bonddim[2];
    int m2 = ne1 / k2;
    int n2 = t2.nelem / k2;
    int ne = m2 * n2;
    Scalar* tele = new Scalar[ne];
    ScalarTraits<Scalar>::gemm('N', 'N', m2, n2, k2,
      Scalar(1), tele1, m2, t2.data, k2, Scalar(0), tele, m2);
    int* bdim = new int[3];
    bdim[0] = t1.bonddim[2];
    bdim[1] = t3.bonddim[1];  // = 1
    bdim[2] = t2.bonddim[2];
    delete[] data;    data    = tele;
    delete[] bonddim; bonddim = bdim;
    nbond = 3; nelem = ne;
    delete[] tele1;
  } else if (flag == 0) {
    tmp1.contract(t1, 0, t3, 0);
    tmp1.shift(1, 0);
    tmp1.mergeindex(2, 3);
    tmp2 = t2; tmp2.mergeindex(0, 1);
    contract(tmp1, 2, tmp2, 0);
  } else {
    tmp1.contract(t1, 1, t3, 0);
    tmp1.exchangeindex(1, 2);
    tmp1.mergeindex(2, 3);
    tmp2 = t2; tmp2.mergeindex(0, 1);
    contract(tmp1, 2, tmp2, 0);
  }
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract_dmrg_operator_pairup(Dense& uu, Dense& vv, Dense& op1, Dense& op2, int flag) {
  if (flag == 0 && (uu.bonddim[0] != op1.bonddim[0] || vv.bonddim[0] != op1.bonddim[2])) {
    cout << "Dense::contract_dmrg_operator_pairup bonddim not consistent" << endl; exit(0);
  }
  if (flag == 1 && (uu.bonddim[1] != op1.bonddim[0] || vv.bonddim[1] != op1.bonddim[2])) {
    cout << "Dense::contract_dmrg_operator_pairup bonddim not consistent" << endl; exit(0);
  }
  Dense tmp1, tmp2, tmp3;
  if (op1.bonddim[1] == 1 && op2.bonddim[0] == 1 && op2.bonddim[1] == 1 && op2.bonddim[2] == 1) {
    contract_dmrg_operator_transformation(uu, vv, op1, flag);
    mergeindex(0, 1);
    (*this) *= op2.data[0];
  } else if (flag == 0) {
    tmp1.contract(uu, 0, op1, 0);
    tmp1.exchangeindex(1, 2);
    tmp1.mergeindex(0, 1);
    tmp2 = op2; tmp2.mergeindex(0, 1);
    tmp3.contract(tmp1, 0, tmp2, 0);
    tmp3.mergeindex(1, 2);
    tmp2 = vv; tmp2.mergeindex(0, 1);
    contract(tmp3, 1, tmp2, 0);
  } else {
    tmp1.contract(uu, 1, op1, 0);
    tmp1.shift(1, 0);
    tmp1.mergeindex(0, 1);
    tmp2 = op2; tmp2.mergeindex(0, 1);
    tmp3.contract(tmp2, 0, tmp1, 0);
    tmp3.mergeindex(0, 1);
    tmp2 = vv; tmp2.mergeindex(0, 1);
    contract(tmp3, 0, tmp2, 0);
  }
  return *this;
}

template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract_dmrg_permutation(Dense& uu, Dense& vv, Dense& vec,
                                                          Dense& op1, Dense& op2, int flag) {
  Dense tmp1, tmp2, tmp3, tmp4;
  int bd1 = op1.bonddim[0], bd2 = op1.bonddim[1], bd3 = op1.bonddim[2];
  int bd4 = op2.bonddim[0], bd5 = op2.bonddim[1], bd6 = op2.bonddim[2];
  bool scalar_case = (bd1==1&&bd2==1&&bd3==1&&bd4==1&&bd5==1&&bd6==1);
  bool flag0_1d    = (flag==0&&bd1==1&&bd2==1&&bd3==2&&bd4==1&&bd5==1&&bd6==2);
  bool flag1_1d    = (flag==1&&bd1==2&&bd2==1&&bd3==1&&bd4==2&&bd5==1&&bd6==1);
  if (scalar_case || flag0_1d || flag1_1d) {
    tmp1 = uu; tmp1.mergeindex(0, 1);
    tmp2 = vec; tmp2.mergeindex(0, 1);
    tmp3 = vv; tmp3.mergeindex(0, 1);
    tmp4.contract(tmp1, 0, tmp2, 0);
    contract(tmp4, 1, tmp3, 0);
    separateindex(0, bonddim[0], 1);
    Scalar fac1 = op1.inner_prod_u(op2);
    (*this) *= fac1;
  } else if (flag == 0) {
    tmp1 = vv; tmp1.exchangeindex(0, 1);
    tmp2.contract(op1, 0, tmp1, 0);
    tmp2.exchangeindex(1, 2);
    tmp2.mergeindex(0, 1);
    tmp1 = vec; tmp1.mergeindex(1, 2);
    tmp3.contract(tmp1, 1, tmp2, 0);
    tmp3.mergeindex(0, 1);
    tmp1.contract(uu, 1, op2, 0);
    tmp1.exchangeindex(1, 2);
    tmp1.mergeindex(2, 3);
    contract(tmp1, 2, tmp3, 0);
  } else {
    tmp1.contract(uu, 0, op2, 1);
    tmp1.exchangeindex(1, 2);
    tmp1.mergeindex(0, 1);
    tmp2 = vec; tmp2.mergeindex(0, 1);
    tmp3.contract(tmp1, 0, tmp2, 0);
    tmp3.mergeindex(1, 2);
    tmp1 = op1; tmp1.exchangeindex(0, 1);
    tmp2.contract(tmp1, 0, vv, 0);
    tmp2.exchangeindex(1, 2);
    tmp2.mergeindex(0, 1);
    contract(tmp3, 1, tmp2, 0);
  }
  return *this;
}

//--------------------------------------------------------------------------------------
template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract_dmrg_operator_transformation_step5(const Dense& t1, const Dense& t2, int nb, const int* bdim, int flag){
//--------------------------------------------------------------------------------------
  int i,m1,n1,n2,nr,nc;
  Scalar alpha=1,beta=0;
  Dense<Scalar> tmp;
  clean();
  if(flag==1){
    tmp=t2;
    tmp.shift(1,0);
  }
  n1=t1.get_nelem();
  n2=t2.get_nelem();
  m1=t1.get_bonddim(3);
  //set tensor parameters
  nbond = nb;
  bonddim =  new int[nbond];
  std::memcpy(bonddim, bdim, nbond * sizeof(int));
  nelem=1;
  for(i = 0; i < nbond; i++) nelem *= bonddim[i];
  nr=n1/m1;
  nc=n2/m1;
  if(nr*nc!=nelem){
    cout<<"wrong bonddim for operator transformation2 step5 "<<endl;
    t1.print();
    t2.print();
    for(i=0;i<nb;i++)
      cout<<i<<"\t"<<bdim[i]<<endl;
    exit(0);
  }
  data=new Scalar[nr*nc];
  if(flag==0)
    ScalarTraits<Scalar>::gemm('N', 'N', nr, nc, m1,
			       Scalar(1), t1.data, nr, t2.data, m1, Scalar(0), data, nr);
  //dgemm_("N","N",&nr,&nc,&m1,&alpha,t1.getptr(),&nr,t2.getptr(),&m1,&beta,data,&nr);
  else
    ScalarTraits<Scalar>::gemm('N', 'N', nr, nc, m1,
			       Scalar(1), t1.data, nr, tmp.data, m1, Scalar(0), data, nr);
  //dgemm_("N","N",&nr,&nc,&m1,&alpha,t1.getptr(),&nr,tmp.getptr(),&m1,&beta,data,&nr);
  return *this;
}

//--------------------------------------------------------------------------------------
template<typename Scalar>
Dense<Scalar>& Dense<Scalar>::contract_dmrg_operator_transformation_step1(const Dense& t1, const Dense& t2, int nb, const int* bdim, int flag){
//--------------------------------------------------------------------------------------
  int i,m1,n1,n2,nr,nc;
  Scalar alpha=1,beta=0;
  clean();
  n1=t1.get_nelem();
  n2=t2.get_nelem();
  m1=t1.get_bonddim(0);
  //set tensor parameters
  nbond = nb;
  bonddim =  new int[nbond];
  std::memcpy(bonddim, bdim, nbond * sizeof(int));
  nelem=1;
  for(i=0;i<nbond;i++)nelem*=bonddim[i];
  nr=n1/m1;
  nc=n2/m1;
  if(nr*nc!=nelem){
    cout<<"wrong bonddim for operator transformation2 step1 2 nele="<<nelem<<"\tnr"<<nr<<"\tnc="<<nc<<endl;
    t1.print();
    t2.print();
    exit(0);
  }
  data=new Scalar[nr*nc];
  ScalarTraits<Scalar>::gemm('T', 'N', nr, nc, m1,
			     Scalar(1), t1.data, m1, t2.data, m1, Scalar(0), data, nr);
  //dgemm_("T","N",&nr,&nc,&m1,&alpha,t1.getptr(),&m1,t2.getptr(),&m1,&beta,aa,&nr);
  return *this;
}

//--------------------------------------------------------------------------------------
template<typename Scalar>
bool sum_direct_product(Dense<Scalar>& ta1, Dense<double>& tb1, Dense<Scalar>& ta2, Dense<double>& tb2){
//--------------------------------------------------------------------------------------
  double nor1;
  Scalar nor2;
  Dense<Scalar> tmp1;
  if(tb1.is_zero()){
    ta1=ta2;
    tb1=tb2;
  }
  else if(tb2.is_zero())
    return true;
  else if(tb1==tb2)
    ta1+=ta2;
  else if(ta1==ta2)
    tb1+=tb2;
  else if(tb2.is_proportional_to(tb1,nor1)){
    tmp1=ta2;
    tmp1*=nor1;
    ta1+=tmp1;
  }
  else if(ta1.is_proportional_to(ta2,nor2)){
    cout<<"this case in sum_direct_product will not occure"<<endl;
    exit(0);
  }
  else{
    cout<<"sum_direct_product(1,2,3,4) can not perform"<<endl;
    ta1.print();
    tb1.print();
    ta2.print();
    tb2.print();
    exit(0);
    return false;
  }
  return true;
}

//--------------------------------------------------------------------------------------
template<typename Scalar>
void sum_direct_product(Dense<Scalar>& ta1, Dense<double>& tb1, Dense<Scalar>& ta2, Dense<double>& tb2, Dense<Scalar>& ta3, Dense<double>& tb3){
//--------------------------------------------------------------------------------------
  bool pass1,pass2;
  pass1=sum_direct_product(ta1,tb1,ta2,tb2);  
  pass2=sum_direct_product(ta1,tb1,ta3,tb3);  
  if(!pass1||!pass2){
    cout<<"sum_direct_product can not sum two direct product tensors"<<endl;
    ta1.print();
    ta2.print();
    ta3.print();
    tb1.print();
    tb2.print();
    tb3.print();
    exit(0);
  }
}

// ── Explicit instantiations ────────────────────────────────────────────────────
template class Dense<double>;
template class Dense<dcmplex>;

template bool sum_direct_product<double> (Dense<double>&,  Dense<double>&, Dense<double>&,  Dense<double>&);
template bool sum_direct_product<dcmplex>(Dense<dcmplex>&, Dense<double>&, Dense<dcmplex>&, Dense<double>&);
template void sum_direct_product<double> (Dense<double>&,  Dense<double>&, Dense<double>&,  Dense<double>&, Dense<double>&,  Dense<double>&);
template void sum_direct_product<dcmplex>(Dense<dcmplex>&, Dense<double>&, Dense<dcmplex>&, Dense<double>&, Dense<dcmplex>&, Dense<double>&);
