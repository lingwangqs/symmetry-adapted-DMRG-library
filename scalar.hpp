/**
 * scalar.hpp — Scalar type traits and BLAS/LAPACK dispatch.
 *
 * Provides a uniform interface for linear algebra routines regardless of
 * whether the wavefunction is real (double) or complex (complex<double>).
 * All inner loops call ScalarTraits<Scalar>::gemm / heev / gesvd / etc.,
 * which resolves at compile time to the appropriate BLAS/LAPACK symbol.
 *
 * Usage
 * -----
 *   ScalarTraits<Scalar>::gemm(...);   // dgemm_ or zgemm_
 *   ScalarTraits<Scalar>::heev(...);   // dsyev_ or zheev_
 *   ScalarTraits<Scalar>::gesvd(...);  // dgesvd_ or zgesvd_
 *
 * Supported instantiations
 * ------------------------
 *   Scalar = double           → real MPS (mps_real)
 *   Scalar = complex<double>  → complex MPS (mps_u1, mps_complex)
 */

#pragma once
#include <complex>
#include <cstring>

using dcmplex = std::complex<double>;

// ── BLAS/LAPACK external declarations ───────────────────────────────────────

extern "C" {
  // Level-3 BLAS: matrix-matrix multiply
  void dgemm_(const char*, const char*, const int*, const int*, const int*,
              const double*, const double*, const int*, const double*, const int*,
              const double*, double*, const int*);
  void zgemm_(const char*, const char*, const int*, const int*, const int*,
              const dcmplex*, const dcmplex*, const int*, const dcmplex*, const int*,
              const dcmplex*, dcmplex*, const int*);

  // Level-1 BLAS: axpy, dot, nrm2
  void   daxpy_(const int*, const double*,  const double*,  const int*, double*,  const int*);
  void   zaxpy_(const int*, const dcmplex*, const dcmplex*, const int*, dcmplex*, const int*);
  double dnrm2_(const int*, const double*,  const int*);
  double dznrm2_(const int*, const dcmplex*, const int*);
  double ddot_ (const int*, const double*,  const int*, const double*,  const int*);
  void   zdotc_(dcmplex*, const int*, const dcmplex*, const int*, const dcmplex*, const int*);
  void   zdotu_(dcmplex*, const int*, const dcmplex*, const int*, const dcmplex*, const int*);
  void   dscal_(const int*, const double*,  double*,  const int*);
  void   zscal_(const int*, const dcmplex*, dcmplex*, const int*);

  // LAPACK: symmetric/hermitian eigensolver
  void dsyev_(const char*, const char*, const int*, double*, const int*,
              double*, double*, const int*, int*);
  void zheev_(const char*, const char*, const int*, dcmplex*, const int*,
              double*, dcmplex*, const int*, double*, const int*, int*);

  // LAPACK: SVD
  void dgesvd_(const char*, const char*, const int*, const int*,
               double*, const int*, double*, double*, const int*,
               double*, const int*, double*, const int*, int*);
  void zgesvd_(const char*, const char*, const int*, const int*,
               dcmplex*, const int*, double*, dcmplex*, const int*,
               dcmplex*, const int*, dcmplex*, const int*, double*, int*);
}

// ── ScalarTraits primary template (unspecialised = compile error) ────────────

template<typename Scalar>
struct ScalarTraits;

// ── Specialisation: double (real) ────────────────────────────────────────────

template<>
struct ScalarTraits<double> {
  using Real = double;

  static double real_part(double x) { return x; }
  static double conj(double x)      { return x; }
  static double norm(double x)      { return x >= 0 ? x : -x; }
  static constexpr bool is_complex = false;

  // y = alpha*x + y
  static void axpy(int n, double alpha, const double* x, double* y) {
    int one = 1;
    daxpy_(&n, &alpha, x, &one, y, &one);
  }

  // ||x||_2
  static double nrm2(int n, const double* x) {
    int one = 1;
    return dnrm2_(&n, x, &one);
  }

  // <x, y> = x^T y  (real dot)
  static double dot(int n, const double* x, const double* y) {
    int one = 1;
    return ddot_(&n, x, &one, y, &one);
  }

  // C = alpha * op(A) * op(B) + beta * C
  static void gemm(char transa, char transb, int m, int n, int k,
                   double alpha, const double* A, int lda,
                   const double* B, int ldb,
                   double beta,  double* C, int ldc) {
    dgemm_(&transa, &transb, &m, &n, &k, &alpha,
           A, &lda, B, &ldb, &beta, C, &ldc);
  }

  // Symmetric eigensolver: A * v = lambda * v  (A overwritten by eigenvectors)
  // evals is length n, work is scratch (size lwork).
  static int heev(char jobz, char uplo, int n, double* A, int lda,
                  double* evals, double* work, int lwork) {
    int info;
    dsyev_(&jobz, &uplo, &n, A, &lda, evals, work, &lwork, &info);
    return info;
  }
  // work size query
  static int heev_lwork(int n) { return 3 * n + 64; }
  // no rwork needed for real
  static int heev_lrwork(int /*n*/) { return 0; }

  // SVD: A = U * S * V^T   (A destroyed)
  static int gesvd(char jobu, char jobvt, int m, int n,
                   double* A, int lda,
                   double* S,
                   double* U, int ldu, double* VT, int ldvt,
                   double* work, int lwork) {
    int info;
    dgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt,
            work, &lwork, &info);
    return info;
  }
  static int gesvd_lwork(int m, int n) {
    return 5 * (m < n ? m : n) + (m > n ? m : n) + 64;
  }
};

// ── Specialisation: complex<double> ─────────────────────────────────────────

template<>
struct ScalarTraits<dcmplex> {
  using Real = double;

  static double  real_part(dcmplex x) { return x.real(); }
  static dcmplex conj(dcmplex x)      { return std::conj(x); }
  static double  norm(dcmplex x)      { return std::abs(x); }
  static constexpr bool is_complex = true;

  static void axpy(int n, dcmplex alpha, const dcmplex* x, dcmplex* y) {
    int one = 1;
    zaxpy_(&n, &alpha, x, &one, y, &one);
  }

  static double nrm2(int n, const dcmplex* x) {
    int one = 1;
    return dznrm2_(&n, x, &one);
  }

  // <x, y> = x^† y  (complex inner product)
  static dcmplex dotc(int n, const dcmplex* x, const dcmplex* y) {
    dcmplex result;
    int one = 1;
    zdotc_(&result, &n, x, &one, y, &one);
    return result;
  }

  // <x, y> = x y  (complex inner product)
  static dcmplex dotu(int n, const dcmplex* x, const dcmplex* y) {
    dcmplex result;
    int one = 1;
    zdotu_(&result, &n, x, &one, y, &one);
    return result;
  }

  static void gemm(char transa, char transb, int m, int n, int k,
                   dcmplex alpha, const dcmplex* A, int lda,
                   const dcmplex* B, int ldb,
                   dcmplex beta, dcmplex* C, int ldc) {
    zgemm_(&transa, &transb, &m, &n, &k, &alpha,
           A, &lda, B, &ldb, &beta, C, &ldc);
  }

  // Hermitian eigensolver; rwork must be length >= max(1, 3n-2)
  static int heev(char jobz, char uplo, int n, dcmplex* A, int lda,
                  double* evals, dcmplex* work, int lwork,
                  double* rwork) {
    int info;
    int lrwork = heev_lrwork(n);
    zheev_(&jobz, &uplo, &n, A, &lda, evals, work, &lwork, rwork, &lrwork, &info);
    return info;
  }
  static int heev_lwork(int n)  { return 2 * n + 64; }
  static int heev_lrwork(int n) { return 3 * n - 2;  }

  // SVD: A = U * S * V^H   (A destroyed)
  static int gesvd(char jobu, char jobvt, int m, int n,
                   dcmplex* A, int lda,
                   double* S,
                   dcmplex* U, int ldu, dcmplex* VT, int ldvt,
                   dcmplex* work, int lwork, double* rwork) {
    int info;
    zgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt,
            work, &lwork, rwork, &info);
    return info;
  }
  static int gesvd_lwork(int m, int n) {
    return 2 * (m < n ? m : n) + (m > n ? m : n) + 64;
  }
};
