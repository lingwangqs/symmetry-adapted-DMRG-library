/**
 * src/symmetry/su2.cpp — SU2Bond, SU2Struct, and CGCTable implementations.
 * Ported from mps_u1/su2bond.cpp, su2struct.cpp, and tensor.cpp::make_cgc.
 */

#include "../../symmetry/su2.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

using std::cout;
using std::endl;
using std::abs;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void sort_by_j(int n, int* jvals, int* dims) {
  for (int i = 0; i < n - 1; ++i)
    for (int j = i + 1; j < n; ++j)
      if (jvals[j] < jvals[i]) {
        std::swap(jvals[i], jvals[j]);
        std::swap(dims[i],  dims[j]);
      }
}

// ── CGCTable ──────────────────────────────────────────────────────────────────

static double factorial(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= i;
  return f;
}

// Compute <j1 m1; j2 m2 | j3 m3>  (all arguments = 2*J, 2*M convention)
// Based on Racah formula as implemented in tensor.cpp::make_cgc.
double CGCTable::cg(int j1, int m1, int j2, int m2, int j3, int m3) {
  if (m1 + m2 != m3) return 0.0;
  // triangle rule
  if (j3 < abs(j1 - j2) || j3 > j1 + j2) return 0.0;
  if ((j1 + j2 + j3) % 2 != 0) return 0.0;

  double fac1 = std::sqrt((double)(j3 + 1) *
    factorial((j3 + j1 - j2) / 2) *
    factorial((j3 - j1 + j2) / 2) *
    factorial((j1 + j2 - j3) / 2) /
    factorial((j1 + j2 + j3) / 2 + 1));

  double f1 = factorial((j1 + m1) / 2) * factorial((j1 - m1) / 2);
  double f2 = factorial((j2 + m2) / 2) * factorial((j2 - m2) / 2);
  double f3 = factorial((j3 + m3) / 2) * factorial((j3 - m3) / 2);
  double fac2 = std::sqrt(f1 * f2 * f3);

  int k1 = (j1 + j2 - j3) / 2;
  int k2 = (j1 - m1) / 2;
  int k3 = (j2 + m2) / 2;
  int k4 = (j3 - j2 + m1) / 2;
  int k5 = (j3 - j1 - m2) / 2;
  int kmax = (j1 + j2 + j3) / 2;

  double sumk = 0.0;
  for (int k0 = 0; k0 <= kmax; ++k0) {
    if (k1 - k0 >= 0 && k2 - k0 >= 0 && k3 - k0 >= 0 &&
        k4 + k0 >= 0 && k5 + k0 >= 0) {
      double sgn = (k0 % 2 == 0) ? 1.0 : -1.0;
      sumk += sgn / (factorial(k0) * factorial(k1 - k0) * factorial(k2 - k0) *
                     factorial(k3 - k0) * factorial(k4 + k0) * factorial(k5 + k0));
    }
  }
  return fac1 * fac2 * sumk;
}

// Fill buf[(j1+1)*(j2+1)*(j3+1)] with CGCs for all m combinations.
// Layout: buf[i + j*(j1+1) + k*(j1+1)*(j2+1)] = <j1,m1=2i-j1; j2,m2=2j-j2 | j3,m3=2k-j3>
void CGCTable::fill_block(int j1, int j2, int j3, double* buf) {
  int n = (j1 + 1) * (j2 + 1) * (j3 + 1);
  std::fill(buf, buf + n, 0.0);
  for (int i = 0; i <= j1; ++i) {
    int m1 = 2 * i - j1;
    for (int j = 0; j <= j2; ++j) {
      int m2 = 2 * j - j2;
      for (int k = 0; k <= j3; ++k) {
        int m3 = 2 * k - j3;
        int idx = i + j * (j1 + 1) + k * (j1 + 1) * (j2 + 1);
        buf[idx] = cg(j1, m1, j2, m2, j3, m3);
      }
    }
  }
}

// ── SU2Bond ───────────────────────────────────────────────────────────────────

SU2Bond::SU2Bond()
  : bonddir(0), nmoment(0), angularmoment(nullptr), bonddim(nullptr), cgcdim(nullptr) {}

SU2Bond::~SU2Bond() { clean(); }

SU2Bond::SU2Bond(int d, int n, const int* jvals, const int* dims)
  : bonddir(0), nmoment(0), angularmoment(nullptr), bonddim(nullptr), cgcdim(nullptr)
{
  set(d, n, jvals, dims);
}

void SU2Bond::clean() {
  delete[] angularmoment; angularmoment = nullptr;
  delete[] bonddim;       bonddim       = nullptr;
  delete[] cgcdim;        cgcdim        = nullptr;
  nmoment = 0;
  bonddir = 0;
}

void SU2Bond::set(int d, int n, const int* jvals, const int* dims) {
  clean();
  bonddir       = d;
  nmoment       = n;
  angularmoment = new int[n];
  bonddim       = new int[n];
  cgcdim        = new int[n];
  std::memcpy(angularmoment, jvals, n * sizeof(int));
  std::memcpy(bonddim,       dims,  n * sizeof(int));
  sort_by_j(n, angularmoment, bonddim);
  for (int i = 0; i < n; ++i)
    cgcdim[i] = angularmoment[i] + 1;   // = 2J + 1
}

SU2Bond& SU2Bond::operator=(const SU2Bond& o) {
  if (this == &o) return *this;
  clean();
  bonddir       = o.bonddir;
  nmoment       = o.nmoment;
  angularmoment = new int[nmoment];
  bonddim       = new int[nmoment];
  cgcdim        = new int[nmoment];
  std::memcpy(angularmoment, o.angularmoment, nmoment * sizeof(int));
  std::memcpy(bonddim,       o.bonddim,       nmoment * sizeof(int));
  std::memcpy(cgcdim,        o.cgcdim,        nmoment * sizeof(int));
  return *this;
}

bool SU2Bond::operator==(const SU2Bond& o) const { return !(*this != o); }

bool SU2Bond::operator!=(const SU2Bond& o) const {
  if (nmoment != o.nmoment || bonddir != o.bonddir) return true;
  for (int i = 0; i < nmoment; ++i)
    if (angularmoment[i] != o.angularmoment[i] || bonddim[i] != o.bonddim[i])
      return true;
  return false;
}

void SU2Bond::print() const {
  cout << "SU2Bond nmoment=" << nmoment << "\tbonddir=" << bonddir << endl;
  for (int i = 0; i < nmoment; ++i)
    cout << angularmoment[i] << "\t" << bonddim[i] << "\t" << cgcdim[i] << endl;
}


bool SU2Bond::check_angularmoment(int j2) const {
  for (int i = 0; i < nmoment; ++i)
    if (angularmoment[i] == j2) return true;
  return false;
}

int SU2Bond::get_angularmoment_index(int j2) const {
  for (int i = 0; i < nmoment; ++i)
    if (angularmoment[i] == j2) return i;
  cout << "SU2Bond::get_angularmoment_index: J=" << j2 << " not found" << endl;
  exit(1);
}

int SU2Bond::total_dim() const {
  int tot = 0;
  for (int i = 0; i < nmoment; ++i) tot += bonddim[i] * cgcdim[i];
  return tot;
}

// Fuse: triangle-rule Clebsch-Gordan decomposition.
// Result direction = -b1.bonddir  (conventional: fusing two outgoing → incoming)
void SU2Bond::fuse(const SU2Bond& b1, const SU2Bond& b2) {
  clean();
  if (b1.bonddir != b2.bonddir) {
    cout << "SU2Bond::fuse: bond directions differ" << endl; exit(1);
  }
  int n1 = b1.nmoment, n2 = b2.nmoment;
  int max_n = 100;
  int* ang = new int[max_n];
  int* dim = new int[max_n];
  int num = 0;
  for (int i = 0; i < n1; ++i) {
    int j1 = b1.angularmoment[i], d1 = b1.bonddim[i];
    for (int j = 0; j < n2; ++j) {
      int j2 = b2.angularmoment[j], d2 = b2.bonddim[j];
      int jmin = abs(j1 - j2), jmax = j1 + j2;
      for (int jk = jmin; jk <= jmax; jk += 2) {
        bool found = false;
        for (int l = 0; l < num; ++l)
          if (ang[l] == jk) { dim[l] += d1 * d2; found = true; break; }
        if (!found) {
          if (num == max_n) {
            max_n *= 2;
            int* ac = new int[max_n]; int* dc = new int[max_n];
            std::memcpy(ac, ang, num * sizeof(int));
            std::memcpy(dc, dim, num * sizeof(int));
            delete[] ang; delete[] dim; ang = ac; dim = dc;
          }
          ang[num] = jk; dim[num] = d1 * d2; ++num;
        }
      }
    }
  }
  sort_by_j(num, ang, dim);
  set(-b1.bonddir, num, ang, dim);
  delete[] ang; delete[] dim;
}

void SU2Bond::fuse(const SU2Bond& b1, const SU2Bond& b2, int dir) {
  clean();
  int n1 = b1.nmoment, n2 = b2.nmoment;
  int max_n = 100;
  int* ang = new int[max_n];
  int* dim = new int[max_n];
  int num = 0;
  for (int i = 0; i < n1; ++i) {
    int j1 = b1.angularmoment[i], d1 = b1.bonddim[i];
    for (int j = 0; j < n2; ++j) {
      int j2 = b2.angularmoment[j], d2 = b2.bonddim[j];
      int jmin = abs(j1 - j2), jmax = j1 + j2;
      for (int jk = jmin; jk <= jmax; jk += 2) {
        bool found = false;
        for (int l = 0; l < num; ++l)
          if (ang[l] == jk) { dim[l] += d1 * d2; found = true; break; }
        if (!found) {
          if (num == max_n) {
            max_n *= 2;
            int* ac = new int[max_n]; int* dc = new int[max_n];
            std::memcpy(ac, ang, num * sizeof(int));
            std::memcpy(dc, dim, num * sizeof(int));
            delete[] ang; delete[] dim; ang = ac; dim = dc;
          }
          ang[num] = jk; dim[num] = d1 * d2; ++num;
        }
      }
    }
  }
  sort_by_j(num, ang, dim);
  set(dir, num, ang, dim);
  delete[] ang; delete[] dim;
}

void SU2Bond::fuse_to_multiplet(const SU2Bond& b1, const SU2Bond& b2, int target_2j) {
  clean();
  if (b1.bonddir != b2.bonddir) {
    cout << "SU2Bond::fuse_to_multiplet: directions differ" << endl; exit(1);
  }
  int n1 = b1.nmoment, n2 = b2.nmoment;
  int dim_tot = 0;
  for (int i = 0; i < n1; ++i) {
    int j1 = b1.angularmoment[i], d1 = b1.bonddim[i];
    for (int j = 0; j < n2; ++j) {
      int j2 = b2.angularmoment[j], d2 = b2.bonddim[j];
      int jmin = abs(j1 - j2), jmax = j1 + j2;
      for (int jk = jmin; jk <= jmax; jk += 2)
        if (jk == target_2j) dim_tot += d1 * d2;
    }
  }
  int ang[1] = { target_2j };
  int dim[1] = { dim_tot };
  set(-b1.bonddir, 1, ang, dim);
}

void SU2Bond::direct_sum(const SU2Bond& b1, const SU2Bond& b2) {
  clean();
  if (b1.bonddir != b2.bonddir) {
    cout << "SU2Bond::direct_sum: directions differ" << endl; exit(1);
  }
  bonddir = b1.bonddir;
  int n1 = b1.nmoment, n2 = b2.nmoment;
  int* aa = new int[n1 + n2];
  int* bb = new int[n1 + n2];
  std::memcpy(aa, b1.angularmoment, n1 * sizeof(int));
  std::memcpy(bb, b1.bonddim,       n1 * sizeof(int));
  nmoment = n1;
  for (int i = 0; i < n2; ++i) {
    int q = b2.angularmoment[i], d = b2.bonddim[i];
    bool found = false;
    for (int j = 0; j < nmoment; ++j)
      if (aa[j] == q) { bb[j] += d; found = true; break; }
    if (!found) { aa[nmoment] = q; bb[nmoment] = d; ++nmoment; }
  }
  sort_by_j(nmoment, aa, bb);
  angularmoment = new int[nmoment];
  bonddim       = new int[nmoment];
  cgcdim        = new int[nmoment];
  for (int i = 0; i < nmoment; ++i) {
    angularmoment[i] = aa[i];
    bonddim[i]       = bb[i];
    cgcdim[i]        = aa[i] + 1;
  }
  delete[] aa; delete[] bb;
}

// ── SU2Struct ─────────────────────────────────────────────────────────────────

SU2Struct::SU2Struct()
  : nbond(0), nten(0), locspin(0),
    bonddir(nullptr), nmoment(nullptr),
    angularmoment(nullptr), bonddim(nullptr), cgcdim(nullptr) {}

SU2Struct::~SU2Struct() { clean(); }

SU2Struct::SU2Struct(int nb, int locsp, const SU2Bond* barr)
  : nbond(0), nten(0), locspin(0),
    bonddir(nullptr), nmoment(nullptr),
    angularmoment(nullptr), bonddim(nullptr), cgcdim(nullptr)
{
  set_su2struct(nb, locsp, barr);
}

void SU2Struct::clean() {
  if (angularmoment) {
    for (int i = 0; i < nbond; ++i) delete[] angularmoment[i];
    delete[] angularmoment; angularmoment = nullptr;
  }
  if (bonddim) {
    for (int i = 0; i < nbond; ++i) delete[] bonddim[i];
    delete[] bonddim; bonddim = nullptr;
  }
  if (cgcdim) {
    for (int i = 0; i < nbond; ++i) delete[] cgcdim[i];
    delete[] cgcdim; cgcdim = nullptr;
  }
  delete[] bonddir;  bonddir  = nullptr;
  delete[] nmoment;  nmoment  = nullptr;
  nbond = nten = locspin = 0;
}

void SU2Struct::set_su2struct(int nb, int locsp, const SU2Bond* barr) {
  clean();
  if (locsp < 0 || nb < 1) {
    cout << "SU2Struct::set: bad args locspin=" << locsp << " nbond=" << nb << endl; exit(1);
  }
  nbond   = nb;
  locspin = locsp;
  bonddir       = new int[nbond];
  nmoment       = new int[nbond];
  angularmoment = new int*[nbond];
  bonddim       = new int*[nbond];
  cgcdim        = new int*[nbond];
  nten = 1;
  for (int i = 0; i < nbond; ++i) {
    bonddir[i] = barr[i].get_bonddir();
    nmoment[i] = barr[i].get_nmoment();
    nten      *= nmoment[i];
    angularmoment[i] = new int[nmoment[i]];
    bonddim[i]       = new int[nmoment[i]];
    cgcdim[i]        = new int[nmoment[i]];
    for (int j = 0; j < nmoment[i]; ++j) {
      angularmoment[i][j] = barr[i].get_angularmoment(j);
      bonddim[i][j]       = barr[i].get_bonddim(j);
      cgcdim[i][j]        = barr[i].get_cgcdim(j);
    }
  }
}

SU2Struct::SU2Struct(const SU2Struct& o)
  : nbond(0), nten(0), locspin(0), bonddir(nullptr), nmoment(nullptr),
    angularmoment(nullptr), bonddim(nullptr), cgcdim(nullptr)
{ *this = o; }

SU2Struct& SU2Struct::operator=(const SU2Struct& o) {
  if (this == &o) return *this;
  SU2Bond* barr = new SU2Bond[o.nbond];
  for (int i = 0; i < o.nbond; ++i) o.get_su2bond(i, barr[i]);
  set_su2struct(o.nbond, o.locspin, barr);
  delete[] barr;
  return *this;
}

bool SU2Struct::operator==(const SU2Struct& o) const { return !(*this != o); }

bool SU2Struct::operator!=(const SU2Struct& o) const {
  if (nbond != o.nbond || locspin != o.locspin || nten != o.nten) return true;
  for (int i = 0; i < nbond; ++i) {
    if (bonddir[i] != o.bonddir[i] || nmoment[i] != o.nmoment[i]) return true;
    for (int j = 0; j < nmoment[i]; ++j)
      if (angularmoment[i][j] != o.angularmoment[i][j] || bonddim[i][j] != o.bonddim[i][j])
        return true;
  }
  return false;
}

void SU2Struct::print() const {
  cout << "SU2Struct: locspin=" << locspin << " nbond=" << nbond << " nten=" << nten << endl;
  for (int i = 0; i < nbond; ++i) {
    cout << "  bond[" << i << "] dir=" << bonddir[i] << " nmoment=" << nmoment[i] << endl;
    for (int j = 0; j < nmoment[i]; ++j)
      cout << "    J=" << angularmoment[i][j] << " dim=" << bonddim[i][j]
           << " cgcdim=" << cgcdim[i][j] << endl;
  }
}

void SU2Struct::take_conjugate() {
  for (int i = 0; i < nbond; ++i) bonddir[i] = -bonddir[i];
}

void SU2Struct::take_conjugate(int ind) {
  if (ind >= nbond) { cout << "SU2Struct::take_conjugate: bad ind=" << ind << endl; exit(1); }
  bonddir[ind] = -bonddir[ind];
}

void SU2Struct::invert_bonddir(int ind) { bonddir[ind] = -bonddir[ind]; }

void SU2Struct::get_su2bond(int i0, SU2Bond& bb) const {
  if (i0 >= nbond) {
    cout << "SU2Struct::get_su2bond: i0=" << i0 << " >= nbond=" << nbond << endl; exit(1);
  }
  bb.set(bonddir[i0], nmoment[i0], angularmoment[i0], bonddim[i0]);
}

int SU2Struct::get_angularmoment_index(int i0, int j2) const {
  if (i0 >= nbond) {
    cout << "SU2Struct::get_angularmoment_index: bond " << i0 << " out of range" << endl; exit(1);
  }
  for (int i = 0; i < nmoment[i0]; ++i)
    if (angularmoment[i0][i] == j2) return i;
  return -1;
}

int SU2Struct::get_tensor_index(const int* moment) const {
  int* aa = new int[nbond];
  for (int i = 0; i < nbond; ++i) {
    aa[i] = get_angularmoment_index(i, moment[i]);
    if (aa[i] == -1) {
      delete[] aa;
      cout<<"SU2Struct::get_tensor_index given moments do not exist"<<endl;
      exit(1);
      //return nten;
    }
  }
  int i0 = 0;
  for (int i = nbond - 1; i >= 0; --i) {
    i0 *= nmoment[i];
    i0 += aa[i];
  }
  delete[] aa;
  return i0;
}

// Check if the given set of J values can be coupled to locspin via triangle rule.
//Note that locspin this value is redundant for the SU2 tensor, one should remove them all from the code, lets keep them for now for the consistency with the U1 symmetry
bool SU2Struct::get_tensor_argument(int i0, int* angm, int* bdim, int* cdim) const {
  if (i0 >= nten) {
    cout << "SU2Struct::get_tensor_argument: i0=" << i0 << " >= nten=" << nten << endl; exit(1);
  }
  int tmp = i0;
  for (int i = 0; i < nbond; ++i) {
    int j = tmp % nmoment[i];
    tmp  /= nmoment[i];
    bdim[i] = bonddim[i][j];
    cdim[i] = cgcdim[i][j];
    angm[i] = angularmoment[i][j];
  }
  return check_angularmoments(angm);
}

bool SU2Struct::get_tensor_argument(int i0, int* angm, int* bdim) const {
  if (i0 >= nten) {
    cout << "SU2Struct::get_tensor_argument: i0=" << i0 << " >= nten=" << nten << endl; exit(1);
  }
  int tmp = i0;
  for (int i = 0; i < nbond; ++i) {
    int j = tmp % nmoment[i];
    tmp  /= nmoment[i];
    bdim[i] = bonddim[i][j];
    angm[i] = angularmoment[i][j];
  }
  return check_angularmoments(angm);
}

bool SU2Struct::check_angularmoments(const int* angm) const {
    if (nbond <= 0) return false;
    if (nbond == 1) return angm[0] == locspin;

    int maxJ = 0;
    for (int i = 0; i < nbond; ++i) maxJ += angm[i];

    std::vector<unsigned char> reach(maxJ + 1, 0), next(maxJ + 1, 0);
    reach[angm[0]] = 1;

    for (int i = 1; i < nbond; ++i) {
        std::fill(next.begin(), next.end(), 0);
        const int j = angm[i];

        for (int J = 0; J <= maxJ; ++J) {
            if (!reach[J]) continue;
            for (int K = std::abs(J - j); K <= J + j; K += 2) {
                next[K] = 1;
            }
        }
        reach.swap(next);
    }

    return (locspin >= 0 && locspin <= maxJ) ? reach[locspin] : false;
}

void SU2Struct::shift(int i0, int i1) {
  if (i0 == i1) return;
  int ishift = (i1 > i0) ? (i1 - i0) : (nbond - (i0 - i1));
  SU2Bond* barr = new SU2Bond[nbond];
  for (int i = 0; i < nbond; ++i)
    get_su2bond(i, barr[(i + ishift) % nbond]);
  int nb = nbond, ts = locspin;
  set_su2struct(nb, ts, barr);
  delete[] barr;
}

void SU2Struct::exchangeindex(int i0, int i1) {
  if (i0 == i1) return;
  SU2Bond* barr = new SU2Bond[nbond];
  for (int i = 0; i < nbond; ++i) {
    if (i == i0) get_su2bond(i1, barr[i]);
    else if (i == i1) get_su2bond(i0, barr[i]);
    else get_su2bond(i, barr[i]);
  }
  int nb = nbond, ts = locspin;
  set_su2struct(nb, ts, barr);
  delete[] barr;
}

// ndata = sum of product of bonddim[i] (reduced matrix elements)
// ncgc  = sum of product of cgcdim[i]  (CGC tensor elements)
void SU2Struct::get_nelement(int& ndata, int& ncgc) const {
  int* angm = new int[nbond];
  int* bdim = new int[nbond];
  int* cdim = new int[nbond];
  ndata = 0; ncgc = 0;
  for (int i = 0; i < nten; ++i) {
    if (get_tensor_argument(i, angm, bdim, cdim)) {
      int k1 = 1, k2 = 1;
      for (int j = 0; j < nbond; ++j) { k1 *= bdim[j]; k2 *= cdim[j]; }
      ndata += k1; ncgc += k2;
    }
  }
  delete[] angm; delete[] bdim; delete[] cdim;
}

void SU2Struct::direct_sum(int ind, const SU2Struct& cgc1, const SU2Struct& cgc2) {
  clean();
  int nb1 = cgc1.nbond, nb2 = cgc2.nbond;
  int ls1 = cgc1.locspin, ls2 = cgc2.locspin;
  if (nb1 != nb2 || ls1 != ls2) {
    cout << "SU2Struct::direct_sum: incompatible structs" << endl; exit(1);
  }
  nbond   = nb1;
  locspin = ls1;
  SU2Bond* bb = new SU2Bond[nbond];
  for (int i = 0; i < nbond; ++i) {
    SU2Bond b1, b2;
    cgc1.get_su2bond(i, b1);
    cgc2.get_su2bond(i, b2);
    if (i != ind) {
      if (b1 == b2) bb[i] = b1;
      else { cout << "SU2Struct::direct_sum: mismatch at bond " << i << endl; delete[] bb; exit(1); }
    } else {
      bb[i].direct_sum(b1, b2);
    }
  }
  set_su2struct(nbond, locspin, bb);
  delete[] bb;
}
