/**
 * src/symmetry/u1.cpp — U1Bond and U1Struct implementations.
 * Ported from mps_u1/u1bond.cpp and mps_u1/u1struct.cpp.
 */

#include "../../symmetry/u1.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <execinfo.h>

using std::cout;
using std::endl;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Sort charges and dims together by charge value (ascending)
static void sort_by_charge(int n, int* charges, int* dims) {
  // bubble sort — arrays are typically small (< 50 sectors)
  for (int i = 0; i < n - 1; ++i)
    for (int j = i + 1; j < n; ++j)
      if (charges[j] < charges[i]) {
        std::swap(charges[i], charges[j]);
        std::swap(dims[i],    dims[j]);
      }
}

// ── U1Bond ────────────────────────────────────────────────────────────────────

U1Bond::U1Bond()
  : bonddir(0), nmoment(0), angularmoment(nullptr), bonddim(nullptr) {}

U1Bond::~U1Bond() { clean(); }

U1Bond::U1Bond(int d, int n, const int* q, const int* dims)
  : bonddir(0), nmoment(0), angularmoment(nullptr), bonddim(nullptr)
{
  set(d, n, q, dims);
}

void U1Bond::clean() {
  delete[] angularmoment; angularmoment = nullptr;
  delete[] bonddim;       bonddim       = nullptr;
  nmoment = 0;
  bonddir = 0;
}

void U1Bond::set(int d, int n, const int* q, const int* dims) {
  clean();
  bonddir       = d;
  nmoment       = n;
  angularmoment = new int[n];
  bonddim       = new int[n];
  std::memcpy(angularmoment, q,    n * sizeof(int));
  std::memcpy(bonddim,       dims, n * sizeof(int));
  sort_by_charge(n, angularmoment, bonddim);
}

U1Bond& U1Bond::operator=(const U1Bond& o) {
  if (this == &o) return *this;
  clean();
  bonddir       = o.bonddir;
  nmoment       = o.nmoment;
  angularmoment = new int[nmoment];
  bonddim       = new int[nmoment];
  std::memcpy(angularmoment, o.angularmoment, nmoment * sizeof(int));
  std::memcpy(bonddim,       o.bonddim,       nmoment * sizeof(int));
  return *this;
}

bool U1Bond::operator==(const U1Bond& o) const { return !(*this != o); }

bool U1Bond::operator!=(const U1Bond& o) const {
  if (nmoment != o.nmoment || bonddir != o.bonddir) return true;
  for (int i = 0; i < nmoment; ++i)
    if (angularmoment[i] != o.angularmoment[i] || bonddim[i] != o.bonddim[i])
      return true;
  return false;
}

void U1Bond::print() const {
  cout << "U1Bond nmoment=" << nmoment << "\tbonddir=" << bonddir << endl;
  for (int i = 0; i < nmoment; ++i)
    cout << angularmoment[i] << "\t" << bonddim[i] << endl;
}


bool U1Bond::check_angularmoment(int q) const {
  for (int i = 0; i < nmoment; ++i)
    if (angularmoment[i] == q) return true;
  return false;
}

int U1Bond::get_angularmoment_index(int q) const {
  for (int i = 0; i < nmoment; ++i)
    if (angularmoment[i] == q) return i;
  cout << "U1Bond::get_angularmoment_index: charge " << q << " not found" << endl;
  exit(1);
}

int U1Bond::total_dim() const {
  int tot = 0;
  for (int i = 0; i < nmoment; ++i) tot += bonddim[i];
  return tot;
}

// fuse two U1 bonds: new charge = dir1*q1 + dir2*q2  (sign chosen so result
// has direction -dir1 = -dir2 when directions are equal, i.e. standard fusion)
void U1Bond::fuse(const U1Bond& b1, const U1Bond& b2) {
  clean();
  int n1 = b1.nmoment, n2 = b2.nmoment;
  int dir1 = b1.bonddir, dir2 = b2.bonddir;
  int max_n = n1 * n2;
  int* val = new int[max_n];
  int* dim = new int[max_n];
  int m = 0;
  for (int j = 0; j < n2; ++j)
    for (int i = 0; i < n1; ++i) {
      int c;
      if (dir1 * dir2 == 1)
        c = b1.angularmoment[i] + b2.angularmoment[j];
      else
        c = - b1.angularmoment[i] + b2.angularmoment[j];
      bool found = false;
      for (int k = 0; k < m; ++k)
        if (val[k] == c) { dim[k] += b1.bonddim[i] * b2.bonddim[j]; found = true; break; }
      if (!found) { val[m] = c; dim[m] = b1.bonddim[i] * b2.bonddim[j]; ++m; }
    }
  sort_by_charge(m, val, dim);
  // Standard fusion: two incoming bonds (dir1==dir2) fuse into one outgoing bond (-dir1).
  // This ensures charge neutrality: dir1*Q1 + dir2*Q2 + (-dir1)*Q_fused = 0.
  bonddir       = -dir2;
  nmoment       = m;
  angularmoment = new int[m];
  bonddim       = new int[m];
  std::memcpy(angularmoment, val, m * sizeof(int));
  std::memcpy(bonddim,       dim, m * sizeof(int));
  delete[] val; delete[] dim;
}

// fuse with explicit output direction
void U1Bond::fuse(const U1Bond& b1, const U1Bond& b2, int dir) {
  clean();
  int n1 = b1.nmoment, n2 = b2.nmoment;
  int dir1 = b1.bonddir, dir2 = b2.bonddir;
  int max_n = n1 * n2;
  int* val = new int[max_n];
  int* dim = new int[max_n];
  int m = 0;
  for (int j = 0; j < n2; ++j)
    for (int i = 0; i < n1; ++i) {
      int c = (-dir) * (dir1 * b1.angularmoment[i] + dir2 * b2.angularmoment[j]);
      bool found = false;
      for (int k = 0; k < m; ++k)
        if (val[k] == c) { dim[k] += b1.bonddim[i] * b2.bonddim[j]; found = true; break; }
      if (!found) { val[m] = c; dim[m] = b1.bonddim[i] * b2.bonddim[j]; ++m; }
    }
  sort_by_charge(m, val, dim);
  bonddir       = dir;
  nmoment       = m;
  angularmoment = new int[m];
  bonddim       = new int[m];
  std::memcpy(angularmoment, val, m * sizeof(int));
  std::memcpy(bonddim,       dim, m * sizeof(int));
  delete[] val; delete[] dim;
}

void U1Bond::direct_sum(const U1Bond& b1, const U1Bond& b2) {
  clean();
  if (b1.bonddir != b2.bonddir) {
    cout << "U1Bond::direct_sum: directions differ" << endl; exit(1);
  }
  bonddir = b1.bonddir;
  int n1 = b1.nmoment, n2 = b2.nmoment;
  int* aa = new int[n1 + n2];
  int* bb = new int[n1 + n2];
  std::memcpy(aa, b1.angularmoment, n1 * sizeof(int));
  std::memcpy(bb, b1.bonddim,       n1 * sizeof(int));
  nmoment = n1;
  for (int i = 0; i < n2; ++i) {
    int q = b2.angularmoment[i];
    int d = b2.bonddim[i];
    bool found = false;
    for (int j = 0; j < nmoment; ++j)
      if (aa[j] == q) { bb[j] += d; found = true; break; }
    if (!found) { aa[nmoment] = q; bb[nmoment] = d; ++nmoment; }
  }
  sort_by_charge(nmoment, aa, bb);
  angularmoment = new int[nmoment];
  bonddim       = new int[nmoment];
  std::memcpy(angularmoment, aa, nmoment * sizeof(int));
  std::memcpy(bonddim,       bb, nmoment * sizeof(int));
  delete[] aa; delete[] bb;
}

// ── U1Struct ──────────────────────────────────────────────────────────────────

U1Struct::U1Struct()
  : nbond(0), nten(0), locspin(0),
    bonddir(nullptr), nmoment(nullptr),
    angularmoment(nullptr), bonddim(nullptr) {}

U1Struct::~U1Struct() { clean(); }

U1Struct::U1Struct(int nb, int locsp, const U1Bond* barr)
  : nbond(0), nten(0), locspin(0),
    bonddir(nullptr), nmoment(nullptr),
    angularmoment(nullptr), bonddim(nullptr)
{
  set_u1struct(nb, locsp, barr);
}

void U1Struct::clean() {
  if (angularmoment) {
    for (int i = 0; i < nbond; ++i) delete[] angularmoment[i];
    delete[] angularmoment; angularmoment = nullptr;
  }
  if (bonddim) {
    for (int i = 0; i < nbond; ++i) delete[] bonddim[i];
    delete[] bonddim; bonddim = nullptr;
  }
  delete[] bonddir;  bonddir  = nullptr;
  delete[] nmoment;  nmoment  = nullptr;
  nbond = nten = locspin = 0;
}

void U1Struct::set_u1struct(int nb, int locsp, const U1Bond* barr) {
  clean();
  if (nb < 1) {
    cout << "U1Struct::set: bad nbond=" << nb << endl; exit(1);
  }
  nbond   = nb;
  locspin = locsp;
  bonddir = new int[nbond];
  nmoment = new int[nbond];
  angularmoment = new int*[nbond];
  bonddim       = new int*[nbond];
  nten = 1;
  for (int i = 0; i < nbond; ++i) {
    bonddir[i] = barr[i].get_bonddir();
    nmoment[i] = barr[i].get_nmoment();
    nten      *= nmoment[i];
    angularmoment[i] = new int[nmoment[i]];
    bonddim[i]       = new int[nmoment[i]];
    for (int j = 0; j < nmoment[i]; ++j) {
      angularmoment[i][j] = barr[i].get_angularmoment(j);
      bonddim[i][j]       = barr[i].get_bonddim(j);
    }
  }
}

U1Struct::U1Struct(const U1Struct& o)
  : nbond(0), nten(0), locspin(0), bonddir(nullptr), nmoment(nullptr),
    angularmoment(nullptr), bonddim(nullptr)
{ *this = o; }

U1Struct& U1Struct::operator=(const U1Struct& o) {
  if (this == &o) return *this;
  U1Bond* barr = new U1Bond[o.nbond];
  for (int i = 0; i < o.nbond; ++i) o.get_u1bond(i, barr[i]);
  set_u1struct(o.nbond, o.locspin, barr);
  delete[] barr;
  return *this;
}

bool U1Struct::operator==(const U1Struct& o) const { return !(*this != o); }

bool U1Struct::operator!=(const U1Struct& o) const {
  if (nbond != o.nbond || locspin != o.locspin || nten != o.nten) return true;
  for (int i = 0; i < nbond; ++i) {
    if (bonddir[i] != o.bonddir[i] || nmoment[i] != o.nmoment[i]) return true;
    for (int j = 0; j < nmoment[i]; ++j)
      if (angularmoment[i][j] != o.angularmoment[i][j] || bonddim[i][j] != o.bonddim[i][j])
        return true;
  }
  return false;
}

void U1Struct::print() const {
  cout << "U1Struct: locspin=" << locspin << " nbond=" << nbond << " nten=" << nten << endl;
  for (int i = 0; i < nbond; ++i) {
    cout << "  bond[" << i << "] dir=" << bonddir[i] << " nmoment=" << nmoment[i] << endl;
    for (int j = 0; j < nmoment[i]; ++j)
      cout << "    q=" << angularmoment[i][j] << " dim=" << bonddim[i][j] << endl;
  }
}

void U1Struct::take_conjugate() {
  for (int i = 0; i < nbond; ++i) bonddir[i] = -bonddir[i];
  locspin = -locspin;
}

void U1Struct::take_conjugate(int ind) {
  if (ind >= nbond) { cout << "U1Struct::take_conjugate: bad ind=" << ind << endl; exit(1); }
  bonddir[ind] = -bonddir[ind];
}

void U1Struct::invert_bonddir(int ind) {
  if (ind >= nbond) { cout << "U1Struct::invert_bonddir: bad ind=" << ind << endl; exit(1); }
  bonddir[ind] = -bonddir[ind];
}

void U1Struct::get_u1bond(int i0, U1Bond& bb) const {
  if (i0 >= nbond) {
    cout << "U1Struct::get_u1bond: i0=" << i0 << " >= nbond=" << nbond << " locspin=" << locspin << " nten=" << nten << endl;
    exit(1);
  }
  bb.set(bonddir[i0], nmoment[i0], angularmoment[i0], bonddim[i0]);
}

int U1Struct::get_angularmoment_index(int i0, int q) const {
  if (i0 >= nbond) {
    cout << "U1Struct::get_angularmoment_index: bond " << i0 << " out of range" << endl; exit(1);
  }
  for (int i = 0; i < nmoment[i0]; ++i)
    if (angularmoment[i0][i] == q) return i;
  return -1;
}

int U1Struct::get_tensor_index(const int* moment) const {
  int* aa = new int[nbond];
  for (int i = 0; i < nbond; ++i) {
    aa[i] = get_angularmoment_index(i, moment[i]);
    if (aa[i] == -1) {
      delete[] aa;
      cout<<"U1Struct::get_tensor_index given moments do not exist"<<endl;
      exit(1);
      //return nten;
    }
  }
  int i0 = 0;
  for (int i = nbond - 1; i >= 0; --i) {
    i0 = i0 * nmoment[i] + aa[i];
  }
  delete[] aa;
  return i0;
}

// Decode flat block index i0 → per-bond charge indices, then check charge conservation.
// Fills moment_indices, dims, and cdim (always 1 for U1).
// Returns true if this block is symmetry-allowed.
bool U1Struct::get_tensor_argument(int i0, int* angm, int* bdim, int* /*cdim*/) const {
  return get_tensor_argument(i0, angm, bdim);
}

bool U1Struct::get_tensor_argument(int i0, int* angm, int* bdim) const {
  if (i0 >= nten) {
    cout << "U1Struct::get_tensor_argument: i0=" << i0 << " >= nten=" << nten << endl; exit(1);
  }
  int totchar = 0;
  int tmp = i0;
  for (int i = 0; i < nbond; ++i) {
    int j = tmp % nmoment[i];
    tmp  /= nmoment[i];
    bdim[i] = bonddim[i][j];
    angm[i] = angularmoment[i][j];
    totchar += bonddir[i] * angularmoment[i][j];
  }
  // For multi-bond tensors: locspin holds the net charge of the tensor.
  // Charge-neutral objects (MPS, environments) have locspin=0.
  // Charge-carrying operators (S+, S-) have locspin = their net charge.
  return (totchar == locspin);
}

bool U1Struct::check_angularmoments(const int* angm) const {
  int totchar = 0;
  for (int i = 0; i < nbond; ++i) {
    int j = get_angularmoment_index(i, angm[i]);
    if (j == -1){
      //cout<<"U1Struct::check_angularmoments wrong angm input"<<endl;
      return false;
    }
    totchar += bonddir[i] * angularmoment[i][j];
  }
  return (totchar == locspin);
}

void U1Struct::shift(int i0, int i1) {
  if (i0 == i1) return;
  int ishift = (i1 > i0) ? (i1 - i0) : (nbond - (i0 - i1));
  U1Bond* barr = new U1Bond[nbond];
  for (int i = 0; i < nbond; ++i)
    get_u1bond(i, barr[(i + ishift) % nbond]);
  int nb = nbond, ts = locspin;
  set_u1struct(nb, ts, barr);
  delete[] barr;
}

void U1Struct::exchangeindex(int i0, int i1) {
  if (i0 == i1) return;
  U1Bond* barr = new U1Bond[nbond];
  for (int i = 0; i < nbond; ++i) {
    if (i == i0) get_u1bond(i1, barr[i]);
    else if (i == i1) get_u1bond(i0, barr[i]);
    else get_u1bond(i, barr[i]);
  }
  int nb = nbond, ts = locspin;
  set_u1struct(nb, ts, barr);
  delete[] barr;
}

// Count total number of data elements across all allowed blocks.
// For U1: ndata = sum of product of dims;  ncgc = ndata (no separate CGC)
void U1Struct::get_nelement(int& ndata, int& /*ncgc*/) const {
  int* angm = new int[nbond];
  int* bdim = new int[nbond];
  ndata = 0;
  for (int i = 0; i < nten; ++i) {
    if (get_tensor_argument(i, angm, bdim)) {
      int k = 1;
      for (int j = 0; j < nbond; ++j) k *= bdim[j];
      ndata += k;
    }
  }
  delete[] angm; delete[] bdim;
}

void U1Struct::direct_sum(int ind, const U1Struct& cgc1, const U1Struct& cgc2) {
  clean();
  int nb1 = cgc1.nbond, nb2 = cgc2.nbond;
  int ls1 = cgc1.locspin, ls2 = cgc2.locspin;
  if (nb1 != nb2 || ls1 != ls2) {
    cout << "U1Struct::direct_sum: incompatible structs" << endl; exit(1);
  }
  nbond   = nb1;
  locspin = ls1;
  U1Bond* bb = new U1Bond[nbond];
  for (int i = 0; i < nbond; ++i) {
    U1Bond b1, b2;
    cgc1.get_u1bond(i, b1);
    cgc2.get_u1bond(i, b2);
    if (i != ind) {
      if (b1 == b2) bb[i] = b1;
      else { cout << "U1Struct::direct_sum: mismatch at bond " << i << endl; delete[] bb; exit(1); }
    } else {
      bb[i].direct_sum(b1, b2);
    }
  }
  set_u1struct(nbond, locspin, bb);
  delete[] bb;
}

// ── U1Bond::check_consistency ─────────────────────────────────────────────────

// Returns true if b is compatible for contraction with *this:
//   directions must be opposite (+1/-1), sectors must match exactly.
bool U1Bond::check_consistency(const U1Bond& b) const {
  if (bonddir * b.bonddir != -1) return false;
  if (nmoment != b.nmoment)      return false;
  for (int i = 0; i < nmoment; ++i) {
    if (bonddim[i]       != b.bonddim[i])       return false;
    if (angularmoment[i] != b.angularmoment[i]) return false;
  }
  return true;
}

// ── U1Struct::contract ────────────────────────────────────────────────────────

// Set *this = result of contracting u1 (bond i1) with u2 (bond i2).
// The remaining bonds of u1 (cyclic order starting at i1+1) come first,
// followed by remaining bonds of u2 (cyclic order starting at i2+1).
// Mirrors u1charge::contract in the original code.
void U1Struct::contract(U1Struct& u1, int i1, U1Struct& u2, int i2) {
  U1Bond b1, b2;
  u1.get_u1bond(i1, b1);
  u2.get_u1bond(i2, b2);
  if (!b1.check_consistency(b2)) {
    cout << "U1Struct::contract: bond info not consistent" << endl;
    b1.print(); b2.print();
    exit(0);
  }
  clean();
  int nbond1  = u1.get_nbond();
  int nbond2  = u2.get_nbond();
  int nb      = nbond1 + nbond2 - 2;
  int totchar = u1.get_locspin() + u2.get_locspin();
  U1Bond* barr = new U1Bond[nb];
  for (int i = 0; i < nbond1 - 1; ++i)
    u1.get_u1bond((i1 + 1 + i) % nbond1, barr[i]);
  for (int i = 0; i < nbond2 - 1; ++i)
    u2.get_u1bond((i2 + 1 + i) % nbond2, barr[i + nbond1 - 1]);
  set_u1struct(nb, totchar, barr);
  delete[] barr;
  // nten is recomputed inside set()
}
