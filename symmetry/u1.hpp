/**
 * symmetry/u1.hpp — U(1) symmetry: bond, struct, and policy class.
 *
 * Replaces: u1bond.hpp, u1bond.cpp, u1struct.hpp, u1struct.cpp
 *           (from mps_u1/)
 *
 * U(1) symmetry uses a conserved integer charge (e.g. 2*S_z or particle
 * number).  Clebsch-Gordan coefficients for U(1) are trivial (±1 phase),
 * so no CGC arrays are needed; the block structure is simpler than SU2.
 *
 * U1Symmetry is the policy class used as the Sym parameter in
 * BlockTensor<Scalar, U1Symmetry> and DMRG<Scalar, U1Symmetry>.
 */

#pragma once
#include <cstring>

// ── U1Bond ───────────────────────────────────────────────────────────────────

class U1Bond {
private:
  int bonddir;        // +1 (outgoing) or -1 (incoming)
  int nmoment;        // number of charge sectors
  int *angularmoment; // charge values [nmoment]  (e.g. 2*Sz: ...-2,-1,0,+1,+2...)
  int *bonddim;       // dimension of each sector [nmoment]

public:
  U1Bond();
  ~U1Bond();
  U1Bond(int bonddir, int nmoment, const int* charges, const int* dims);
  void set(int bonddir, int nmoment, const int* charges, const int* dims);
  void fuse(const U1Bond& a, const U1Bond& b);
  void fuse(const U1Bond& a, const U1Bond& b, int dir);
  void clean();

  U1Bond& operator=(const U1Bond&);
  bool    operator==(const U1Bond&) const;
  bool    operator!=(const U1Bond&) const;

  int get_bonddir()           const { return bonddir; }
  int get_nmoment()           const { return nmoment; }
  int get_angularmoment(int i)const { return angularmoment[i]; }
  int get_bonddim(int i)      const { return bonddim[i]; }
  // U1 has no CGC multiplicity — always 1, provided for interface compatibility
  int get_cgcdim(int /*i*/)   const { return 1; }

  void invert_bonddir()              { bonddir = -bonddir; }
  void set_bonddir(int d)            { bonddir = d; }
  bool check_angularmoment(int q)    const;
  int  get_angularmoment_index(int q)const;
  // Returns true if b is compatible for contraction with *this:
  // opposite bonddir, same number of sectors, matching charges and dims.
  bool check_consistency(const U1Bond& b) const;
  void print()                       const;
  void direct_sum(const U1Bond& a, const U1Bond& b);

  // Total dimension (sum of all block dims)
  int total_dim() const;
};

// ── U1Struct ─────────────────────────────────────────────────────────────────

class U1Struct {
private:
  int nbond;           // number of bonds (3 for standard MPS tensor)
  int nten;            // total number of symmetry-allowed blocks
  int locspin;         // local spin at this site (2*S, e.g. 1 for spin-1/2)
  int *bonddir;        // [nbond]
  int *nmoment;        // [nbond] — number of charge sectors per bond
  int **angularmoment; // [nbond][nmoment[i]]
  int **bonddim;       // [nbond][nmoment[i]]

public:
  U1Struct();
  ~U1Struct();
  U1Struct(const U1Struct&);
  U1Struct(int nbond, int locspin, const U1Bond* bonds);
  void set_u1struct(int nbond, int locspin, const U1Bond* bonds);
  void set_struct(int nbond, int locspin, const U1Bond* bonds)        { set_u1struct(nbond, locspin, bonds); }
  void clean();

  U1Struct& operator=(const U1Struct&);
  bool      operator==(const U1Struct&) const;
  bool      operator!=(const U1Struct&) const;

  int get_nbond()                const { return nbond; }
  int get_nten()                 const { return nten; }
  int get_locspin()              const { return locspin; }
  int get_nmoment(int i)         const { return nmoment[i]; }
  int get_bonddir(int i)         const { return bonddir[i]; }
  int get_angularmoment(int i,int j) const { return angularmoment[i][j]; }
  int get_bonddim(int i, int j)  const { return bonddim[i][j]; }
  // No cgcdim for U1: always 1
  int get_cgcdim(int /*i*/, int /*j*/) const { return 1; }

  void get_u1bond(int bond_idx, U1Bond& out) const;
  void get_bond(int bond_idx, U1Bond& out)   const { get_u1bond(bond_idx, out); }
  // 3-arg convenience wrapper (delegates to 4-arg; consistent with SU2Struct interface)
  bool get_tensor_argument(int block_idx, int* moments,
			   int* reduced_dims) const;
  bool get_tensor_argument(int block_idx, int* moments,
			   int* reduced_dims, int* cdim) const;
  // 4-arg primary: block_idx → moment_indices, block dims, cgc dims (always 1 for U1)
  int  get_angularmoment_index(int bond_idx, int q) const;
  int  get_tensor_index(const int* moment_indices)  const; //input charge values, output tensor index
  bool check_angularmoments(const int* moment_indices) const;

  void take_conjugate();
  void take_conjugate(int bond_idx);
  void print()         const;
  void shift(int bond_idx, int delta);
  void exchangeindex(int i, int j);
  void invert_bonddir(int bond_idx);
  void set_bonddir(int i, int d) { bonddir[i] = d; }
  void get_nelement(int& ndata, int& ncgc) const;
  void direct_sum(int bond_idx, const U1Struct& a, const U1Struct& b);
  // Set *this = contraction of u1 (bond i1) with u2 (bond i2).
  // Remaining bonds of u1 (starting at i1+1) come first, then u2 (starting at i2+1).
  void contract(U1Struct& u1, int i1, U1Struct& u2, int i2);
};

// ── U1Symmetry policy ────────────────────────────────────────────────────────
// Used as Sym template parameter in BlockTensor<Scalar, U1Symmetry>.

struct U1Symmetry {
  using BondType   = U1Bond;
  using StructType = U1Struct;

  // U(1) has no non-trivial CG coefficients; block data is the full tensor.
  static constexpr bool has_cgc = false;

  // Charge conservation: charges must sum to zero (or target total charge)
  // Returns true if (q0, q1, q2) with directions (d0, d1, d2) is allowed.
  static bool charge_allowed(int q0, int d0, int q1, int d1, int q2, int d2) {
    return (d0 * q0 + d1 * q1 + d2 * q2) == 0;
  }
};
