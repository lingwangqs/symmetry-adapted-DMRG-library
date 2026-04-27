/**
 * symmetry/su2.hpp — SU(2) symmetry: bond, struct, and policy class.
 *
 * Replaces: su2bond.hpp, su2bond.cpp, su2struct.hpp, su2struct.cpp
 *           (from mps_real/ and mps_u1/ — identical between both)
 *
 * SU(2) blocks are labelled by angular momentum J (stored as 2J ∈ ℤ≥0).
 * Within each (J_left, J_phys, J_right) block, the actual tensor element is
 *
 *   A^{J}_{m_L m_P m_R} = Â^J · <J_L m_L; J_P m_P | J_R m_R>   (Wigner-Eckart)
 *
 * where Â^J is the reduced matrix element (the actual variational parameter)
 * and <·|·> is the Clebsch-Gordan coefficient (stored separately as real).
 *
 * SU2Symmetry is the policy class used as the Sym parameter in
 * BlockTensor<Scalar, SU2Symmetry> and DMRG<Scalar, SU2Symmetry>.
 */

#pragma once
#include <cstring>
#include <cmath>

// ── SU2Bond ──────────────────────────────────────────────────────────────────

class SU2Bond {
private:
  int bonddir;        // +1 (outgoing) or -1 (incoming)
  int nmoment;        // number of J sectors
  int *angularmoment; // 2*J values [nmoment]  (e.g. 0,1,2 for J=0,1/2,1)
  int *bonddim;       // dimension of reduced (J-independent) space [nmoment]
  int *cgcdim;        // multiplicity of m-states = 2J+1 = angularmoment[i]+1 [nmoment]

public:
  SU2Bond();
  ~SU2Bond();
  SU2Bond(int bonddir, int nmoment, const int* jvals, const int* dims);
  void set(int bonddir, int nmoment, const int* jvals, const int* dims);
  // fuse two SU2 bonds using triangle rule (Clebsch-Gordan decomposition)
  void fuse(const SU2Bond& a, const SU2Bond& b);
  void fuse(const SU2Bond& a, const SU2Bond& b, int dir);
  void fuse_to_multiplet(const SU2Bond& a, const SU2Bond& b, int target_2j);
  void clean();

  SU2Bond& operator=(const SU2Bond&);
  bool     operator==(const SU2Bond&) const;
  bool     operator!=(const SU2Bond&) const;

  int get_bonddir()           const { return bonddir; }
  int get_nmoment()           const { return nmoment; }
  int get_angularmoment(int i)const { return angularmoment[i]; }
  int get_bonddim(int i)      const { return bonddim[i]; }
  int get_cgcdim(int i)       const { return cgcdim[i]; }   // = 2J+1

  void invert_bonddir()             { bonddir = -bonddir; }
  void set_bonddir(int d)           { bonddir = d; }
  bool check_angularmoment(int j2)  const;
  int  get_angularmoment_index(int j2) const;
  void print()                      const;
  void direct_sum(const SU2Bond& a, const SU2Bond& b);

  // Full (uncompressed) bond dimension: sum_i bonddim[i] * cgcdim[i]
  int total_dim() const;
};

// ── SU2Struct ─────────────────────────────────────────────────────────────────

class SU2Struct {
private:
  int nbond;           // number of bonds (3 for standard MPS tensor)
  int nten;            // number of symmetry-allowed blocks  (= nten in SU2 counting)
  int locspin;         // 2*J of the local physical spin
  int *bonddir;        // [nbond]
  int *nmoment;        // [nbond] — number of J sectors per bond
  int **angularmoment; // [nbond][nmoment[i]] — 2*J values
  int **bonddim;       // [nbond][nmoment[i]] — reduced dimensions
  int **cgcdim;        // [nbond][nmoment[i]] — 2*J+1  (CGC multiplicities)

public:
  SU2Struct();
  ~SU2Struct();
  SU2Struct(const SU2Struct&);
  SU2Struct(int nbond, int locspin, const SU2Bond* bonds);
  void set_su2struct(int nbond, int locspin, const SU2Bond* bonds);
  void set_struct(int nbond, int locspin, const SU2Bond* bonds)        { set_su2struct(nbond, locspin, bonds); }
  void clean();

  SU2Struct& operator=(const SU2Struct&);
  bool       operator==(const SU2Struct&) const;
  bool       operator!=(const SU2Struct&) const;

  int get_nbond()                    const { return nbond; }
  int get_nten()                     const { return nten; }
  int get_locspin()                  const { return locspin; }
  int get_nmoment(int i)             const { return nmoment[i]; }
  int get_bonddir(int i)             const { return bonddir[i]; }
  int get_angularmoment(int i, int j)const { return angularmoment[i][j]; }
  int get_bonddim(int i, int j)      const { return bonddim[i][j]; }
  int get_cgcdim(int i, int j)       const { return cgcdim[i][j]; }

  void get_su2bond(int bond_idx, SU2Bond& out) const;
  void get_bond(int bond_idx, SU2Bond& out)    const { get_su2bond(bond_idx, out); }
  // Overload that matches BlockTensor's uniform 2-array interface
  bool get_tensor_argument(int block_idx, int* moments,
			   int* reduced_dims) const;
  bool get_tensor_argument(int block_idx, int* moments,
                           int* reduced_dims, int* cgc_dims) const;
  int  get_angularmoment_index(int bond_idx, int j2) const;
  int  get_tensor_index(const int* moment_indices)   const;//input angularmoment values and output tensor index
  // block_idx → (i0,i1,i2) and reduced dims + CGC dims
  bool check_angularmoments(const int* moment_indices) const;

  void take_conjugate();
  void take_conjugate(int bond_idx);
  void print()         const;
  void shift(int bond_idx, int delta);
  void exchangeindex(int i, int j);
  void invert_bonddir(int bond_idx);
  void set_bonddir(int i, int d) { bonddir[i] = d; }
  // ndata  = number of variational (reduced) matrix elements across all blocks
  // ncgc   = number of CGC coefficient values across all blocks
  void get_nelement(int& ndata, int& ncgc) const;
  void direct_sum(int bond_idx, const SU2Struct& a, const SU2Struct& b);
};

// ── Clebsch-Gordan coefficient table ─────────────────────────────────────────
// Build and look up <j1 m1; j2 m2 | j3 m3> (all 2*j, 2*m conventions).

class CGCTable {
public:
  // Compute single CGC: <j1 m1; j2 m2 | j3 m3>  (all arguments = 2*J, 2*M)
  static double cg(int j1, int m1, int j2, int m2, int j3, int m3);

  // Fill a dense [cgcdim1, cgcdim2, cgcdim3] tensor with the CGCs
  // for the block (j1, j2, j3) with all m values.
  static void fill_block(int j1, int j2, int j3, double* buf);
};

// ── SU2Symmetry policy ────────────────────────────────────────────────────────
// Used as Sym template parameter in BlockTensor<Scalar, SU2Symmetry>.

struct SU2Symmetry {
  using BondType   = SU2Bond;
  using StructType = SU2Struct;

  // SU(2) has non-trivial CGC coefficients stored as a separate real tensor
  // alongside each block's reduced matrix elements.
  static constexpr bool has_cgc = true;

  // Triangle rule: |j1 - j2| <= j3 <= j1 + j2,  j1+j2+j3 even
  static bool triangle(int j1, int j2, int j3) {
    if (j3 < abs(j1 - j2)) return false;
    if (j3 > j1 + j2)      return false;
    if ((j1 + j2 + j3) % 2 != 0) return false;
    return true;
  }

  // Check if three bonds (with their J values and directions) form an
  // SU(2)-allowed block: triangle rule AND J conservation with signs.
  static bool block_allowed(int j1, int d1, int j2, int d2, int j3, int d3) {
    // For an MPS tensor with bonds (left, phys, right):
    //   d_left=+1, d_phys=+1, d_right=-1  (conventional)
    // The allowed block satisfies the triangle rule.
    // Direction signs are used in fusing; this just checks triangle.
    return triangle(j1, j2, j3);
  }
};
