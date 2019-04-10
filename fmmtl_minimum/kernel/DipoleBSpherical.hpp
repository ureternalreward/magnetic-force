#pragma once
/** @file LaplaceSpherical.hpp
* @brief Implements the Laplace kernel with spherical expansions.
*
* K(t,s) = 1 / |s-t|        // Laplace potential
* K(t,s) = (s-t) / |s-t|^3  // Laplace force
*/
#include "Dipole_B.kern"
#include "Util/SphericalMultipole3D.hpp"

#include "fmmtl/Expansion.hpp"
#include "fmmtl/numeric/Vec.hpp"
#include "fmmtl/numeric/Complex.hpp"

#include <complex>
#define _USE_MATH_DEFINES
#include <cmath>
#include <type_traits>



namespace fmmtl {

//template <int P=5>
class DipoleBSpherical
  : public fmmtl::Expansion<DipoleBKernel, DipoleBSpherical> {
public:
  typedef double real_type;
  typedef std::complex<real_type> complex_type;


  /*Vec<3, double> promote_vec(const Vec<3, float> & _in) {
  Vec<3, double> result;
  for (size_t i = 0; i < 3; i++) {
  result[i] = (double)_in[i];
  }
  return result;
  }

  const Vec<3, double>& promote_vec(const Vec<3, double> & _in) {
  return _in;
  }*/

  //! Point type
  typedef Vec<3, real_type> point_type;

  //! Multipole expansion type
  typedef std::vector<complex_type> multipole_type;
  //! Local expansion type
  typedef std::vector<complex_type> local_type;

  //! Transform operators
  typedef SphericalMultipole3D<point_type, multipole_type, local_type> SphOp;

  //! Expansion order
  const int P;

  //! Constructor
  DipoleBSpherical(int _P)
    :P(_P) {
  }

  DipoleBSpherical() : P(6) {};

  /** Initialize a multipole expansion with the size of a box at this level */
  void init_multipole(multipole_type& M, const point_type&, unsigned) const {
    M = std::vector<complex_type>(P*(P + 1) / 2);
  }
  /** Initialize a local expansion with the size of a box at this level */
  void init_local(local_type& L, const point_type&, unsigned) const {
    L = std::vector<complex_type>(P*(P + 1) / 2);
  }

  point_type S2P(const source_type& _in) const {
    return point_type{ real_type(_in[0]),real_type(_in[1]),real_type(_in[2]) };
  }

  point_type T2P(const target_type& _in) const {
    return point_type{ real_type(_in[0]),real_type(_in[1]),real_type(_in[2]) };
  }


  /** Kernel S2M operation
  * M += Op(s) * c where M is the multipole and s is the source
  *
  * @param[in] source The point source
  * @param[in] charge The source's corresponding charge
  * @param[in] center The center of the box containing the multipole expansion
  * @param[in,out] M The multipole expansion to accumulate into
  *  _AM_n^m=(-1)^m*i^(|m|)A_n^m_TM_n^m
  * _AM_n^m the actual multipole expansion
  * _TM_n^m theoretical multipole expansion
  * _TM_n^m=1/(4pi)*sum p\cdot grad(r^nY_n^-m(\theta,\phi))
  */
  void S2M(const source_type& source,
           const charge_type& charge,
           const point_type& center,
           multipole_type& M) const {
    real_type rho, theta, phi;
    SphOp::cart2sph(rho, theta, phi, Vec<3, real_type>(source) - Vec<3, real_type>(center));
    if (theta == 0) {
      theta = 1e-20;
    }
    complex_type *Z = new complex_type[P*(P + 1) / 2];   // Avoid initialization?
    complex_type *dZ = new complex_type[P*(P + 1) / 2]; //dZ/dtheta
    SphOp::evalZ(rho, theta, phi, P, Z, dZ);

    //derivatives in spherical coordinates
    complex_type ddr, ddt, ddp;

    //derivatives in cartesian coordinates
    point_type Re_ddxyz;
    point_type Im_ddxyz;
    real_type neg1powmn;

    int nm = 0;   // n*(n+1)/2+m
    complex_type temp_M{ 0,0 };
    for (int n = 0; n < P; ++n) {
      for (int m = 0; m <= n; ++m, ++nm) {
        temp_M._Val[0] = 0;
        temp_M._Val[1] = 0;
        neg1powmn = SphOp::neg1pow(m + n);
        ddr = neg1powmn*n*conj(Z[nm]) / rho;
        ddt = neg1powmn*conj(dZ[nm]);
        ddp = -neg1powmn*m*complex_type{ 0, 1 }*conj(Z[nm]);
        auto tempvec = Vec<3, real_type>{ ddr.real(), ddt.real(), ddp.real() };
        Re_ddxyz = SphOp::sph2cart(rho, theta, phi, tempvec);
        tempvec = Vec<3, real_type>{ ddr.imag(), ddt.imag(), ddp.imag() };
        Im_ddxyz = SphOp::sph2cart(rho, theta, phi, tempvec);
        for (int i = 0; i < 3; i++) {
          temp_M += real_type(charge[i]) * complex_type { Re_ddxyz[i], Im_ddxyz[i] };
        }
        M[nm] += temp_M*0.25*M_1_PI;
      }
    }
    delete[] Z;
    delete[] dZ;
  }

  /** Kernel M2M operator
  * M_t += Op(M_s) where M_t is the target and M_s is the source
  *
  * @param[in] source The multipole source at the child level
  * @param[in,out] target The multipole target to accumulate into
  * @param[in] translation The vector from source to target
  * @pre Msource includes the influence of all points within its box
  */
  void M2M(const multipole_type& Msource,
           multipole_type& Mtarget,
           const point_type& translation) const {
    return SphOp::M2M(P, Msource, Mtarget, translation);
  }

  /** Kernel M2L operation
  * L += Op(M)
  *
  * @param[in] Msource The multpole expansion source
  * @param[in,out] Ltarget The local expansion target
  * @param[in] translation The vector from source to target
  * @pre translation obeys the multipole-acceptance criteria
  * @pre Msource includes the influence of all points within its box
  */
  void M2L(const multipole_type& Msource,
           local_type& Ltarget,
           const point_type& translation) const {
    return SphOp::M2L(P, Msource, Ltarget, translation);
  }

  /** Kernel L2L operator
  * L_t += Op(L_s) where L_t is the target and L_s is the source
  *
  * @param[in] source The local source at the parent level
  * @param[in,out] target The local target to accumulate into
  * @param[in] translation The vector from source to target
  * @pre Lsource includes the influence of all points outside its box
  */
  void L2L(const local_type& Lsource,
           local_type& Ltarget,
           const point_type& translation) const {
    return SphOp::L2L(P, Lsource, Ltarget, translation);
  }

  /** Kernel L2T operation
  * r += Op(L, t) where L is the local expansion and r is the result
  *
  * @param[in] L The local expansion
  * @param[in] center The center of the box with the local expansion
  * @param[in] target The target of this L2T operation
  * @param[in] result The result to accumulate into
  * @pre L includes the influence of all sources outside its box
  */
  void L2T(const local_type& L, const point_type& center,
           const target_type& target, result_type& result) const {
    using std::real;
    using std::imag;

    real_type rho, theta, phi;
    SphOp::cart2sph(rho, theta, phi, Vec<3, real_type>(target) - center);
    complex_type *Z = new complex_type[P*(P + 1) / 2];
    complex_type *dZ = new complex_type[P*(P + 1) / 2];
    //complex_type Z[P*(P + 1) / 2], dZ[P*(P + 1) / 2];
    SphOp::evalZ(rho, theta, phi, P, Z, dZ);

    point_type sph = point_type();
    //sph: gradient of the potential, along r, theta, phi direction.
    //r, theta, phi is defined relatively to the center of the expansion
    int nm = 0;
    for (int n = 0; n != P; ++n) {
      //m=0, base function Y_n^0 is symmetric around \phi axis, no derivative around phi
      const real_type LZ = real(L[nm])*real(Z[nm]) - imag(L[nm])*imag(Z[nm]);
      sph[0] += LZ / rho * n;
      sph[1] += real(L[nm])*real(dZ[nm]) - imag(L[nm])*imag(dZ[nm]);

      ++nm;
      //m=1->n
      for (int m = 1; m <= n; ++m, ++nm) {
        const complex_type LZ = L[nm] * Z[nm];
        sph[0] += 2 * real(LZ) / rho * n;
        sph[1] += 2 * (real(L[nm])*real(dZ[nm]) - imag(L[nm])*imag(dZ[nm]));
        sph[2] += 2 * -imag(LZ) * m;
      }
    }

    const point_type cart = SphOp::sph2cart(rho, theta, phi, sph);
    //negative gradient is the field direction.
    result[0] += -cart[0];
    result[1] += -cart[1];
    result[2] += -cart[2];
    delete[] Z;
    delete[] dZ;
  }
};

}
