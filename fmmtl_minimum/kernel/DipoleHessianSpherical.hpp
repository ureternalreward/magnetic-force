#pragma once
/** @file LaplaceSpherical.hpp
* @brief Implements the Laplace kernel with spherical expansions.
*
* K(t,s) = 1 / |s-t|        // Laplace potential
* K(t,s) = (s-t) / |s-t|^3  // Laplace force
*/
#include "Dipole_hessian.kern"
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
class DipoleHessianSpherical
  : public fmmtl::Expansion<DipoleHessianKernel, DipoleHessianSpherical> {
public:
  typedef double real_type;
  typedef std::complex<real_type> complex_type;


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
  DipoleHessianSpherical(int _P)
    :P(_P) {
  }

  DipoleHessianSpherical() : P(6) {};

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
  * _TM_n^m=1/(4pi)*sum p\cdot grad(_r^nY_n^-m(\theta,\phi))
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
	//        temp_M._Val[0] = 0;
	//        temp_M._Val[1] = 0;
	temp_M = complex_type{0,0};
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


  //evaluate the second order derivative of Z
  void eval_dZdtdt(complex_type * __restrict out_dZdtdt, complex_type * __restrict const in_dZ, size_t _P,real_type phi) const {
    out_dZdtdt[0] = complex_type{ 0,0 };
    //e^(i(phi-pi/2))
    const complex_type eiphimpi2 = complex_type{ std::cos(phi - M_PI / 2),std::sin(phi - M_PI / 2) };
    const complex_type conjeiphimp2 = conj(eiphimpi2);
    int mn = 1;
    for (int n = 1; n < _P; n++) {
      for (int m = 0; m <= n; m++, mn++) {
        if (m == 0) {
          //mn+1 means Z_n^(m+1)
          out_dZdtdt[mn] = in_dZ[mn + 1]*real_type(n+1)*conjeiphimp2;
        }
        else if (m < n) {
          //0<m<n
          out_dZdtdt[mn] = (-0.5)*(real_type(n-m+1)*in_dZ[mn - 1]*eiphimpi2 - real_type(n+m+1)*in_dZ[mn + 1]*conjeiphimp2);
        }
        else {
          //m=n
          out_dZdtdt[mn] = -0.5*in_dZ[mn - 1]*eiphimpi2;
        }
      }
    }
  }

  struct sin_cos_invr {
    sin_cos_invr(real_type theta, real_type phi, real_type r) {
      using std::sin;
      using std::cos;
      st = sin(theta);
      ct = cos(theta);
      sp = sin(phi);
      cp = cos(phi);
      r = r;
    }
    real_type st, ct, sp, cp, r;
  };
  //hessian in spherical coordinates
  struct sph_hessian {
    sph_hessian() {
      dPhidrr = 0;
      dPhidrt = 0;
      dPhidrp = 0;
      dPhidtt = 0;
      dPhidtp = 0;
      dPhidpp = 0;
    };
    //using new m and n
    //update will change the value in the data
    //the result is the sum of n,+m and n,-m
    void update(const complex_type & _Lmn, complex_type * _Z, complex_type * _dZ, complex_type * _dZdtt, real_type _r, int _m, int _n, int mn) {
      complex_type LZ = _Lmn*_Z[mn];

      if (_m == 0) {
        dPhidrr += real(LZ) * real_type(_n *(_n - 1) / _r / _r);
        dPhidrt += (real(_Lmn) * real(_dZ[mn]) - imag(_Lmn)*imag(_dZ[mn])) * real_type(_n / _r);
        //dPhidrp += 0;
        dPhidtt += real(_Lmn*_dZdtt[mn]);
        //dPhidtp += 0;
        //dPhidpp += 0;
      }
      else {
        //two for -m and +m
        dPhidrr += real(LZ) * real_type(2 * _n *(_n - 1) / _r / _r);
        dPhidrt += (real(_Lmn) * real(_dZ[mn]) - imag(_Lmn)*imag(_dZ[mn])) * real_type(2 * _n / _r);
        dPhidrp += -2.f * imag(LZ)*_m*_n / _r;
        dPhidtt += real_type(2.f)*real(_Lmn*_dZdtt[mn]);
        dPhidtp += -2.f * imag(_Lmn*_dZ[mn])*_m;
        dPhidpp += real(LZ) *real_type(-2 * _m*_m);
      }
    };

    real_type dPhidrr, dPhidrt, dPhidrp, dPhidtt, dPhidtp, dPhidpp;
  };

  real_type sphLaplacian(sph_hessian _H, point_type sph_grad,real_type r, real_type theta, real_type phi) const{
    using std::sin;
    using std::cos;
    return _H.dPhidrr + 
      2 / r*sph_grad[0] 
      + 1 / (r*r*sin(theta)*sin(theta))*_H.dPhidpp 
      + cos(theta) / (r*r*sin(theta))*sph_grad[1]
      + 1 / (r*r)*_H.dPhidtt;
  }
  /* function sphericalHessian2cartHessian
    with the second order derivatives and first order derivatives, generate the negative hessian of potential.
  */
  void sphericalHessian2cartHessian(result_type & result,
                                    const sph_hessian & _in_sphHessian,
                                    const point_type & _in_grad_sph,
                                    real_type st, real_type ct,
                                    real_type sp, real_type cp,
                                    real_type r, real_type invr,
                                    real_type theta, real_type phi) const {
    const real_type & dPhidr = _in_grad_sph[0];
    const real_type & dPhidt = _in_grad_sph[1];
    const real_type & dPhidp = _in_grad_sph[2];
    const real_type mu_0 = 1.25663706e-6;
    point_type negdHx_drtp, negdHy_drtp, negdHz_drtp;
    ///Hx
    negdHx_drtp[0] = cp*st*_in_sphHessian.dPhidrr + cp*ct*invr*_in_sphHessian.dPhidrt - cp*ct*invr*invr*dPhidt
      + sp*invr*invr / st*dPhidp - sp*invr / st*_in_sphHessian.dPhidrp;

    negdHx_drtp[1] = cp*ct*dPhidr + cp*st*_in_sphHessian.dPhidrt - cp*st*invr*dPhidt + cp*ct*invr*_in_sphHessian.dPhidtt
      + ct*sp*invr / (st*st)*dPhidp - sp*invr / st*_in_sphHessian.dPhidtp;

    negdHx_drtp[2] = -sp*st*dPhidr + cp*st*_in_sphHessian.dPhidrp - sp*ct *invr*dPhidt + cp*ct*invr*_in_sphHessian.dPhidtp
      - cp *invr / st*dPhidp - sp *invr / st*_in_sphHessian.dPhidpp;

    point_type negHx_dxyz = mu_0*SphOp::sph2cart(r, theta, phi, negdHx_drtp);
    ///Hy
    negdHy_drtp[0] = sp*st*_in_sphHessian.dPhidrr - sp*ct*invr*invr*dPhidt + sp*ct*invr*_in_sphHessian.dPhidrt
      - cp*invr*invr / (st)*dPhidp + cp*invr / (st)*_in_sphHessian.dPhidrp;

    negdHy_drtp[1] = sp*ct*dPhidr + sp*st*_in_sphHessian.dPhidrt - sp*st*invr*dPhidt + sp*ct*invr*_in_sphHessian.dPhidtt
      - ct*cp*invr / (st*st)*dPhidp + cp*invr / st*_in_sphHessian.dPhidtp;

    negdHy_drtp[2] = cp*st*dPhidr + sp*st*_in_sphHessian.dPhidrp + cp*ct*invr*dPhidt + sp*ct*invr*_in_sphHessian.dPhidtp
      - sp*invr / (st)*dPhidp + cp*invr / (st)*_in_sphHessian.dPhidpp;

    point_type negHy_dxyz = mu_0*SphOp::sph2cart(r, theta, phi, negdHy_drtp);

    ///Hz
    negdHz_drtp[0] = ct*_in_sphHessian.dPhidrr + st*invr*invr*dPhidt - st*invr*_in_sphHessian.dPhidrt;

    negdHz_drtp[1] = -st*dPhidr + ct*_in_sphHessian.dPhidrt - ct*invr*dPhidt - st*invr*_in_sphHessian.dPhidtt;

    negdHz_drtp[2] = ct*_in_sphHessian.dPhidrp - st*invr*_in_sphHessian.dPhidtp;

    point_type negHz_dxyz = mu_0*SphOp::sph2cart(r, theta, phi, negdHz_drtp);

    //point_type temp = SphOp::sph2cart(r, theta, phi, _in_grad_sph);

    //temp2 should equal to the trace of the force matrix.
    //real_type temp2 = mu_0*sphLaplacian(_in_sphHessian, _in_grad_sph, r, theta, phi);
    //unit_test_dZdtheta2();
    result[0][0] -= negHx_dxyz[0];
    result[0][1] -= negHx_dxyz[1];
    result[0][2] -= negHx_dxyz[2];

    result[1][0] -= negHy_dxyz[0];
    result[1][1] -= negHy_dxyz[1];
    result[1][2] -= negHy_dxyz[2];

    result[2][0] -= negHz_dxyz[0];
    result[2][1] -= negHz_dxyz[1];
    result[2][2] -= negHz_dxyz[2];
  }

  void unit_test_dZdtheta2() const{
    int order = 5;
    real_type rho = 1;
    real_type theta = M_PI / 3;
    real_type phi = 3.2*M_PI / 4;
    complex_type *Z = new complex_type[order*(order + 1) / 2];
    complex_type *dZ = new complex_type[order*(order + 1) / 2];
    complex_type *dZdtt = new complex_type[order*(order + 1) / 2];
    //complex_type Z[order*(order + 1) / 2], dZ[order*(order + 1) / 2];
    SphOp::evalZ(rho, theta, phi, order, Z, dZ);
    eval_dZdtdt(dZdtt, dZ, order, phi);

    auto st = std::sin(theta);
    auto ct = std::cos(theta);

    //d^2P_n^m/dtheta^2=d^2P_n^m/dx^2*(1-x^2)+xdP_n^m/dx
    //P44 = 105*(1-x^2)^2
    //d^2P44/dtheta^2=420*sin(theta)^2*(cos(theta)^2 - 1) - 420*cos(theta)^2*(cos(theta)^2 - 1) + 840*cos(theta)^2*sin(theta)^2
    auto dp44dtt = [](real_type ct,real_type st) {
      return 420 * st *st * (ct *ct - 1) - 420 *ct *ct * (ct *ct - 1) + 840 * ct *ct * st *st;
    };
    auto dZ44dtt = [](real_type rho, real_type _val_dp44dtt, real_type phi) {
      const complex_type ei = complex_type(sin(phi), -cos(phi));
      return ei*ei*ei*ei*rho*rho*rho*rho*real_type(1 / 40320.f)*_val_dp44dtt;
    };

    //P43 = -105x*(1-x^2)^3/2

    //P42 = 15/2(7x^2-1)(1-x^2)
    auto dp42dtt = [](real_type ct, real_type st) {
      return 105.f * ct *ct * (ct *ct - 1.f)
        - 105.f * st *st * (ct *ct - 1.f)
        + 2.f * ct *ct * ((105.f * ct *ct) / 2.f - 15.f / 2.f)
        - 2.f * st *st * ((105.f * ct *ct) / 2.f - 15.f / 2.f)
        - 420.f * ct *ct * st *st;
    };
    auto dZ42dtt = [](real_type rho, real_type _val_dp42dtt, real_type phi) {
      const complex_type ei = complex_type(sin(phi), -cos(phi));
      return ei*ei*rho*rho*rho*rho*real_type(1 / 720.f)*_val_dp42dtt;
    };

    //P41 = -5/2*(7x^3-3x)(1-x^2)1/2

    auto val_dp44dtt = dp44dtt(ct, st);
    auto val_dZ44dtt = dZ44dtt(rho, val_dp44dtt, phi);
    auto val_dp42dtt = dp42dtt(ct, st);
    auto val_dZ42dtt = dZ42dtt(rho, val_dp42dtt, phi);

    auto num_dZ44dtt = dZdtt[14];
    auto num_dZ42dtt = dZdtt[12];
    std::cout << "analytical:val_dZ44dtt : " << val_dZ44dtt << std::endl;
    std::cout << "numerical:num_dZ44dtt:" << num_dZ44dtt << std::endl;
    std::cout << "analytical:val_dZ42dtt : " << val_dZ42dtt << std::endl;
    std::cout << "numerical:num_dZ42dtt:" << num_dZ42dtt << std::endl;
  }

  /** Kernel L2T operation
  * _r += Op(L, t) where L is the local expansion and _r is the result
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
    using std::sin;
    using std::cos;

    real_type rho, theta, phi;


    SphOp::cart2sph(rho, theta, phi, Vec<3, real_type>(target) - center);

    real_type st = sin(theta), ct = cos(theta);
    real_type sp = sin(phi), cp = cos(phi);

    real_type r = rho;
    real_type invr = 1 / r;

    complex_type *Z = new complex_type[P*(P + 1) / 2];
    complex_type *dZ = new complex_type[P*(P + 1) / 2];
    complex_type *dZdtt = new complex_type[P*(P + 1) / 2];

    SphOp::evalZ(rho, theta, phi, P, Z, dZ);
    eval_dZdtdt(dZdtt, dZ, P, phi);

    //dhx d _r theta phi
    point_type dhxdrtp = point_type();
    point_type dhydrtp = point_type();
    point_type dhzdrtp = point_type();

    point_type sph = point_type();
    sph_hessian shessian;
    //sph: gradient of the potential, along _r, theta, phi direction.
    //_r, theta, phi is defined relatively to the center of the expansion
    int nm = 0;
    for (int n = 0; n != P; ++n) {

      //m=0, base function Y_n^0 is symmetric around \phi axis, no derivative around phi
      const real_type LZ = real(L[nm])*real(Z[nm]) - imag(L[nm])*imag(Z[nm]);
      sph[0] += LZ / rho * n;
      sph[1] += real(L[nm])*real(dZ[nm]) - imag(L[nm])*imag(dZ[nm]);

      shessian.update(L[nm], Z, dZ, dZdtt, r, 0, n, nm);
      ++nm;
      //m=1->n

      for (int m = 1; m <= n; ++m, ++nm) {
        const complex_type LZ = L[nm] * Z[nm];
        shessian.update(L[nm], Z, dZ, dZdtt, r, m, n, nm);
        sph[0] += 2 * real(LZ) / rho * n;
        sph[1] += 2 * (real(L[nm])*real(dZ[nm]) - imag(L[nm])*imag(dZ[nm]));
        sph[2] += 2 * -imag(LZ) * m;
      }
    }

    //to this step, the first order derivative, the second order derivative in spherical coordinates are complete.
    delete[] Z;
    delete[] dZ;
    delete[] dZdtt;

    sphericalHessian2cartHessian(result, shessian, sph, st, ct, sp, cp, r, invr, theta, phi);

  }
};

}
