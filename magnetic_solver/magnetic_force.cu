#include "magnetic_force.cuh"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
namespace Libo {


//cuda kernels
__global__ void
cuda_Apply_G(const float * x, const float * y, const float * z, float * out_hx, float * out_hy, float * out_hz, const float * in_hx, const float * in_hy, const float * in_hz, const size_t np, const  float h) {
  //maximum load size
  const int batch_np = 512;
  float curr_out_hx = 0;
  float curr_out_hy = 0;
  float curr_out_hz = 0;

  float rx = 0;
  float ry = 0;
  float rz = 0;
  float r = 0;
  float invr = 1;

  float w_avr = 0;
  float w_dens = 0;

  float temp_v1 = 0;
  float temp_v2 = 0;

  float q = 0;
  float coe = 0;

  int i = blockDim.x*blockIdx.x + threadIdx.x;

  int idx_g = 0;
  //prepare space for shared memory
  __shared__ float s_x[batch_np];
  __shared__ float s_y[batch_np];
  __shared__ float s_z[batch_np];
  __shared__ float s_in_hx[batch_np];
  __shared__ float s_in_hy[batch_np];
  __shared__ float s_in_hz[batch_np];

  //number of batches of data
  int load_size = 0;
  for (int load_offset = 0; load_offset < np; load_offset += batch_np) {
    load_size = np - load_offset > batch_np ? batch_np : np - load_offset;
    idx_g = load_offset + threadIdx.x;
    if (threadIdx.x < load_size) {
      s_x[threadIdx.x] = x[idx_g];
      s_y[threadIdx.x] = y[idx_g];
      s_z[threadIdx.x] = z[idx_g];
      s_in_hx[threadIdx.x] = in_hx[idx_g];
      s_in_hy[threadIdx.x] = in_hy[idx_g];
      s_in_hz[threadIdx.x] = in_hz[idx_g];
    }
    __syncthreads();
    load_size--;
    //determine the number of sources loaded this time


    //add the influence of those particles in the shared memory to the current point
    if (i < np) {
      for (; load_size >= 0; load_size--) {
        //r distance
        rx = x[i] - s_x[load_size];
        ry = y[i] - s_y[load_size];
        rz = z[i] - s_z[load_size];
        r = sqrtf(rx*rx + ry*ry + rz*rz);

        q = r / h;
        coe = 1 / CUDART_PI_F / (h*h*h);

        if (q >= 2) {
          w_avr = 3.f / (4 * CUDART_PI_F * r*r*r);
          w_dens = 0;
        }
        if (q >= 1 && q < 2) {
          w_avr = -(3.f * (q*q*q*q*q*q / 6.f - (6.f * q*q*q*q*q) / 5.f + 3.f * q*q*q*q - (8.f * q*q*q) / 3.f + 1.f / 15)) / (4 * CUDART_PI_F*r*r*r);
          w_dens = coe*0.25*(2 - q)*(2 - q)*(2 - q);
        }
        if (q < 1) {
          w_avr = (15.f * q*q*q - 36.f * q*q + 40.f) / (40 * CUDART_PI_F*h*h*h);
          w_dens = coe*(0.25*(2 - q)*(2 - q)*(2 - q) - (1 - q)*(1 - q)*(1 - q));
        }

        if (r != 0) {
          invr = 1 / r;
          //normalized r below
          rx = rx*invr;
          ry = ry*invr;
          rz = rz*invr;
        }

        temp_v1 = (rx*s_in_hx[load_size] + ry*s_in_hy[load_size] + rz*s_in_hz[load_size])*(w_avr - w_dens);
        temp_v2 = w_avr *0.33333333333333333333333f - w_dens;

        curr_out_hx += rx*temp_v1 - temp_v2*s_in_hx[load_size];
        curr_out_hy += ry*temp_v1 - temp_v2*s_in_hy[load_size];
        curr_out_hz += rz*temp_v1 - temp_v2*s_in_hz[load_size];
      }
    }
    __syncthreads();
  }

  //write the current out_hx,out_hy,out_hz to the global memory
  if (i < np) {
    out_hx[i] = curr_out_hx;
    out_hy[i] = curr_out_hy;
    out_hz[i] = curr_out_hz;
  }
}

//function cuda_ext_field_kernel
//the position and field for output:
//const float * x, const float * y, const float * z, 
//float * out_hx, float * out_hy, float * out_hz, 
//the position and field for input:
//const float * in_x, const float * in_y, const float * in_z,
//const float * in_hx, const float * in_hy, const float * in_hz,
//each thread is in charge of one output particle.
__global__ void
cuda_ext_field_kernel(const float * x, const float * y, const float * z,
                      float * out_hx, float * out_hy, float * out_hz,
                      const float * in_x, const float * in_y, const float * in_z,
                      const float * in_hx, const float * in_hy, const float * in_hz,
                      const float3 constHext,
                      const size_t np_target,
                      const size_t np_source,
                      const  float h) {
  //maximum load size
  const int batch_np = 512;
  const int bd = blockDim.x;
  float curr_out_hx = 0;
  float curr_out_hy = 0;
  float curr_out_hz = 0;

  float rx = 0;
  float ry = 0;
  float rz = 0;
  float r = 0;
  float invr = 1;

  float w_avr = 0;
  float w_dens = 0;

  float temp_v1 = 0;
  float temp_v2 = 0;

  float q = 0;
  float coe = 0;

  int i = bd*blockIdx.x + threadIdx.x;


  int idx_g = 0;
  //prepare space for shared memory
  __shared__ float s_x[batch_np];
  __shared__ float s_y[batch_np];
  __shared__ float s_z[batch_np];
  __shared__ float s_in_hx[batch_np];
  __shared__ float s_in_hy[batch_np];
  __shared__ float s_in_hz[batch_np];

  //number of batches of data
  int total_batches = np_source / batch_np;
  if (np_source%batch_np != 0) total_batches++;

  int load_offset = 0;
  int sources_in_smem = 0;
  for (load_offset = 0; load_offset < np_source; load_offset += batch_np) {
    sources_in_smem = np_source - load_offset > batch_np ? batch_np : np_source - load_offset;

    //load a bunch of particles into the shared memory
    idx_g = load_offset + threadIdx.x;
    if ((threadIdx.x < sources_in_smem)) {
      s_x[threadIdx.x] = in_x[idx_g];
      s_y[threadIdx.x] = in_y[idx_g];
      s_z[threadIdx.x] = in_z[idx_g];
      s_in_hx[threadIdx.x] = in_hx[idx_g];
      s_in_hy[threadIdx.x] = in_hy[idx_g];
      s_in_hz[threadIdx.x] = in_hz[idx_g];
    }
    __syncthreads();

    //determine the number of sources loaded this time

    sources_in_smem--;
    //add the influence of those particles in the shared memory to the current point
    if (i < np_target) {
      for (; sources_in_smem >= 0; sources_in_smem--) {
        //r distance
        rx = x[i] - s_x[sources_in_smem];
        ry = y[i] - s_y[sources_in_smem];
        rz = z[i] - s_z[sources_in_smem];
        r = sqrtf(rx*rx + ry*ry + rz*rz);

        q = r / h;
        coe = 1 / CUDART_PI_F / (h*h*h);

        if (q >= 2) {
          w_avr = 3.f / (4 * CUDART_PI_F * r*r*r);
          w_dens = 0;
        }
        if (q >= 1 && q < 2) {
          w_avr = -(3.f * (q*q*q*q*q*q / 6.f - (6.f * q*q*q*q*q) / 5.f + 3.f * q*q*q*q - (8.f * q*q*q) / 3.f + 1.f / 15)) / (4 * CUDART_PI_F*r*r*r);
          w_dens = coe*0.25*(2 - q)*(2 - q)*(2 - q);
        }
        if (q < 1) {
          w_avr = (15.f * q*q*q - 36.f * q*q + 40.f) / (40 * CUDART_PI_F*h*h*h);
          w_dens = coe*(0.25*(2 - q)*(2 - q)*(2 - q) - (1 - q)*(1 - q)*(1 - q));
        }

        if (r != 0) {
          invr = 1 / r;
          //normalized r below
          rx = rx*invr;
          ry = ry*invr;
          rz = rz*invr;
        }

        temp_v1 = (rx*s_in_hx[sources_in_smem] + ry*s_in_hy[sources_in_smem] + rz*s_in_hz[sources_in_smem])*(w_avr - w_dens);
        temp_v2 = w_avr *0.3333333333333f - w_dens;

        curr_out_hx += rx*temp_v1 - temp_v2*s_in_hx[sources_in_smem];
        curr_out_hy += ry*temp_v1 - temp_v2*s_in_hy[sources_in_smem];
        curr_out_hz += rz*temp_v1 - temp_v2*s_in_hz[sources_in_smem];
      }
    }
    __syncthreads();
  }

  //write the current out_hx,out_hy,out_hz to the global memory
  if (i < np_target) {
    out_hx[i] = curr_out_hx + constHext.x;
    out_hy[i] = curr_out_hy + constHext.y;
    out_hz[i] = curr_out_hz + constHext.z;
  }
}

__device__ void get_far_field_force_tensor(
  glm::mat3 & Bij,
  const glm::vec3 & r_vec,
  const glm::vec3 & s_vec,
  const float & q,
  const float & h) {
  float r = q*h;
  float w_avr = 3.f / (4 * CUDART_PI_F * r*r*r);
  float invr = 1;
  if (r != 0) {
    invr = float(1) / r;
  }

  //calculate the gradHferro at this point
  //formulas see plan.tm
  const float mu_0 = 1.25663706e-6;
  float A = mu_0*invr*invr*(w_avr);
  float rdotm = dot(r_vec, s_vec);
  float dAdr = mu_0*invr*invr*(5 * invr*(-w_avr))*rdotm*invr;
  /*dBdr = 3 * invr*(w_dens - w_avr)*invr/3;*/
  float dBdr = mu_0*invr*(-w_avr)*invr;

  const float & rx = r_vec.x;
  const float & ry = r_vec.y;
  const float & rz = r_vec.z;

  //Gxx
  Bij[0][0] += rx*s_vec.x * A + rx*dAdr*rx - s_vec.x * dBdr* rx;
  Bij[0][0] += rdotm*A;
  //Gxy
  //Gxy->B01->Bij[1][0]
  Bij[1][0] += rx*s_vec.y * A + rx*dAdr*ry - s_vec.x * dBdr* ry;
  //Gxz
  //Gxz->B02->Bij[2][0]
  Bij[2][0] += rx*s_vec.z * A + rx*dAdr*rz - s_vec.x * dBdr* rz;


  //Gyx
  Bij[0][1] += ry*s_vec.x * A + ry*dAdr*rx - s_vec.y * dBdr* rx;
  //Gyy
  Bij[1][1] += ry*s_vec.y * A + ry*dAdr*ry - s_vec.y * dBdr* ry;
  Bij[1][1] += rdotm*A;
  //Gyz
  Bij[2][1] += ry*s_vec.z * A + ry*dAdr*rz - s_vec.y * dBdr* rz;

  //Gzx
  Bij[0][2] += rz*s_vec.x * A + rz*dAdr*rx - s_vec.z * dBdr* rx;
  //Gzy
  Bij[1][2] += rz*s_vec.y * A + rz*dAdr*ry - s_vec.z * dBdr* ry;
  //Gzz
  Bij[2][2] += rz*s_vec.z * A + rz*dAdr*rz - s_vec.z * dBdr* rz;
  Bij[2][2] += rdotm*A;
}


//curve device function.
#define DEF_POLY4_FUNC_naive(_FUNCNAME,ARG1,ARG2,ARG3,ARG4,ARG5)\
template <typename _Ty>\
__device__ _Ty _FUNCNAME(_Ty x){\
return x*(x*(x*(x*(ARG1)+(ARG2))+(ARG3))+(ARG4))+(ARG5);\
}

DEF_POLY4_FUNC_naive(c1p0, 9.97813616438174e-09, -2.97897856524718e-08, 2.38918644566813e-09, 4.53199938857366e-08, 2.44617454752747e-11);
DEF_POLY4_FUNC_naive(c1p1, -2.76473728643294e-09, 2.86975546540539e-08, -9.94582836806651e-08, 1.25129924573675e-07, -2.37010166723652e-08);
DEF_POLY4_FUNC_naive(c1p2, -1.09679990621465e-09, 9.77055663264614e-09, -2.54781238661150e-08, 2.65020634884934e-09, 5.00787562417835e-08);
DEF_POLY4_FUNC_naive(c1p3, 3.79927162333632e-10, -6.26368404962679e-09, 3.94760528277489e-08, -1.13580541622200e-07, 1.27491333574323e-07);

DEF_POLY4_FUNC_naive(c2p0, 6.69550479838731e-08, -1.61753307173877e-07, 1.68213714992711e-08, 1.34558143036838e-07, 1.10976027980100e-10);
DEF_POLY4_FUNC_naive(c2p1, -3.08460139955194e-08, 2.29192245602275e-07, -5.88399621128587e-07, 5.61170054591844e-07, -1.14421132829680e-07);
DEF_POLY4_FUNC_naive(c2p2, 3.50477408060213e-09, -5.25956271895141e-08, 2.78876509535747e-07, -6.24199554212217e-07, 4.91807818904985e-07);
DEF_POLY4_FUNC_naive(c2p3, 7.33485346367840e-10, -9.58855788627803e-09, 4.37085309763591e-08, -7.48004594092261e-08, 2.34161209605651e-08);



__device__ void evaluate_curve1_2(float & C1, float & C2, const float & q) {
  if (0 < q&&q <= 1) {
    C1 = c1p0(q);
    C2 = c2p0(q);
  }
  else if (1 < q&&q <= 2) {
    C1 = c1p1(q);
    C2 = c2p1(q);
  }
  else if (2 < q&&q <= 3) {
    C1 = c1p2(q);
    C2 = c2p2(q);
  }
  else if (3 < q&&q <= 4) {
    C1 = c1p3(q);
    C2 = c2p3(q);
  }
}


__device__ void get_near_field_force_tensor(
  glm::mat3 & Bij,
  const glm::vec3 & r_vec,
  const glm::vec3 & s_vec,
  const float & q,
  const float & h) {
  //curve 1 and 2 are the two non zero component in Tijk tensor
  //Tijk(q,h): i force dir, J source dir, k target dir
  //gives the force when source and target are on z axis.
  //6 component have curve1, Tzzz have curve 2.
  float curve1_q_hm4;
  float curve2_q_hm4;
  if (q == 0) {
    //when two particles overlap, no force
    return;
  }
  evaluate_curve1_2(curve1_q_hm4, curve2_q_hm4, q);
  curve1_q_hm4 *= 1 / (h*h*h*h);
  curve2_q_hm4 *= 1 / (h*h*h*h);
  //the non-zero component in the unrotated tensor is complete
  //time to calculate the rotated tensor.
  glm::vec3 zeta = glm::normalize(r_vec);
  glm::vec3 y{ 0, 1, 0 };
  glm::vec3 z{ 0, 0, 1 };
  glm::vec3 eta = glm::cross(zeta, z);
  //in case eta=0 because zeta is on z direction
  //make default the y direction.
  eta.y += 1e-10;
  eta = glm::normalize(eta);

  glm::vec3 xi = glm::cross(eta, zeta);

  //constructor using three column vectors
  glm::mat3 M{ xi, eta, zeta };
  //note in GLM matrix, M[i] gives the ith column!

  //calculation formula:
  //Bal=M^T_(i a)T_(i j k)(q,h)M^T_(j p)(m_p)~M^T_(k l)
  //a is the force direction in world, l is the target dipole direction in world

  //source in the rotated axis:
  //M^float *s_vec
  glm::vec3 s_xez = s_vec*M;

  //U=T_(i j k)(q,h)M^T_(j p)(m_p)^~
  //is the center matrix
  //this constructor gives first column vector, then second
  glm::mat3 U{
    s_xez.z*curve1_q_hm4,0                   ,s_xez.x*curve1_q_hm4,
    0,                   s_xez.z*curve1_q_hm4,s_xez.y*curve1_q_hm4,
    s_xez.x*curve1_q_hm4,s_xez.y*curve1_q_hm4,s_xez.z*curve2_q_hm4 };

  ////U1,U2,U3 row vector
  //glm::vec3 U1{ s_xez.z*C1,0                   ,s_xez.x*C1 };
  //glm::vec3 U2{ 0,                   s_xez.z*C1,s_xez.y*C1 };
  //glm::vec3 U3{ s_xez.x*C1,s_xez.y*C1,s_xez.z*C2 };
  //
  ////U*M^float
  //U1 = U1*M;
  //U2 = U2*M;
  //U3 = U3*M;

  Bij += M *U*glm::transpose(M);
}

//Fi=Bijmj
//mj is the magnetic dipole at this point
//Bij is the internal force tensor at this point

__device__ void get_force_Tensor(
  glm::mat3 & Bij,
  const glm::vec3 & r_vec,
  const glm::vec3 & s_vec,
  const float & h) {
  float r = glm::l2Norm(r_vec);
  float q = r * (1 / h);
  if (q > 4) {
    //far field force tensor
    get_far_field_force_tensor(Bij,
                               r_vec,
                               s_vec,
                               q,
                               h);
  }
  else {
    //debug, only calculate far-field interaction.
    //return;
    get_near_field_force_tensor(Bij,
                                r_vec,
                                s_vec,
                                q,
                                h);
  }

}

//function cuda_ext_fitted_force_tensor_kernel
//give the force tensor at target places.

__global__ void
cuda_ext_fitted_force_tensor_kernel(
  const float * _out_x, const float * _out_y, const float * _out_z,
  float * _out_dhextxdx, float * _out_dhextydx, float * _out_dhextzdx,
  float * _out_dhextxdy, float * _out_dhextydy, float * _out_dhextzdy,
  float * _out_dhextxdz, float * _out_dhextydz, float * _out_dhextzdz,
  const float * _in_x, const float * _in_y, const float * _in_z,
  const float * in_mx, const float * in_my, const float * in_mz,
  const size_t np_target,
  const size_t np_source,
  const  float h) {
  //maximum load size
  const int batch_np = 512;

  //gradient of Hferro contributed from other particles at this particle position
  glm::mat3 Bij{ 0 };

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int i_source = 0;
  int load_offset = 0;

  int idx_g = 0;
  //prepare space for shared memory
  __shared__ float s_x[batch_np];
  __shared__ float s_y[batch_np];
  __shared__ float s_z[batch_np];
  __shared__ float s_in_mx[batch_np];
  __shared__ float s_in_my[batch_np];
  __shared__ float s_in_mz[batch_np];

  //number of batches of data
  //if (np%batch_np != 0) total_batches++;

  for (load_offset = 0; load_offset < np_source; load_offset += batch_np) {

    i_source = np_source - load_offset < batch_np ? np_source - load_offset : batch_np;
    i_source--;
    //load a bunch of particles into the shared memory
    idx_g = load_offset + threadIdx.x;
    if ((threadIdx.x < batch_np) && (idx_g < np_source)) {
      s_x[threadIdx.x] = _in_x[idx_g];
      s_y[threadIdx.x] = _in_y[idx_g];
      s_z[threadIdx.x] = _in_z[idx_g];
      s_in_mx[threadIdx.x] = in_mx[idx_g];
      s_in_my[threadIdx.x] = in_my[idx_g];
      s_in_mz[threadIdx.x] = in_mz[idx_g];
    }
    __syncthreads();

    //add the influence of those particles in the shared memory to the current point
    if (i < np_target) {
      for (; i_source >= 0; i_source--) {
        //process i_source
        glm::vec3 r_vec{
          _out_x[i] - s_x[i_source],
          _out_y[i] - s_y[i_source],
          _out_z[i] - s_z[i_source] };
        get_force_Tensor(Bij,
                         r_vec,
                         glm::vec3{ s_in_mx[i_source],s_in_my[i_source] ,s_in_mz[i_source] },
                         h);

      }
    }
    __syncthreads();
  }


  //write the current out_hx,out_hy,out_hz to the global memory
  if (i < np_target) {
    _out_dhextxdx[i] = Bij[0][0];
    _out_dhextxdy[i] = Bij[0][1];
    _out_dhextxdz[i] = Bij[0][2];
    _out_dhextydx[i] = Bij[1][0];
    _out_dhextydy[i] = Bij[1][1];
    _out_dhextydz[i] = Bij[1][2];
    _out_dhextzdx[i] = Bij[2][0];
    _out_dhextzdy[i] = Bij[2][1];
    _out_dhextzdz[i] = Bij[2][2];
  }
}



__global__ void
fitted_force_ofM_kernel(
  const float * x, const float * y, const float * z,
  float * out_fx, float * out_fy, float * out_fz,
  const float * dhextxdx, const float * dhextydx, const float * dhextzdx,
  const float * dhextxdy, const float * dhextydy, const float * dhextzdy,
  const float * dhextxdz, const float * dhextydz, const float * dhextzdz,
  const float * in_mx, const float * in_my, const float * in_mz,
  const size_t np, const  float h) {
  //maximum load size
  const int batch_np = 512;

  //gradient of Hferro contributed from other particles at this particle position
  glm::mat3 Bij{ 0 };

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int i_source = 0;
  int load_offset = 0;

  int idx_g = 0;
  //prepare space for shared memory
  __shared__ float s_x[batch_np];
  __shared__ float s_y[batch_np];
  __shared__ float s_z[batch_np];
  __shared__ float s_in_mx[batch_np];
  __shared__ float s_in_my[batch_np];
  __shared__ float s_in_mz[batch_np];

  //number of batches of data
  //if (np%batch_np != 0) total_batches++;

  for (load_offset = 0; load_offset < np; load_offset += batch_np) {

    i_source = np - load_offset < batch_np ? np - load_offset : batch_np;
    i_source--;
    //load a bunch of particles into the shared memory
    idx_g = load_offset + threadIdx.x;
    if ((threadIdx.x < batch_np) && (idx_g < np)) {
      s_x[threadIdx.x] = x[idx_g];
      s_y[threadIdx.x] = y[idx_g];
      s_z[threadIdx.x] = z[idx_g];
      s_in_mx[threadIdx.x] = in_mx[idx_g];
      s_in_my[threadIdx.x] = in_my[idx_g];
      s_in_mz[threadIdx.x] = in_mz[idx_g];
    }
    __syncthreads();

    //add the influence of those particles in the shared memory to the current point
    if (i < np) {
      for (; i_source >= 0; i_source--) {
        //process i_source
        glm::vec3 r_vec{
          x[i] - s_x[i_source],
          y[i] - s_y[i_source],
          z[i] - s_z[i_source] };
        get_force_Tensor(Bij,
                         r_vec,
                         glm::vec3{ s_in_mx[i_source],s_in_my[i_source] ,s_in_mz[i_source] },
                         h);

      }
    }
    __syncthreads();
  }


  //write the current out_hx,out_hy,out_hz to the global memory
  if (i < np) {
    glm::vec3 M{ in_mx[i],in_my[i],in_mz[i] };

    const double mu_0 = 1.25663706e-6;
    glm::vec3 F_ferro{ 0 };
    F_ferro = Bij*M;
    F_ferro.x += mu_0*(
      M.x * dhextxdx[i] +
      M.y * dhextxdy[i] +
      M.z * dhextxdz[i]);
    F_ferro.y += mu_0*(
      M.x * dhextydx[i] +
      M.y * dhextydy[i] +
      M.z * dhextydz[i]);
    F_ferro.z += mu_0*(
      M.x * dhextzdx[i] +
      M.y * dhextzdy[i] +
      M.z * dhextzdz[i]);
    out_fx[i] = F_ferro.x;
    out_fy[i] = F_ferro.y;
    out_fz[i] = F_ferro.z;
  }
}


cuda_magnet_cgOptimizer::cuda_magnet_cgOptimizer(const typename cuda_magnet_cgOptimizer::CtorParam & _in_arg) :
  xyz(_in_arg.num_particles),
  Hferro(_in_arg.num_particles),
  Hext(_in_arg.num_particles),
  M(_in_arg.num_particles),
  total_B_over_mu0(_in_arg.num_particles),
  dHextdx(_in_arg.num_particles),
  dHextdy(_in_arg.num_particles),
  dHextdz(_in_arg.num_particles),
  magnetic_bodyforce(_in_arg.num_particles),
  step_dir(_in_arg.num_particles),
  old_step_dir(_in_arg.num_particles),
  gradient(_in_arg.num_particles),
  old_gradient(_in_arg.num_particles),
  residual(_in_arg.num_particles)
{
  chi = _in_arg.chi;
  np = _in_arg.num_particles;
  tolerance = 0.1f;
  max_iteration = 80;
  i_step = 0;
  total_error.reserve(max_iteration);
  volume = _in_arg.volume;
  h = _in_arg.h;

  xyz.import_data(_in_arg.target_pos);
  //set external magnet sources
  //Hext_source_pos.reserve(_in_arg.ext_source_pos.size());
  Hext_source_pos.import_data(_in_arg.ext_source_pos);
  //Hext_source_M.reserve(_in_arg.ext_source_M.size());
  Hext_source_M.import_data(_in_arg.ext_source_M);


  //fast fast multiple field.
  typedef fmmtl::DipoleBSpherical::source_type source_type;
  typedef fmmtl::DipoleBSpherical::charge_type charge_type;
  typedef fmmtl::DipoleBSpherical::_K_real_ _K_real_;
  // Init the FMM Kernel and options
  FMMOptions opts{ h,true };// = get_options(argc, argv);
  crit_size = 384;
  opts.ncrit = crit_size;

  //set source and target positions for magnetizable body.
  fmm_object_pos.reserve(_in_arg.target_pos.size());
  for (int i = 0; i < _in_arg.target_pos.size(); i++) {
    fmm_object_pos.push_back(source_type{
      _K_real_(_in_arg.target_pos[i].x),
      _K_real_(_in_arg.target_pos[i].y),
      _K_real_(_in_arg.target_pos[i].z)
    });
  }
  //object field fmm solver, results is hferro+M
  //{kernel, target, source}
  fmm_field_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>>(fmmtl::DipoleBSpherical{ 6 }, fmm_object_pos, fmm_object_pos);
  fmm_field_evaluator->kernel().m_h = h;
  fmm_field_evaluator->set_options(opts);

  //hferro evaluator,
  fmm_hferro_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleFieldSpherical>>(fmmtl::DipoleFieldSpherical{ 6 }, fmm_object_pos, fmm_object_pos);
  fmm_hferro_evaluator->kernel().m_h = h;
  fmm_hferro_evaluator->set_options(opts);
  //object force fmm solver
  fmm_hessian_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleHessianSpherical>>(fmmtl::DipoleHessianSpherical{ 6 }, fmm_object_pos, fmm_object_pos);
  fmm_hessian_evaluator->kernel().m_h = h;
  fmm_hessian_evaluator->set_options(opts);
  //set source positions and charge for external magnetic field.
  fmm_hext_pos.reserve(_in_arg.ext_source_pos.size());
  fmm_hext_charge.reserve(_in_arg.ext_source_M.size());

  for (int i = 0; i < _in_arg.ext_source_M.size(); i++) {
    fmm_hext_pos.push_back(source_type{
      _K_real_(_in_arg.ext_source_pos[i].x),
      _K_real_(_in_arg.ext_source_pos[i].y),
      _K_real_(_in_arg.ext_source_pos[i].z) });
    fmm_hext_charge.push_back(charge_type{
      _K_real_(_in_arg.ext_source_M[i].x),
      _K_real_(_in_arg.ext_source_M[i].y),
      _K_real_(_in_arg.ext_source_M[i].z) });
  }

  FMMOptions opts_ext{ _in_arg.ext_mag_h ,true };// = get_options(argc, argv);
  opts_ext.ncrit = crit_size;
  //external field fmm solver
  //{kernel, target, source}
  fmm_hext_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>>(fmmtl::DipoleBSpherical{ 6 }, fmm_object_pos, fmm_hext_pos);
  fmm_hext_evaluator->kernel().m_h = _in_arg.ext_mag_h;
  fmm_hext_evaluator->set_options(opts_ext);
  //external force fmm solver
  fmm_gradhext_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleHessianSpherical>>(fmmtl::DipoleHessianSpherical{ 6 }, fmm_object_pos, fmm_hext_pos);
  fmm_gradhext_evaluator->kernel().m_h = _in_arg.ext_mag_h;
  fmm_gradhext_evaluator->set_options(opts_ext);
  const_hext.x = _in_arg.const_ext_field[0];
  const_hext.y = _in_arg.const_ext_field[1];
  const_hext.z = _in_arg.const_ext_field[2];
  naive = false;
}


void cuda_magnet_cgOptimizer::Resize(size_t _np)
{
  //when the imported particle size is different from the one in this solver
  if (_np == np) {
    return;
  }
  size_t old_np = np;
  np = _np;
  //save the old initial data
  cuda_abstract_vector<float> old_init_data;
  old_init_data = total_B_over_mu0;


  xyz.reserve(np);
  Hferro.reserve(np);
  Hext.reserve(np);
  M.reserve(np);
  total_B_over_mu0.reserve(np);
  dHextdx.reserve(np);
  dHextdy.reserve(np);
  dHextdz.reserve(np);
  magnetic_bodyforce.reserve(np);

  step_dir.reserve(np);
  old_step_dir.reserve(np);
  gradient.reserve(np);
  old_gradient.reserve(np);
  residual.reserve(np);

  size_t copy_count = np > old_np ? old_np : np;

  //save some of the initial data

  cudaMemcpy(total_B_over_mu0.dev_x, old_init_data.dev_x, copy_count * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(total_B_over_mu0.dev_y, old_init_data.dev_y, copy_count * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(total_B_over_mu0.dev_z, old_init_data.dev_z, copy_count * sizeof(float), cudaMemcpyDeviceToDevice);

}


void cuda_magnet_cgOptimizer::setHext(const typename cuda_magnet_cgOptimizer::setHextParams& _in_arg)
{
  Hext_source_pos.import_data(_in_arg.ext_source_pos);
  //Hext_source_M.reserve(_in_arg.ext_source_M.size());
  Hext_source_M.import_data(_in_arg.ext_source_M);
  typedef fmmtl::DipoleBSpherical::source_type source_type;
  typedef fmmtl::DipoleBSpherical::charge_type charge_type;
  typedef fmmtl::DipoleBSpherical::_K_real_ _K_real_;

  if (fmm_hext_charge.size() == _in_arg.ext_source_pos.size()) {
    for (int i = 0; i < fmm_hext_charge.size(); i++) {
      fmm_hext_charge[i] = charge_type{
        _K_real_(_in_arg.ext_source_M[i].x),
        _K_real_(_in_arg.ext_source_M[i].y),
        _K_real_(_in_arg.ext_source_M[i].z) };
      fmm_hext_pos[i] = source_type{
        _K_real_(_in_arg.ext_source_pos[i].x),
        _K_real_(_in_arg.ext_source_pos[i].y),
        _K_real_(_in_arg.ext_source_pos[i].z) };
    }
  }
  else {
    fmm_hext_pos.clear();
    fmm_hext_pos.reserve(_in_arg.ext_source_pos.size());
    fmm_hext_charge.clear();
    fmm_hext_charge.reserve(_in_arg.ext_source_pos.size());
    for (int i = 0; i < _in_arg.ext_source_M.size(); i++) {
      fmm_hext_pos.push_back(source_type{
        _K_real_(_in_arg.ext_source_pos[i].x),
        _K_real_(_in_arg.ext_source_pos[i].y),
        _K_real_(_in_arg.ext_source_pos[i].z) });
      fmm_hext_charge.push_back(charge_type{
        _K_real_(_in_arg.ext_source_M[i].x),
        _K_real_(_in_arg.ext_source_M[i].y),
        _K_real_(_in_arg.ext_source_M[i].z) });
    }
  }


  //when the position of external magnetic sources is changed, the fmm solver need to updated
  fmm_gradhext_evaluator.reset();
  fmm_hext_evaluator.reset();
  // Init the FMM Kernel and options
  FMMOptions opts{ _in_arg.ext_mag_h,true };// = get_options(argc, argv);
  opts.ncrit = crit_size;
  //update the fmm external field part solver.
  //{kernel, target, source}
  fmm_hext_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>>(fmmtl::DipoleBSpherical{ 6 }, fmm_object_pos, fmm_hext_pos);
  fmm_hext_evaluator->kernel().m_h = _in_arg.ext_mag_h;
  fmm_hext_evaluator->set_options(opts);
  fmm_gradhext_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleHessianSpherical>>(fmmtl::DipoleHessianSpherical{ 6 }, fmm_object_pos, fmm_hext_pos);
  fmm_gradhext_evaluator->kernel().m_h = _in_arg.ext_mag_h;
  fmm_gradhext_evaluator->set_options(opts);

  const_hext.x = _in_arg.const_ext_field[0];
  const_hext.y = _in_arg.const_ext_field[1];
  const_hext.z = _in_arg.const_ext_field[2];
}


void cuda_magnet_cgOptimizer::setTargetpos(const std::vector<float3>& _in)
{
  Resize(_in.size());

  xyz.import_data(_in);
  //position data is ready, fmmtl matrix can be initialized;
  int order = 6;
  typedef fmmtl::DipoleBSpherical::source_type source_type;
  typedef fmmtl::DipoleBSpherical::target_type target_type;

  if (fmm_object_pos.size() == _in.size()) {
    for (int i = 0; i < _in.size(); i++) {
      fmm_object_pos[i] = source_type{
        float(_in[i].x),
        float(_in[i].y),
        float(_in[i].z) };
    }

  }
  else {
    np = _in.size();
    fmm_object_pos.clear();
    fmm_object_pos.reserve(_in.size());
    for (int i = 0; i < _in.size(); i++) {
      fmm_object_pos.push_back(source_type{
        float(_in[i].x),
        float(_in[i].y),
        float(_in[i].z) });
    }
  }

  //when the position of target are changed, the fmm solver need to be updated.
  float ext_par_h = fmm_field_evaluator->kernel().m_h;
  fmm_field_evaluator.reset();
  fmm_hferro_evaluator.reset();
  fmm_hessian_evaluator.reset();
  fmm_hext_evaluator.reset();
  fmm_gradhext_evaluator.reset();

  //new object evaluator 
  //{kernel target, source}
  // Init the FMM Kernel and options
  FMMOptions opts{ h,true };// = get_options(argc, argv);
  opts.ncrit = crit_size;
  fmm_field_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>>(fmmtl::DipoleBSpherical{ order }, fmm_object_pos, fmm_object_pos);
  fmm_field_evaluator->kernel().m_h = h;
  fmm_field_evaluator->set_options(opts);

  fmm_hferro_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleFieldSpherical>>(fmmtl::DipoleFieldSpherical{ order }, fmm_object_pos, fmm_object_pos);
  fmm_hferro_evaluator->kernel().m_h = h;
  fmm_hferro_evaluator->set_options(opts);

  fmm_hessian_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleHessianSpherical>>(fmmtl::DipoleHessianSpherical{ order }, fmm_object_pos, fmm_object_pos);
  fmm_hessian_evaluator->kernel().m_h = h;
  fmm_hessian_evaluator->set_options(opts);

  FMMOptions opts_Ext{ ext_par_h,true };// = get_options(argc, argv);
  opts_Ext.ncrit = crit_size;
  //new hext and force evaluator
  fmm_hext_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>>(fmmtl::DipoleBSpherical{ order }, fmm_object_pos, fmm_hext_pos);
  fmm_hext_evaluator->kernel().m_h = ext_par_h;
  fmm_hext_evaluator->set_options(opts_Ext);
  fmm_gradhext_evaluator = std::make_shared<fmmtl::kernel_matrix<fmmtl::DipoleHessianSpherical>>(fmmtl::DipoleHessianSpherical{ order }, fmm_object_pos, fmm_hext_pos);
  fmm_gradhext_evaluator->kernel().m_h = ext_par_h;
  fmm_gradhext_evaluator->set_options(opts_Ext);
}


void cuda_magnet_cgOptimizer::solve(size_t niter)
{
  clear();

  auto stop_solving = start();
  if (stop_solving) {
    return;
  }
  for (int i = 0; i < niter && i < max_iteration; i++) {
    printf("iteration %d \n", i);
    auto stop_iteration = step_iteration();
    if (stop_iteration)
      break;
  }
}

void cuda_magnet_cgOptimizer::evaluate_fitted_near_foece()
{
  //bool naive = false;
  if (naive) {
    size_t threaddim = 512;
    size_t blockdim = np / threaddim;
    if (np % threaddim != 0) {
      blockdim++;
    }
    M = volume * chi / (chi + 1)*total_B_over_mu0;

    fitted_force_ofM_kernel << <blockdim, threaddim >> > (
      xyz.dev_x, xyz.dev_y, xyz.dev_z,
      magnetic_bodyforce.dev_x, magnetic_bodyforce.dev_y, magnetic_bodyforce.dev_z,
      dHextdx.dev_x, dHextdx.dev_y, dHextdx.dev_z,
      dHextdy.dev_x, dHextdy.dev_y, dHextdy.dev_z,
      dHextdz.dev_x, dHextdz.dev_y, dHextdz.dev_z,
      M.dev_x, M.dev_y, M.dev_z,
      np, h);
  }//end naive
  else {
    //use fast multipole method to evaluate the forces
    //external force is already evaluated in the dhextx dhextdy, dhextdz
    //the only thing need to be done is to evaluate forces within objects.
    //prepare the data
    M = volume * chi / (chi + 1)*total_B_over_mu0;
    //prepare the charge
    typedef fmmtl::DipoleHessianSpherical::charge_type charge_type;
    typedef fmmtl::DipoleHessianSpherical::result_type result_type;
    typedef fmmtl::DipoleHessianSpherical::_K_real_ _K_real_;

    float * hostx, *hosty, *hostz;
    hostx = new float[M.np];
    hosty = new float[M.np];
    hostz = new float[M.np];
    M.export_data(hostx, hosty, hostz);
    if (fmm_object_charge.size() != M.np) {
      fmm_object_charge.clear();
      fmm_object_charge.reserve(M.np);
      for (int i = 0; i < M.np; i++) {
        fmm_object_charge.emplace_back(
          charge_type{
          _K_real_(hostx[i]),
          _K_real_(hosty[i]),
          _K_real_(hostz[i]) });
      }

    }
    else {
      for (int i = 0; i < M.np; i++) {
        fmm_object_charge[i] = charge_type{
          _K_real_(hostx[i]),
          _K_real_(hosty[i]),
          _K_real_(hostz[i]) };
      }
    }

    //charge is ready, time to get the result
    std::vector<result_type> fmmresult(M.np);
    fmmresult = (*fmm_hessian_evaluator) * fmm_object_charge;
    //fmmtl::direct(DipoleHessianKernel{h}, fmm_object_pos, fmm_object_charge, fmm_object_pos, fmmresult);
    //fmm result is ready, time to copy to the cuda abstract vector result;
    float * hxx, *hxy, *hxz;
    float * hyx, *hyy, *hyz;
    float * hzx, *hzy, *hzz;
    hxx = new float[M.np];
    hxy = new float[M.np];
    hxz = new float[M.np];
    hyx = new float[M.np];
    hyy = new float[M.np];
    hyz = new float[M.np];
    hzx = new float[M.np];
    hzy = new float[M.np];
    hzz = new float[M.np];
    dHextdx.export_data(hxx, hxy, hxz);
    dHextdy.export_data(hyx, hyy, hyz);
    dHextdz.export_data(hzx, hzy, hzz);

    for (int i = 0; i < M.np; i++) {
      glm::mat3 temp_hext_tensor{
        hxx[i],hxy[i],hxz[i],
        hyx[i],hyy[i],hyz[i],
        hzx[i],hzy[i],hzz[i] };
      glm::vec3 temp_M{ float(fmm_object_charge[i][0]),
        float(fmm_object_charge[i][1]),
        float(fmm_object_charge[i][2]) };
      glm::vec3 temp_force = (temp_hext_tensor + fmmresult[i])*temp_M;
      hostx[i] = float(temp_force.x);
      hosty[i] = float(temp_force.y);
      hostz[i] = float(temp_force.z);
    }
    magnetic_bodyforce.import_data(hostx, hosty, hostz);

    delete[] hxx;
    delete[] hxy;
    delete[] hxz;
    delete[] hyx;
    delete[] hyy;
    delete[] hyz;
    delete[] hzx;
    delete[] hzy;
    delete[] hzz;
    delete[] hostx;
    delete[] hosty;
    delete[] hostz;
  }//end !naive

}


void cuda_magnet_cgOptimizer::clear()
{
  total_error.clear();
  i_step = 0;
}



int cuda_magnet_cgOptimizer::start()
{
  i_step = 0;

  //calculate external magnetic fiield.
  eval_hext();

  //get the residual
  M = volume * chi / (chi + 1)*total_B_over_mu0;
  residual = G(M) + Hext - total_B_over_mu0;

  //store the current error
  auto curr_error = float(0.5)*residual.norm2();
  total_error.push_back(curr_error);
  printf("starting error: %.5e \n", curr_error);

  if (sqrt(curr_error / np) < tolerance) {
    //1 means stop iteration
    return 1;
  }

  //get the gradient D = (G_b\Gamma-I)
  gradient = D_transpose(residual);

  //first step as the negative gradient.
  step_dir = -gradient;

  //step
  float curr_stepsize = calc_alpha();
  total_B_over_mu0 = total_B_over_mu0 + (curr_stepsize*step_dir);

  return 0;
}


int cuda_magnet_cgOptimizer::step_iteration()
{

  //record old data
  old_gradient = gradient;
  old_step_dir = step_dir;

  //Standing at a new point
  M = volume*chi / (chi + 1)*total_B_over_mu0;
  residual = G(M) + Hext - total_B_over_mu0;

  float curr_error = float(0.5)*residual.norm2();
  //store the error
  total_error.push_back(curr_error);
  printf("Iteration %d: error: %.5e \n", i_step, curr_error);

  //check if criterion is made
  if (sqrt(curr_error / np) < tolerance) {
    //1 means stop iteration
    return 1;
  }
  //current error greater then last error
  if (curr_error > *(total_error.rbegin() + 1)) {
    //1 means stop iteration
    return 1;
  }
  i_step++;

  //calculate the gradient;

  gradient = D_transpose(residual);
  //calculate beta
  float curr_beta = 0;
  if (i_step % 10 == 0) {
    //reset conjugate gradient
    curr_beta = 0;
  }
  else {
    curr_beta = calc_beta();
    if (curr_beta < 0) {
      curr_beta = 0;
    }
  }

  //stepdirection
  step_dir = -gradient + curr_beta*old_step_dir;

  //step
  float curr_stepsize = calc_alpha();
  total_B_over_mu0 = total_B_over_mu0 + (curr_stepsize*step_dir);

  return 0;
}

//function eval_hext
//evaluate the external force and external force tensor
//input: external sources as particles, with positions and magnetization 
//additional constant field is also provided.

void cuda_magnet_cgOptimizer::eval_hext()
{
  //bool naive = 0;
  if (naive) {
    //evaluate the external magnetic field
    size_t threaddim = 512;
    size_t blockdim = np / threaddim;
    if (np % threaddim != 0) {
      blockdim++;
    }
    cudaError_t cudaStatus = cudaGetLastError();;
    cuda_ext_field_kernel << <blockdim, threaddim >> > (xyz.dev_x, xyz.dev_y, xyz.dev_z,
                                                        Hext.dev_x, Hext.dev_y, Hext.dev_z,
                                                        Hext_source_pos.dev_x, Hext_source_pos.dev_y, Hext_source_pos.dev_z,
                                                        Hext_source_M.dev_x, Hext_source_M.dev_y, Hext_source_M.dev_z,
                                                        const_hext,
                                                        np, Hext_source_M.np, h);
    cudaStatus = cudaGetLastError();

    //evaluate the force tensor of external magnetic field.
    cuda_ext_fitted_force_tensor_kernel << <blockdim, threaddim >> > (
      xyz.dev_x, xyz.dev_y, xyz.dev_z,
      dHextdx.dev_x, dHextdx.dev_y, dHextdx.dev_z,
      dHextdy.dev_x, dHextdy.dev_y, dHextdy.dev_z,
      dHextdz.dev_x, dHextdz.dev_y, dHextdz.dev_z,
      Hext_source_pos.dev_x, Hext_source_pos.dev_y, Hext_source_pos.dev_z,
      Hext_source_M.dev_x, Hext_source_M.dev_y, Hext_source_M.dev_z,
      np, Hext_source_M.np, h);
    cudaStatus = cudaGetLastError();
  }//end naive
  else {
    //use fast multipole method to calculate the external magnetic field 
    //and external magnetic field tensor.

    //the charge is ready, only do the multiplication here.
    typedef fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>::result_type field_result_type;
    std::vector<field_result_type> fmm_field_result = (*fmm_hext_evaluator)*fmm_hext_charge;

    typedef fmmtl::kernel_matrix<fmmtl::DipoleHessianSpherical>::result_type hessian_result_type;
    std::vector<hessian_result_type> fmm_hessian_result = (*fmm_gradhext_evaluator)*fmm_hext_charge;

    float *hextx, *hexty, *hextz;
    float * hxx, *hxy, *hxz;
    float * hyx, *hyy, *hyz;
    float * hzx, *hzy, *hzz;
    hextx = new float[np];
    hexty = new float[np];
    hextz = new float[np];
    hxx = new float[np];
    hxy = new float[np];
    hxz = new float[np];
    hyx = new float[np];
    hyy = new float[np];
    hyz = new float[np];
    hzx = new float[np];
    hzy = new float[np];
    hzz = new float[np];
    for (int i = 0; i < np; i++) {
      hextx[i] = fmm_field_result[i][0] + const_hext.x;
      hexty[i] = fmm_field_result[i][1] + const_hext.y;
      hextz[i] = fmm_field_result[i][2] + const_hext.z;

      hxx[i] = fmm_hessian_result[i][0][0];
      hyy[i] = fmm_hessian_result[i][1][1];
      hzz[i] = fmm_hessian_result[i][2][2];
      hxy[i] = fmm_hessian_result[i][0][1];
      hyx[i] = fmm_hessian_result[i][1][0];
      hxz[i] = fmm_hessian_result[i][0][2];
      hzx[i] = fmm_hessian_result[i][2][0];
      hyz[i] = fmm_hessian_result[i][1][2];
      hzy[i] = fmm_hessian_result[i][2][1];
    }

    Hext.import_data(hextx, hexty, hextz);
    dHextdx.import_data(hxx, hxy, hxz);
    dHextdy.import_data(hyx, hyy, hyz);
    dHextdz.import_data(hzx, hzy, hzz);
    delete[] hxx;
    delete[] hxy;
    delete[] hxz;
    delete[] hyx;
    delete[] hyy;
    delete[] hyz;
    delete[] hzx;
    delete[] hzy;
    delete[] hzz;
    delete[] hextx;
    delete[] hexty;
    delete[] hextz;
  }//end !naive
}//end evaluate external field.



cuda_abstract_vector<float> cuda_magnet_cgOptimizer::G(const cuda_abstract_vector<float>& in)
{
  cuda_abstract_vector<float> result{ np };
  //bool naive = false;
  if (naive) {
    size_t threaddim = 512;
    size_t blockdim = np / threaddim;
    if (np % threaddim != 0) {
      blockdim++;
    }
    cudaError_t cudaStatus;

    cuda_Apply_G << <blockdim, threaddim >> > (xyz.dev_x,
                                               xyz.dev_y,
                                               xyz.dev_z,
                                               result.dev_x,
                                               result.dev_y,
                                               result.dev_z,
                                               in.dev_x,
                                               in.dev_y,
                                               in.dev_z,
                                               np,
                                               h);
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaGetLastError();
    return result;
  }
  else {
    //below this line uses fast multipole method.
    //prepare the charge
    typedef fmmtl::DipoleBSpherical::charge_type charge_type;
    typedef fmmtl::DipoleBSpherical::result_type result_type;
    typedef fmmtl::DipoleBSpherical::_K_real_ _K_real_;

    float * hostx, *hosty, *hostz;
    hostx = new float[in.np];
    hosty = new float[in.np];
    hostz = new float[in.np];
    in.export_data(hostx, hosty, hostz);
    if (fmm_object_charge.size() != in.np) {
      fmm_object_charge.clear();
      fmm_object_charge.reserve(in.np);
      for (int i = 0; i < in.np; i++) {
        fmm_object_charge.emplace_back(
          charge_type{
          _K_real_(hostx[i]),
          _K_real_(hosty[i]),
          _K_real_(hostz[i]) });
      }

    }
    else {
      for (int i = 0; i < in.np; i++) {
        fmm_object_charge[i] = charge_type{
          _K_real_(hostx[i]),
          _K_real_(hosty[i]),
          _K_real_(hostz[i]) };
      }
    }


    //charge is ready, time to get the result
    std::vector<result_type> fmmresult = (*fmm_field_evaluator) * fmm_object_charge;
    //fmm result is ready, time to copy to the cuda abstract vector result;
    for (int i = 0; i < in.np; i++) {
      hostx[i] = float(fmmresult[i][0]);
      hosty[i] = float(fmmresult[i][1]);
      hostz[i] = float(fmmresult[i][2]);
    }
    result.import_data(hostx, hosty, hostz);
    delete[] hostx;
    delete[] hosty;
    delete[] hostz;
    return result;
  }
}



cuda_abstract_vector<float> cuda_magnet_cgOptimizer::D(const cuda_abstract_vector<float>& _in_B_over_mu0)
{
  return G(volume*(chi) / (1 + chi)*_in_B_over_mu0) - _in_B_over_mu0;
}


cuda_abstract_vector<float> cuda_magnet_cgOptimizer::D_transpose(const cuda_abstract_vector<float>& _in)
{
  return volume*(chi) / (1 + chi)*G(_in) - _in;
}


inline float cuda_magnet_cgOptimizer::calc_alpha()
{
  const float epsl = 1e-10;
  cuda_abstract_vector<float> temp;
  temp.reserve(np);

  temp = D(step_dir);

  auto normtemp = temp.dot(temp);
  //indicates convergence.
  if (normtemp < epsl) {
    return 0;
  }

  return -(temp.dot(residual) / (normtemp + epsl));
}


inline float cuda_magnet_cgOptimizer::calc_beta()
{
  if (i_step == 0) {
    return 0;
  }
  else {
    float numer = gradient.norm2();
    float denom = old_step_dir.dot(old_gradient - gradient);
    return -numer / (denom + 1e-10f);
  }
}


void cuda_magnet_cgOptimizer::export_data(const char * file_name)
{
  std::ofstream out;
  out.open(file_name);
  float * host_x = new float[np];
  float * host_y = new float[np];
  float * host_z = new float[np];
  float * host_Hferrox = new float[np];
  float * host_Hferroy = new float[np];
  float * host_Hferroz = new float[np];
  float * host_Hextx = new float[np];
  float * host_Hexty = new float[np];
  float * host_Hextz = new float[np];

  cudaMemcpy(host_x, xyz.dev_x, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_y, xyz.dev_y, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_z, xyz.dev_z, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hferrox, total_B_over_mu0.dev_x, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hferroy, total_B_over_mu0.dev_y, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hferroz, total_B_over_mu0.dev_z, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hextx, Hext.dev_x, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hexty, Hext.dev_y, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hextz, Hext.dev_z, sizeof(float)*np, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < np; i++) {
    float pos[3] = { host_x[i],host_y[i],host_z[i] };

    float vec[3] = { host_Hferrox[i],host_Hferroy[i] ,host_Hferroz[i] };

    float H_ext[3] = { host_Hextx[i],host_Hexty[i],host_Hextz[i] };

    out << i << " ";
    out << pos[0] << " " << pos[1] << " " << pos[2] << " ";
    out << vec[0] << " " << vec[1] << " " << vec[2] << " ";
    out << H_ext[0] << " " << H_ext[1] << " " << H_ext[2] << std::endl;
  }
  out.close();
  delete[] host_x;
  delete[] host_y;
  delete[] host_z;
  delete[] host_Hferrox;
  delete[] host_Hferroy;
  delete[] host_Hferroz;
  delete[] host_Hextx;
  delete[] host_Hexty;
  delete[] host_Hextz;
}

void cuda_magnet_cgOptimizer::export_force(const char *file_name)
{
  std::ofstream out;
  out.open(file_name);
  float * host_x = new float[np];
  float * host_y = new float[np];
  float * host_z = new float[np];
  float * host_forcex = new float[np];
  float * host_forcey = new float[np];
  float * host_forcez = new float[np];
  float * host_Hextx = new float[np];
  float * host_Hexty = new float[np];
  float * host_Hextz = new float[np];

  cudaMemcpy(host_x, xyz.dev_x, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_y, xyz.dev_y, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_z, xyz.dev_z, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_forcex, magnetic_bodyforce.dev_x, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_forcey, magnetic_bodyforce.dev_y, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_forcez, magnetic_bodyforce.dev_z, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hextx, Hext.dev_x, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hexty, Hext.dev_y, sizeof(float)*np, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_Hextz, Hext.dev_z, sizeof(float)*np, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < np; i++) {
    float pos[3] = { host_x[i],host_y[i],host_z[i] };

    float vec[3] = { host_forcex[i],host_forcey[i] ,host_forcez[i] };

    float H_ext[3] = { host_Hextx[i],host_Hexty[i],host_Hextz[i] };

    out << i << " ";
    out << pos[0] << " " << pos[1] << " " << pos[2] << " ";
    out << vec[0] << " " << vec[1] << " " << vec[2] << " ";
    out << H_ext[0] << " " << H_ext[1] << " " << H_ext[2] << std::endl;
  }
  out.close();
  delete[] host_x;
  delete[] host_y;
  delete[] host_z;
  delete[] host_forcex;
  delete[] host_forcey;
  delete[] host_forcez;
  delete[] host_Hextx;
  delete[] host_Hexty;
  delete[] host_Hextz;
}

}//end namespace Libo