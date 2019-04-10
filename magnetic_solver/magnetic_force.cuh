#pragma once
#ifndef LIBO_MAGNETIC_FORCE_SOLVER 
#define LIBO_MAGNETIC_FORCE_SOLVER


#include "cuda_abstract_vector.cuh"
#include "math_constants.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "vector_types.h"

//fast multipole includes
#include "fmmtl/config.hpp"
#include "kernel/DipoleBSpherical.hpp"
#include "kernel/DipoleFieldSpherical.hpp"
#include "kernel/DipoleHessianSpherical.hpp"
#include "fmmtl/KernelMatrix.hpp"
#include "fmmtl/Direct.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include <memory>


namespace Libo {


//cuda_magnet_cgOptimizer 
//provides an environment for nonlinear optimization using conjugate gradient
//Minimizing H_ferro=S(H_ferro+H_ext) in the least square sense
//dS/dH=K=dH/dm*dm/dh=G*dmdh;

class cuda_magnet_cgOptimizer {

public:
  struct CtorParam {
    CtorParam(size_t _num_particles, float _h, float _ext_mag_h, float _volume, float _chi, float* _const_ext_field, const  std::vector<float3> & _ext_source_pos, const std::vector<float3> & _ext_source_M, const std::vector<float3> &_target_pos) {
      num_particles = _num_particles;
      h = _h;
      ext_mag_h = _ext_mag_h;
      volume = _volume;
      chi = _chi;
      const_ext_field[0] = _const_ext_field[0];
      const_ext_field[1] = _const_ext_field[1];
      const_ext_field[2] = _const_ext_field[2];
      ext_source_pos = _ext_source_pos;
      ext_source_M = _ext_source_M;
      target_pos = _target_pos;
    }

    CtorParam(size_t _num_particles, float _h, float _ext_mag_h, float _volume, float _chi, float3 _const_ext_field, const  std::vector<float3> && _ext_source_pos, const std::vector<float3> && _ext_source_M, const std::vector<float3> &_target_pos) {
      num_particles = _num_particles;
      h = _h;
      ext_mag_h = _ext_mag_h;
      volume = _volume;
      chi = _chi;
      const_ext_field[0] = _const_ext_field.x;
      const_ext_field[1] = _const_ext_field.y;
      const_ext_field[2] = _const_ext_field.z;
      ext_source_pos = _ext_source_pos;
      ext_source_M = _ext_source_M;
      target_pos = _target_pos;
    }
    size_t num_particles;
    float h;
    float ext_mag_h;
    float volume;
    float chi;
    float const_ext_field[3];
    std::vector<float3> ext_source_pos;
    std::vector<float3> ext_source_M;
    std::vector<float3> target_pos;
  };

  struct setHextParams {
    float const_ext_field[3];
    std::vector<float3> ext_source_pos;
    std::vector<float3> ext_source_M;
    float ext_mag_h;
  };

  //constructors
  cuda_magnet_cgOptimizer(const typename cuda_magnet_cgOptimizer::CtorParam & _in_arg);
  //destructor
  ~cuda_magnet_cgOptimizer() {
  }

  //Methods
  void Resize(size_t _np);
  void setHext(const typename cuda_magnet_cgOptimizer::setHextParams &);
  void setTargetpos(const std::vector<float3> &);
  void solve(size_t);
  int  step_iteration();
  int start();
  void evaluate_fitted_near_foece();

  void clear();

  void eval_hext();

  float calc_alpha();
  float calc_beta();

  //G:Hferro+M(density)
  cuda_abstract_vector<float> G(const cuda_abstract_vector<float>&);

  cuda_abstract_vector<float> D_transpose(const cuda_abstract_vector<float>&);

  cuda_abstract_vector<float> D(const cuda_abstract_vector<float>&);
  void export_data(const char *);
  void export_force(const char *);
  //Data

  //optimization variables
  //index 0 is the initial state
  float tolerance;
  size_t max_iteration;
  size_t i_step;
  std::vector<float> total_error;
  std::vector<float> line_search_record;

  //Magnetic field variables
  size_t np;
  float chi;
  float h;

  //Use this density if M calculated as density
  float volume;

  //optimization variables
  cuda_abstract_vector<float> step_dir;
  cuda_abstract_vector<float> old_step_dir;
  cuda_abstract_vector<float> gradient;
  cuda_abstract_vector<float> old_gradient;
  cuda_abstract_vector<float> residual;

  //magnetic field part
  cuda_abstract_vector<float> xyz;
  cuda_abstract_vector<float> Hferro;
  cuda_abstract_vector<float> Hext;

  cuda_abstract_vector<float> M;
  cuda_abstract_vector<float> total_B_over_mu0;
  //external magnetic field as particles
  cuda_abstract_vector<float> Hext_source_pos;
  cuda_abstract_vector<float> Hext_source_M;
  float3 const_hext;
  //force part
  //gradient of the external field used for force calculation
  cuda_abstract_vector<float> dHextdx;
  cuda_abstract_vector<float> dHextdy;
  cuda_abstract_vector<float> dHextdz;

  cuda_abstract_vector<float> magnetic_bodyforce;


  //fast multipole solver
  bool naive;
  int crit_size;
  std::vector<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>::source_type> fmm_hext_pos;
  std::vector<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>::source_type> fmm_object_pos;
  std::vector<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>::charge_type> fmm_hext_charge;
  std::vector<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>::charge_type> fmm_object_charge;

  std::shared_ptr<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>> fmm_field_evaluator;
  std::shared_ptr<fmmtl::kernel_matrix<fmmtl::DipoleFieldSpherical>> fmm_hferro_evaluator;
  std::shared_ptr<fmmtl::kernel_matrix<fmmtl::DipoleBSpherical>> fmm_hext_evaluator;

  std::shared_ptr<fmmtl::kernel_matrix<fmmtl::DipoleHessianSpherical>> fmm_hessian_evaluator;
  std::shared_ptr<fmmtl::kernel_matrix<fmmtl::DipoleHessianSpherical>> fmm_gradhext_evaluator;
};


}//end namespace Libo
#endif //define LIBO_MAGNETIC_FORCE_SOLVER