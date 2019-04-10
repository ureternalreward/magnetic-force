#include "magnetic_force.cuh"
#define one
int main() {

  int nx = 2, ny = 2, nz = 2;
  float dx = 1;

  //source particles must not be empty.
#ifndef one
  //two particles, bug free
  auto ext_source_pos = std::vector<float3>{ float3{ 0.f,-100.f, 0.f },float3{ 0.f,-101.f, 0.f } };
  auto ext_source_M = std::vector<float3>{ float3{ 0,1,0 }, float3{ 0,1,0 } };
#else
  //one particle, singularity bug
  auto ext_source_pos = std::vector<float3>{ float3{ 0.f, 0, 0.f } };
  auto ext_source_M = std::vector<float3>{ float3{ 0,1,0 }};
#endif



  auto target_pos = std::vector<float3>{};
  auto updated_target_pos = target_pos;
  target_pos.reserve(nx*ny*nz);
  updated_target_pos.reserve(nx*ny*(nz - 1));
  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        target_pos.push_back(float3{ i*dx,j*dx,k*dx });
        if (k < nz - 1) {
          updated_target_pos.push_back(float3{ i*dx, j*dx, k*dx + 1 });
        }
      }
    }
  }

  size_t  num_particles = target_pos.size();
  float h = dx;
  float ext_mag_h = dx;
  float volume = dx*dx*dx;
  float chi = 1;
  float const_ext_field[3] = { 0,1,0 };

  Libo::cuda_magnet_cgOptimizer::CtorParam solver_ctor{ num_particles,h,ext_mag_h,volume,chi,const_ext_field,ext_source_pos,ext_source_M,target_pos };
  Libo::cuda_magnet_cgOptimizer a{ solver_ctor };
  a.solve(100);
  a.evaluate_fitted_near_foece();

  a.setTargetpos(updated_target_pos);
  Libo::cuda_magnet_cgOptimizer::setHextParams seth;
  seth.const_ext_field[0] = const_ext_field[0];
  seth.const_ext_field[1] = const_ext_field[1];
  seth.const_ext_field[2] = const_ext_field[2];
  seth.ext_source_M = ext_source_M;
  seth.ext_source_pos = ext_source_pos;
  seth.ext_mag_h = h;
  a.setHext(seth);
  printf("particle position updated\n");
  a.solve(100);
  a.evaluate_fitted_near_foece();
  a.export_data("temp.txt");
  a.export_force("force.txt");
}