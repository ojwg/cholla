#ifdef STAR_FORMATION

#include <cassert>
#include <math.h>
#include <vector>
#include <limits>
#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../grid/grid_enum.h"
#include "../io/io.h"
#include "../particles/new_particles.h"

#ifdef MPI_CHOLLA
#include "../mpi/mpi_routines.h"
#endif


namespace new_star_particles {
  int *d_recent_particles;
  int *d_init_values;
}


void new_star_particles::Initialize(Grid3D &G) 
{
  if ((G.H.dx != G.H.dy) || (G.H.dx != G.H.dz)) {
    chprintf("ERROR; star formation assumes cubic grid\n");
    exit(-1);
  }
  CHECK( cudaMalloc (&d_recent_particles, G.H.n_cells*sizeof(int)) );
  CHECK( cudaMalloc (&d_init_values, G.H.n_cells*sizeof(int)) );
  std::vector<part_int_t> h_init_values (G.H.n_cells, -1);
  CHECK( cudaMemcpy(d_init_values, h_init_values.data(), G.H.n_cells*sizeof(int), cudaMemcpyHostToDevice) );
}


__device__ void getLocationInGrid(int xid, int yid, int zid, int n_ghost,
                                  Real xMin, Real yMin, Real zMin, Real dx, 
                                  Real &x, Real &y, Real &z) {
    x = xMin + (xid - n_ghost + 0.5) * dx;
    y = yMin + (yid - n_ghost + 0.5) * dx;
    z = zMin + (zid - n_ghost + 0.5) * dx;
}


__host__ __device__ part_int_t generateParticleId(int index, int n_step, int procID, int n_proc) {
  #ifdef PARTICLES_LONG_INTS
    return 1.0 * procID/n_proc * LONG_MAX + 1000 * n_step + index;
  #else
    return 1.0 * procID/n_proc * INT_MAX + 1000 * n_step + index;
  #endif  
}


__device__ bool isRecent(Real t, Real age) 
{ //TODO rename age
  return t >= age && t - age <= MAX_RECENT_AGE;
}


// TODO: put in correct implementation
__device__ Real mass_rate(Real density, Real dV) {
  return 0.1 * density / sqrt( 3 * M_PI / (32 * GN * density) ) * dV;
}


__device__ void accretion(Real *dev_conserved, Real *Fx, Real *Fy, Real *Fz, Real &vx_p, Real &vy_p, Real &vz_p,
                          Real &mass_p, int n_cells, int n_p_fields, int id, Real dx, Real dt) {
  Real dV = dx * dx * dx; 

  Real* density    = dev_conserved;
  Real* momentum_x = &dev_conserved[n_cells*grid_enum::momentum_x];
  Real* momentum_y = &dev_conserved[n_cells*grid_enum::momentum_y];
  Real* momentum_z = &dev_conserved[n_cells*grid_enum::momentum_z];  
  Real* energy     = &dev_conserved[n_cells*grid_enum::Energy];
  Real* gas_energy = &dev_conserved[n_cells*grid_enum::GasEnergy];

  Real mass_transf = mass_rate(density[id], dV) * dt;

  // frac (accreted mass / mass of cell) will be how much momentum the hydro cell loses.
  Real frac = mass_transf / dV / dev_conserved[id];

  // add accreted quantities to particle
  vx_p    = (vx_p * mass_p + frac * momentum_x[id] * dV)/(mass_p + mass_transf);
  vy_p    = (vy_p * mass_p + frac * momentum_y[id] * dV)/(mass_p + mass_transf);
  vz_p    = (vz_p * mass_p + frac * momentum_z[id] * dV)/(mass_p + mass_transf);
  mass_p +=  mass_transf;

  // remove accreted quantities from hydro 
  density[id]    -= mass_transf / dV;
  momentum_x[id] *= 1 - frac; // hydro velocities don't change
  momentum_y[id] *= 1 - frac;
  momentum_z[id] *= 1 - frac;
  energy[id]     *= 1 - frac;
  gas_energy[id] *= 1 - frac; // process is isothermal 
}


/**
 * @brief  returns an array of clusters with age less than a certain amount (see isRecent() for actual value.)
 *         The array length is the same as the hydro grid, and the values are indices of the particle attributes arrays.
 *  
 * @return
 */
__global__ void FindRecentParticlesKernel(int *d_recent_particles, int n_local, int nx, int ny, int nz, int n_ghost,
  Real dx, Real xMin, Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real *pos_x, Real *pos_y, Real *pos_z,
  Real *age, Real t)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= n_local)         return;
  if (!isRecent(t, age[tid])) return;
  if ( (pos_x[tid] < xMin || pos_x[tid] >= xMax) ||
       (pos_y[tid] < yMin || pos_y[tid] >= yMax) ||
       (pos_z[tid] < zMin || pos_z[tid] >= zMax) ) return;
  
  int i = (int)floor((pos_x[tid] - xMin) / dx) + n_ghost;
  int j = (int)floor((pos_y[tid] - yMin) / dx) + n_ghost;
  int k = (int)floor((pos_z[tid] - zMin) / dx) + n_ghost;
  int indx = i + j*nx + k*nx*ny;
    
  d_recent_particles[indx] = tid;
}


__global__ void CountSinkLocationsKernel(Real *dev_conserved, int n_g_fields, int nx, int ny, int nz,
    int n_ghost, Real dx, Real *F_x, Real *F_y, Real *F_z, int *n_new) 
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = id / (nx*ny);
  int yid = (id - zid*nx*ny) / nx;
  int xid = id - zid*nx*ny - yid*nx;
  int imo = xid-1 + yid*nx + zid*nx*ny;
  int jmo = xid + (yid-1)*nx + zid*nx*ny;
  int kmo = xid + yid*nx + (zid-1)*nx*ny;
  int n_cells = nx*ny*nz;
  Real d_thr;  

  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 && zid < nz  - n_ghost) {
    if ((F_x[imo] > 0 && F_x[id] < 0) &&
        (F_y[jmo] > 0 && F_y[id] < 0) &&
        (F_z[kmo] > 0 && F_z[id] < 0)) {
          d_thr = L_P_DENSITY*dev_conserved[grid_enum::GasEnergy*n_cells + id]/dev_conserved[id]/dx/dx;
          if (dev_conserved[id] > d_thr) {
            atomicAdd(n_new, 1);
          }
    }
  }

}


/**
 * @brief Looks for cells where mass flow is converging and densities are over the Larson-Penston threshold. If 
 * provided with a list of young clusters, will transfer mass and momentum to those if the locations match.  
 * Otherwise creates new cluster particles and transfers mass/momentum there.  Note that these new particles are 
 * not added to existing particle attribute arrays until after this function returns.
 * 
 */
__global__ void AccretionKernel(int * d_recent_particles, Real *buffer, Real *dev_conserved,
                                int n_g_fields, int nx, int ny, int nz, Real xMin, Real yMin, Real zMin, 
                                int n_cells, int n_ghost, int n_p_fields, Real dx, Real *F_x, Real *F_y,
                                Real *F_z, Real *pos_x, Real *pos_y, Real *pos_z, Real *vel_x, Real *vel_y, 
                                Real *vel_z, Real *mass, part_int_t *pid, Real *age, int n_step, Real t, Real dt,
                                int *n_new, int procID, int n_proc)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = id / (nx*ny);
  int yid = (id - zid*nx*ny) / nx;
  int xid = id - zid*nx*ny - yid*nx;
  int imo = xid-1 + yid*nx + zid*nx*ny;
  int jmo = xid + (yid-1)*nx + zid*nx*ny;
  int kmo = xid + yid*nx + (zid-1)*nx*ny;
  Real d_thr;
  Real x, y, z;

  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 && zid < nz  - n_ghost) {
    //if (dev_conserved[id] > d_thr     && 
    if ((F_x[imo] > 0 && F_x[id] < 0) && (F_y[jmo] > 0 && F_y[id] < 0) && (F_z[kmo] > 0 && F_z[id] < 0)) {
      d_thr = L_P_DENSITY*dev_conserved[grid_enum::GasEnergy*n_cells + id]/dev_conserved[id]/dx/dx;
      if (dev_conserved[id] > d_thr) {
        if (d_recent_particles != nullptr && d_recent_particles[id] > -1) {
              // over threshold case AND existing accreting cluster.
              printf(">>> THRESHOLD AND YOUNG CLUSTER [%d] at iteration %d, rank %d, pos (%d, %d, %d), dens: %.4e > %.4e\n", 
                          d_recent_particles[id], n_step, procID, xid, yid, zid, dev_conserved[id], d_thr);
              accretion(dev_conserved, F_x, F_y, F_z, vel_x[d_recent_particles[id]], vel_y[d_recent_particles[id]],
                        vel_z[d_recent_particles[id]], mass[d_recent_particles[id]],  n_cells, n_p_fields, id, dx, dt);
        } else {
              // we're over threshold and creating a new sink..
              // get the x, y, z location of the sink.
              getLocationInGrid(xid, yid, zid, n_ghost, xMin, yMin, zMin, dx, x, y, z);

              int index = atomicAdd(n_new, 1);  //index used for particle data in the buffer array.

              buffer[index * n_p_fields + 0] = x;
              buffer[index * n_p_fields + 1] = y;
              buffer[index * n_p_fields + 2] = z;
              buffer[index * n_p_fields + 3] = 0; // initialize vx to zero
              buffer[index * n_p_fields + 4] = 0; // initialize vy to zero
              buffer[index * n_p_fields + 5] = 0; // initialize vz to zero
              buffer[index * n_p_fields + 6] = 0; // initialize mass to zero
              buffer[index * n_p_fields + 7] = generateParticleId(index, n_step, procID, n_proc);
              buffer[index * n_p_fields + 8] = t; //TODO use enum instead of number

              accretion(dev_conserved, F_x, F_y, F_z, buffer[index * n_p_fields + 3], buffer[index * n_p_fields + 4],
                    buffer[index * n_p_fields + 5], buffer[index * n_p_fields + 6],  n_cells, n_p_fields, id, dx, dt);

              printf(">>> THRESHOLD AND NEW SINK at iteration %d, index %d, rank %d, pos (%.3f, %.3f, %.3f), dens: %.4e > %.4e\n",
                          n_step, index, procID, x, y, z, dev_conserved[id]*DENSITY_UNIT, d_thr*DENSITY_UNIT);
              // set the age to the time t
        }
      } else if (d_recent_particles != nullptr && d_recent_particles[id] > -1) {  //FIXME: does this make sense?  Can accretion happen if density < thrhld?
            //  For sure not over the threshold density, but ongoing accretion may be possible.
            printf(">>> found a young cluster [%d] at iteration %d, pos (%d, %d, %d), dens: %.4e > %.4e\n", 
            d_recent_particles[id], n_step, xid, yid, zid, dev_conserved[id], d_thr);
      }
    }
  }
}


void new_star_particles::FormStarParticles(Grid3D& G) {
  int n_cells = G.H.nx*G.H.ny*G.H.nz;
  assert(n_cells == G.H.n_cells);
  int ngrid = (n_cells + TPB - 1) / TPB;

  int rank{0}, n_ranks{1};
  #ifdef MPI_CHOLLA
    rank = procID;
    n_ranks = nproc;
  #endif

  // how many sinks are there?
  int *d_n_new, h_n_new;
  CHECK(cudaMalloc(&d_n_new, sizeof(int)));
  CHECK(cudaMemset(d_n_new, 0, sizeof(int)));
  // Note that this may be a "premature optimization" to not have to find all recent particles whether or not there are sinks.
  hipLaunchKernelGGL(CountSinkLocationsKernel, ngrid, TPB, 0, 0, G.C.device, G.H.n_fields, G.H.nx, G.H.ny, G.H.nz, 
                     G.H.n_ghost, G.H.dx, F_x, F_y, F_z, d_n_new);
  CHECK(cudaMemcpy(&h_n_new, d_n_new, sizeof(int), cudaMemcpyDeviceToHost));
  
  if (h_n_new == 0) {
    return;
  }

  printf("found %d new sink(s) at rank %d\n", h_n_new, rank);
  
  // fetch a list of recently created particles, in case the sink locations correspond to ongoing accretion.
  int *d_recent_particles {nullptr};

  if (G.Particles.n_local > 0) {
    int ngrid_p =  (G.Particles.n_local + TPB - 1) / TPB;

    std::vector<part_int_t> h_init_values (G.H.n_cells, -1);
    CHECK( cudaMalloc (&d_recent_particles, G.H.n_cells*sizeof(int)) );
    CHECK( cudaMemcpy(d_recent_particles, h_init_values.data(), G.H.n_cells*sizeof(int), cudaMemcpyHostToDevice) );

    // find recent particles that may still be accreting
    hipLaunchKernelGGL(FindRecentParticlesKernel, ngrid_p, TPB, 0, 0, d_recent_particles, G.Particles.n_local, G.H.nx,
                G.H.ny, G.H.nz, G.H.n_ghost, G.H.dx, G.Particles.G.xMin, G.Particles.G.yMin, G.Particles.G.zMin,
                G.Particles.G.xMax, G.Particles.G.yMax, G.Particles.G.zMax, G.Particles.pos_x_dev,
                G.Particles.pos_y_dev, G.Particles.pos_z_dev, G.Particles.age_dev, G.H.t);
    CudaCheckError();
    //cudaDeviceSynchronize(); //probably don't need this since the cudaMemset will wait until the kernel finishes.
  }

  // d_n_new will now be the count of new particles to be created.
  CHECK(cudaMemset(d_n_new, 0, sizeof(int)));

  // Remove conserved quantities from hydro.  If accretion is onto existing particles, update those particles.  
  // Otherwise the accreted data is packaged into a buffer that's sent over to MPI-related code.
  hipLaunchKernelGGL(AccretionKernel, ngrid, TPB, 0, 0, d_recent_particles, G.Particles.G.recv_buffer_x0_d, 
    G.C.device, G.H.n_fields, G.H.nx, G.H.ny, G.H.nz, G.H.xblocal, G.H.yblocal, G.H.zblocal, G.H.n_cells, G.H.n_ghost,
    N_DATA_PER_PARTICLE_TRANSFER, G.H.dx, F_x, F_y, F_z,  G.Particles.pos_x_dev, G.Particles.pos_y_dev,
    G.Particles.pos_z_dev, G.Particles.vel_x_dev, G.Particles.vel_y_dev, G.Particles.vel_z_dev, G.Particles.mass_dev,
    G.Particles.partIDs_dev, G.Particles.age_dev, G.H.n_step, G.H.t, G.H.dt, d_n_new, rank, n_ranks);

  CHECK(cudaMemcpy(&h_n_new, d_n_new, sizeof(int), cudaMemcpyDeviceToHost));

  if (h_n_new > 0) { // create the brand new particles.
    printf("creating %d new particles at %d\n", h_n_new, rank);
    G.Particles.Copy_Transfer_Particles_from_Buffer_GPU(h_n_new, G.Particles.G.recv_buffer_x0_d);
  }
  
}


#endif // STAR_FORMATION
