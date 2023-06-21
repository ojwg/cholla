#if ! ( defined(PARTICLE_AGE) && defined(PARTICLES_GPU) && defined(STAR_FORMATION) && defined(DE) )
  #error PARTICLE_AGE / PARTICLE_GPU / STAR_FORMATION / DE must be set
#endif

#pragma once

#include <math.h>

#include "../global/global.h" 
#include "../grid/grid3D.h"

#define L_P_GAMMA 1.666666
#define L_P_DENSITY  8.86/M_PI/GN*(L_P_GAMMA-1)*L_P_GAMMA
// MAX_RECENT_AGE is roughly age at first SNe
#define MAX_RECENT_AGE 4000


namespace new_star_particles {

extern int *d_recent_particles;
extern int *d_init_values;
/**
 * @brief creates star cluster particles if the ISM conditions are right, meaning:
 *   - density is above the Larson Penston threshold.
 *   - mass flux is everywhere into the cell at the boundaries.
 * 
 */
void FormStarParticles(Grid3D&);


/**
 * @brief checks that the grid we're using is cubic (i.e. dx==dy==dz)
 * 
 */
void Initialize(Grid3D&);

}
