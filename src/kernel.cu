#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__device__ __host__ int divup(int total, int divisor) {
    return (total - 1) / divisor + 1;
}

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

glm::vec3 *dev_posMixed; // position arr, arrange to align to grid cells
glm::vec3 *dev_vel1Mixed; // velocity arr, arranged to align to grid cells

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, numObjects * sizeof(int));
  checkCUDAError("cudaMalloc particleArrayIndices failed!");
  cudaMalloc((void**)&dev_particleGridIndices, numObjects * sizeof(int));
  checkCUDAError("cudaMalloc particleGridIndices failed!");
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAError("cudaMalloc gridCellStartIndices failed!");
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAError("cudaMAlloc gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_posMixed, numObjects * sizeof(glm::vec3));
  checkCUDAError("cudaMalloc posMixed failed!");
  cudaMalloc((void**)&dev_vel1Mixed, numObjects * sizeof(glm::vec3));
  checkCUDAError("cudaMalloc vel1Mixed failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  
    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    // Rule 2: boids try to stay a distance d away from each other
    // Rule 3: boids try to match the speed of surrounding boids

    glm::vec3 perceived_center = glm::vec3(0);
    glm::vec3 awayVec = glm::vec3(0);
    glm::vec3 perceived_velocity = glm::vec3(0);

    unsigned numNeighbors1 = 0;
    unsigned numNeighbors3 = 0;

    glm::vec3 selfPos = pos[iSelf];

    for (unsigned i = 0; i < N; ++i) {
        if (i == iSelf) {
            continue;
        }

        glm::vec3 currPos = pos[i];
        glm::vec3 currVel = vel[i];
        float distance = glm::distance(currPos, selfPos);

        if (distance < rule1Distance) {
            perceived_center += currPos;
            numNeighbors1 += 1;
        }
        if (distance < rule2Distance) {
            awayVec -= (currPos - selfPos);
        }
        if (distance < rule3Distance) {
            perceived_velocity += currVel;
            numNeighbors3 += 1;
        }
    }

    glm::vec3 rule1Contribution = glm::vec3(0);
    if (numNeighbors1 > 0) {
        perceived_center /= numNeighbors1;
        rule1Contribution = (perceived_center - selfPos);
    }
    if (numNeighbors3 > 0) {
        perceived_velocity /= numNeighbors3;
    }

    return rule1Contribution * rule1Scale + awayVec * rule2Scale + perceived_velocity * rule3Scale;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    glm::vec3 velChange = computeVelocityChange(N, idx, pos, vel1);
    glm::vec3 newVel = vel1[idx] + velChange;

  // Clamp the speed
    newVel = glm::clamp(newVel, glm::vec3(-maxSpeed), glm::vec3(maxSpeed));
  
  // Record the new velocity into vel2. Question: why NOT vel1?
    vel2[idx] = newVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    // Find index of the current boid
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Return if thread is out of range
    if (idx >= N) {
        return;
    }

    // Find the grid cell the current boid resides in
    glm::vec3 selfPos = pos[idx];
    glm::ivec3 gridCoords = glm::floor((selfPos - gridMin) * inverseCellWidth);
    
    assert(gridCoords.x > 0 && gridCoords.x < gridResolution);
    assert(gridCoords.y > 0 && gridCoords.y < gridResolution);
    assert(gridCoords.z > 0 && gridCoords.z < gridResolution);
    
    unsigned gridIdx = gridIndex3Dto1D(gridCoords.x, gridCoords.y, gridCoords.z, gridResolution);
    indices[idx] = idx;
    gridIndices[idx] = gridIdx;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
    
    // There is one thread per particle, but these indices don't represent any particle
    // They just represent an idx into the GridIndices array
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N - 1) {
        int currGridIdx = particleGridIndices[idx];
        int nextGridIdx = particleGridIndices[idx + 1];
        if (currGridIdx != nextGridIdx) {
            gridCellStartIndices[nextGridIdx] = idx + 1;
            gridCellEndIndices[currGridIdx] = idx;
        }
        if (idx == 0) {
            gridCellStartIndices[currGridIdx] = 0;
        }
        if (idx == N - 2) {
            gridCellEndIndices[nextGridIdx] = N - 1;
        }
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    glm::vec3 selfPos = pos[idx];

  // - Identify the grid cell that this particle is in
    glm::ivec3 gridCoords = glm::floor((selfPos - gridMin) * inverseCellWidth);

    assert(gridCoords.x > 0 && gridCoords.x < gridResolution);
    assert(gridCoords.y > 0 && gridCoords.y < gridResolution);
    assert(gridCoords.z > 0 && gridCoords.z < gridResolution);


  // - Identify which cells may contain neighbors. This isn't always 8.
    float temp = rule1Distance > rule2Distance ? rule1Distance : rule2Distance;
    float maxRadius = rule3Distance > temp ? rule3Distance : temp;
    
    glm::ivec3 gridCoordsMin = glm::floor((selfPos - gridMin - maxRadius * glm::vec3(1)) * inverseCellWidth);
    glm::ivec3 gridCoordsMax = glm::floor((selfPos - gridMin + maxRadius * glm::vec3(1)) * inverseCellWidth);

  // - For each cell, read the start/end indices in the boid pointer array.


    glm::vec3 perceived_center = glm::vec3(0);
    glm::vec3 awayVec = glm::vec3(0);
    glm::vec3 perceived_velocity = glm::vec3(0);

    unsigned numNeighbors1 = 0;
    unsigned numNeighbors3 = 0;


    for (int k = gridCoordsMin.z; k <= gridCoordsMax.z; ++k) {
        for (int j = gridCoordsMin.y; j <= gridCoordsMax.y; ++j) {
            for (int i = gridCoordsMin.x; i <= gridCoordsMax.x; ++i) {

                int gridIdx = gridIndex3Dto1D(i, j, k, gridResolution);
                if (gridIdx >= 0 && gridIdx < gridResolution * gridResolution * gridResolution) {

                    int startIdx = gridCellStartIndices[gridIdx];
                    int endIdx = gridCellEndIndices[gridIdx];
                    //assert(endIdx == -1 || startIdx <= endIdx);
                    //assert(startIdx <= N && startIdx >= 0);
                    //assert(endIdx < N && endIdx >= -1);
                    //for (int a = 0; a < N; ++a) {
                    //    int idxa = particleArrayIndices[a];
                    //    assert(idxa >= 0 && idxa < N);
                    //}
                    // - Access each boid in the cell and compute velocity change from
                    //   the boids rules, if this boid is within the neighborhood distance.
                    for (int l = startIdx; l <= endIdx; ++l) {
                        //assert(l >= 0 && l < N);
                        int l_idx = particleArrayIndices[l];
                        if (l_idx == idx) {
                            continue;
                        }
                        //assert(l_idx >= 0);
                        //assert(l_idx < N);
                        // TODO: Fix this.
                        if (l_idx < 0 || l_idx >= N) {
                            l_idx = 0;
                        }
                        glm::vec3 currPos = pos[l_idx];
                        glm::vec3 currVel = vel1[l_idx];
                        float distance = glm::distance(currPos, selfPos);

                        if (distance < rule1Distance) {
                            perceived_center += currPos;
                            numNeighbors1 += 1;
                        }
                        if (distance < rule2Distance) {
                            awayVec -= (currPos - selfPos);
                        }
                        if (distance < rule3Distance) {
                            perceived_velocity += currVel;
                            numNeighbors3 += 1;
                        }
                    }

                }
            }
        }
    }

    glm::vec3 rule1Contribution = glm::vec3(0);
    if (numNeighbors1 > 0) {
        perceived_center /= numNeighbors1;
        rule1Contribution = (perceived_center - selfPos);
    }
    if (numNeighbors3 > 0) {
        perceived_velocity /= numNeighbors3;
    }

    glm::vec3 velChange = rule1Contribution * rule1Scale + awayVec * rule2Scale + perceived_velocity * rule3Scale;
    glm::vec3 newVel = vel1[idx] + velChange;

  // - Clamp the speed change before putting the new speed in vel2
    newVel = glm::clamp(newVel, glm::vec3(-maxSpeed), glm::vec3(maxSpeed));

  // Record the new velocity into vel2
    vel2[idx] = newVel;

}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    glm::vec3 selfPos = pos[idx];

    // - Identify the grid cell that this particle is in
    glm::ivec3 gridCoords = glm::floor((selfPos - gridMin) * inverseCellWidth);

    // - Identify which cells may contain neighbors. This isn't always 8.
    float temp = rule1Distance > rule2Distance ? rule1Distance : rule2Distance;
    float maxRadius = rule3Distance > temp ? rule3Distance : temp;

    glm::ivec3 gridCoordsMin = glm::floor((selfPos - gridMin - maxRadius * glm::vec3(1)) * inverseCellWidth);
    glm::ivec3 gridCoordsMax = glm::floor((selfPos - gridMin + maxRadius * glm::vec3(1)) * inverseCellWidth);

    // - For each cell, read the start/end indices in the boid pointer array.
    glm::vec3 perceived_center = glm::vec3(0);
    glm::vec3 awayVec = glm::vec3(0);
    glm::vec3 perceived_velocity = glm::vec3(0);

    unsigned numNeighbors1 = 0;
    unsigned numNeighbors3 = 0;

    for (int k = gridCoordsMin.z; k <= gridCoordsMax.z; ++k) {
        for (int j = gridCoordsMin.y; j <= gridCoordsMax.y; ++j) {
            for (int i = gridCoordsMin.x; i <= gridCoordsMax.x; ++i) {

                int gridIdx = gridIndex3Dto1D(i, j, k, gridResolution);
                if (gridIdx >= 0 && gridIdx < gridResolution * gridResolution * gridResolution) {

                    int startIdx = gridCellStartIndices[gridIdx];
                    int endIdx = gridCellEndIndices[gridIdx];
                    // - Access each boid in the cell and compute velocity change from
                    //   the boids rules, if this boid is within the neighborhood distance.
                    for (int l = startIdx; l <= endIdx; ++l) {
                        // TODO, test this
                        //assert(l >= 0 && l < N);

                        if (l < 0 || l >= N) {
                            l = 0;
                        }
                        glm::vec3 currPos = pos[l];
                        glm::vec3 currVel = vel1[l];
                        float distance = glm::distance(currPos, selfPos);

                        if (distance < rule1Distance) {
                            perceived_center += currPos;
                            numNeighbors1 += 1;
                        }
                        if (distance < rule2Distance) {
                            awayVec -= (currPos - selfPos);
                        }
                        if (distance < rule3Distance) {
                            perceived_velocity += currVel;
                            numNeighbors3 += 1;
                        }
                    }

                }
            }
        }
    }

    glm::vec3 rule1Contribution = glm::vec3(0);
    if (numNeighbors1 > 0) {
        perceived_center /= numNeighbors1;
        rule1Contribution = (perceived_center - selfPos);
    }
    if (numNeighbors3 > 0) {
        perceived_velocity /= numNeighbors3;
    }

    glm::vec3 velChange = rule1Contribution * rule1Scale + awayVec * rule2Scale + perceived_velocity * rule3Scale;
    glm::vec3 newVel = vel1[idx] + velChange;

    // - Clamp the speed change before putting the new speed in vel2
    newVel = glm::clamp(newVel, glm::vec3(-maxSpeed), glm::vec3(maxSpeed));

    // Record the new velocity into vel2
    vel2[idx] = newVel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  
    kernUpdatePos<<<divup(numObjects, blockSize), blockSize>>>(numObjects, dt, dev_pos, dev_vel1);
    checkCUDAError("kernUpdatePos failed!");

    kernUpdateVelocityBruteForce<<<divup(numObjects, blockSize), blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
    checkCUDAError("kernUpdateVelocityBruteForce failed!");

  // TODO-1.2 ping-pong the velocity buffers
    glm::vec3 *temp;
    temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;

}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed
    kernUpdatePos <<<divup(numObjects, blockSize), blockSize>>>(numObjects, dt, dev_pos, dev_vel1);
    checkCUDAError("kernUpdatePos failed!");


    kernComputeIndices<<<divup(numObjects, blockSize), blockSize>>>(
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // Sort array indices by the grid cell they are each in
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
    checkCUDAErrorWithLine("A");
    cudaDeviceSynchronize();


    kernResetIntBuffer<<<divup(gridCellCount, blockSize), blockSize>>>(
        gridCellCount, dev_gridCellStartIndices, numObjects);

    kernResetIntBuffer<<<divup(gridCellCount, blockSize), blockSize>>>(
        gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd<<<divup(numObjects, blockSize), blockSize>>>(
        numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("B");
    cudaDeviceSynchronize();

    //int* beginIndices = new int[gridCellCount];
    //cudaMemcpy(beginIndices, dev_gridCellStartIndices, gridCellCount * sizeof(int), cudaMemcpyDeviceToHost);
    //printf("Arr of beginnings\n");
    //for (int i = 0; i < gridCellCount; ++i) {
    //    printf("(%i: %i)", i, beginIndices[i]);
    //}
    //printf("\n");

    //int* endIndices = new int[gridCellCount];
    //cudaMemcpy(endIndices, dev_gridCellEndIndices, gridCellCount * sizeof(int), cudaMemcpyDeviceToHost);
    //printf("Arr of ends\n");
    //for (int i = 0; i < gridCellCount; ++i) {
    //    printf("(%i: %i)", i, endIndices[i]);
    //}
    //printf("\n");

    //int* arrayIndices = new int[numObjects];
    //cudaMemcpy(arrayIndices, dev_particleArrayIndices, numObjects * sizeof(int), cudaMemcpyDeviceToHost);
    //printf("Arr of indicies\n");
    //for (int i = 0; i < numObjects; ++i) {
    //    printf("(%i: %i)", i, arrayIndices[i]);
    //}
    //printf("\n");

    kernUpdateVelNeighborSearchScattered<<<divup(numObjects, blockSize), blockSize>>>(
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("C");
    cudaDeviceSynchronize();
    // Ping-pong the velocity buffers
    glm::vec3* temp;
    temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
}

__global__ void kernMixPosAndVel(int N, int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel, glm::vec3* posOut, glm::vec3* velOut) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int newIdx = particleArrayIndices[idx];
        posOut[idx] = pos[newIdx];
        velOut[idx] = vel[newIdx];
    }
}

__global__ void kernUnmixVel(int N, int* particleArrayIndices, glm::vec3* velMixed, glm::vec3* velOut) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int newIdx = particleArrayIndices[idx];
        velOut[newIdx] = velMixed[idx];
    }
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.

    kernUpdatePos << <divup(numObjects, blockSize), blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
    checkCUDAError("kernUpdatePos failed!");


    kernComputeIndices << <divup(numObjects, blockSize), blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAError("kernComputeIndices failed!");

    // Sort array indices by the grid cell they are each in
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);
    cudaDeviceSynchronize();


    kernResetIntBuffer << <divup(gridCellCount, blockSize), blockSize >> > (
        gridCellCount, dev_gridCellStartIndices, numObjects);

    kernResetIntBuffer << <divup(gridCellCount, blockSize), blockSize >> > (
        gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <divup(numObjects, blockSize), blockSize >> > (
        numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    cudaDeviceSynchronize();

    kernMixPosAndVel << <divup(numObjects, blockSize), blockSize >> > (
        numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_posMixed, dev_vel1Mixed);
    checkCUDAError("mix failed!");
    kernUpdateVelNeighborSearchCoherent << <divup(numObjects, blockSize), blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_posMixed, dev_vel1Mixed, dev_vel1); //dev_vel1 will end up mixed
    cudaDeviceSynchronize();
    checkCUDAError("velUpdateCoherent failed!");
    kernUnmixVel << <divup(numObjects, blockSize), blockSize >> > (
        numObjects, dev_particleArrayIndices, dev_vel1, dev_vel2); //unmixes dev_vel1 and puts it into dev_vel2
    checkCUDAError("unmix failed!");
     // Ping-pong the velocity buffers
    glm::vec3* temp;
    temp = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = temp;
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_posMixed);
  cudaFree(dev_vel1Mixed);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");


  std::unique_ptr<glm::vec3[]>pos{ new glm::vec3[10] };
  std::unique_ptr<glm::vec3[]>vel{new glm::vec3[10]};
  std::unique_ptr<int[]>arrayIndices{new int[10]};

  pos[0] = glm::vec3(0);   vel[0] = glm::vec3(10);    arrayIndices[0] = 2;
  pos[1] = glm::vec3(1);   vel[1] = glm::vec3(11);    arrayIndices[1] = 8;
  pos[2] = glm::vec3(2);   vel[2] = glm::vec3(12);    arrayIndices[2] = 4;
  pos[3] = glm::vec3(3);   vel[3] = glm::vec3(13);    arrayIndices[3] = 9;
  pos[4] = glm::vec3(4);   vel[4] = glm::vec3(14);    arrayIndices[4] = 3;
  pos[5] = glm::vec3(5);   vel[5] = glm::vec3(15);    arrayIndices[5] = 7;
  pos[6] = glm::vec3(6);   vel[6] = glm::vec3(16);    arrayIndices[6] = 0;
  pos[7] = glm::vec3(7);   vel[7] = glm::vec3(17);    arrayIndices[7] = 1;
  pos[8] = glm::vec3(8);   vel[8] = glm::vec3(18);    arrayIndices[8] = 6;
  pos[9] = glm::vec3(9);   vel[9] = glm::vec3(19);    arrayIndices[9] = 5;

  int* dev_arrayIndices;
  glm::vec3 *dev_pos, *dev_vel, *dev_posOut, *dev_velOut;
  cudaMalloc((void**)&dev_pos, 10 * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");
  cudaMalloc((void**)&dev_vel, 10 * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel failed!");
  cudaMalloc((void**)&dev_arrayIndices, 10 * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_arrayIndices failed!");
  cudaMalloc((void**)&dev_posOut, 10 * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_posOut failed!");
  cudaMalloc((void**)&dev_velOut, 10 * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_velOut failed!");

  cudaMemcpy(dev_pos, pos.get(), 10 * sizeof(glm::vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_vel, vel.get(), 10 * sizeof(glm::vec3), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_arrayIndices, arrayIndices.get(), 10 * sizeof(int), cudaMemcpyHostToDevice);

  kernMixPosAndVel<<<1, 10>>>(10, dev_arrayIndices, dev_pos, dev_vel, dev_posOut, dev_velOut);

  cudaMemcpy(pos.get(), dev_posOut, 10 * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
  cudaMemcpy(vel.get(), dev_velOut, 10 * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  std::cout << "after mixing: " << std::endl;
  for (int i = 0; i < 10; i++) {
      std::cout << "  order: " << arrayIndices[i] << std::endl;
  }
  for (int i = 0; i < 10; i++) {
      std::cout << "  pos: " << pos[i].x;
      std::cout << " vel: " << vel[i].x << std::endl;
  }

  kernUnmixVel<<<1, 10>>>(10, dev_arrayIndices, dev_velOut, dev_vel);

  cudaMemcpy(vel.get(), dev_vel, 10 * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  std::cout << "after unmixing: " << std::endl;
  for (int i = 0; i < 10; i++) {
      std::cout << "  vel: " << vel[i].x << std::endl;
  }

  return;
}
