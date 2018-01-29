#pragma once

#include "Definitions.h"
#include "Repository.h"
#include <vector>
#ifdef CUDA_BACKEND
#include <cuda_runtime.h>
#endif
#include <cassert>
#include <memory>

#define N_CONCURRENT_HALOS 2

/**
* @class AllToAllV
* Class holding the horizontal diffusion stencil for u and v
*/
class AllToAllV {
    DISALLOW_COPY_AND_ASSIGN(AllToAllV);

  public:
    AllToAllV();
    ~AllToAllV();

    /**
    * Method applying the u stencil
    */
    void Apply();

  private:
    const int commSize_;
    int numRanks_;
    int rankId_;

    Real* recBuff_;
    Real* sendBuff_;

#ifdef CUDA_BACKEND
    cudaStream_t kernelStream_;
#endif

    void fillRandom(SimpleStorage< Real >& storage);
    void generateFields(Repository& repository);
};
