#include "AllToAllV.h"
#include <exception>

#ifdef CUDA_BACKEND
#include "Kernel.h"
#endif
#include "MPIHelper.h"
#include <iostream>
#include <random>

AllToAllV::AllToAllV()
    : commSize_(128*128)
{
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId_);

    recBuff_ = (Real*)malloc(sizeof(Real) * commSize_ * numRanks_);
    sendBuff_ = (Real*)malloc(sizeof(Real) * commSize_ * numRanks_);

}
AllToAllV::~AllToAllV() {
}

void AllToAllV::Apply() {

    int sendcounts[numRanks_], senddispl[numRanks_], recvcounts[numRanks_], recvdispl[numRanks_];
    for(size_t cnt = 0; cnt < numRanks_; ++cnt) {
       sendcounts[cnt] = commSize_;
       senddispl[cnt] = commSize_*cnt;
       recvcounts[cnt] = commSize_;
       recvdispl[cnt] = commSize_*cnt;
    }
for(int i=0; i < 100; ++i)
    MPI_Alltoallv(sendBuff_, sendcounts, senddispl, MPITYPE,  recBuff_, recvcounts, recvdispl, MPITYPE, MPI_COMM_WORLD); 

}

