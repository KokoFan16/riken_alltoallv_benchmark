/************************************************************************************
 *  Name: Ke Fan                                                                    *                                                                *
 *  Project: RIKEN -- All-to-allv                                                   *                                              *
 *  Date: June 04, 2023                                                             *
 *                                                                                  *
 *                                                                                  *
 *  To Compile: mpicc benchmarks.cpp -o benchmarks                                  *
 *  To run: mpirun -n <number of processes> ./benchmarks <filename>                 *
 *                                                                                  *
 ************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <fstream>

#include <mpi.h>

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>


int ITE = 50;
int nprocs, rank;
int nzero, dist;
float mb;

int run(int loopcount, int warmup, int m);
void my_spreadout_nonblocking(char* sendbuf, int *sendcounts, int *sdispls,
				MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
				MPI_Datatype recvtype, MPI_Comm comm);


// Main entry
int main(int argc, char **argv) {

    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");

    if (argc < 3) {
    	std::cout << "Usage: mpirun -n <nprocs> " << argv[0] << " <loop-count> <is_sort> " << std::endl;
    	return -1;
    }

    int loopCount = atoi(argv[1]);
    int sorted = atoi(argv[2]);

 	run(20, 1, 0); // warm-up
 	run(loopCount, 0, sorted);


    MPI_Finalize();
    return 0;
}


int run(int loopcount, int warmup, int m) {


	int sendcounts[nprocs]; // the size of data each process send to other process
	memset(sendcounts, 0, nprocs*sizeof(int));
	int sdispls[nprocs];

	for (int i = 0; i < nprocs; i++) {
		int id = (rank - i + nprocs) % nprocs;
		int v = pow(4, id);
		if ( v > INT_MAX ) { v = INT_MAX; }
		sendcounts[i] = v;
	}

	if (m == 0) {
		// Random shuffling the sendcounts array
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(&sendcounts[0], &sendcounts[nprocs], std::default_random_engine(seed));
	}


	// Initial send offset array
	int soffset = 0;
	for (int i = 0; i < nprocs; ++i) {
		sdispls[i] = soffset;
		soffset += sendcounts[i];
	}

	int recvcounts[nprocs];
	MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
	int rdispls[nprocs];
	int roffset = 0;
	for (int i = 0; i < nprocs; ++i) {
		rdispls[i] = roffset;
		roffset += recvcounts[i];
	}

	long long* send_buffer = new long long[soffset];
	long long* recv_buffer = new long long[roffset];

	int index = 0;
	for (int i = 0; i < nprocs; i++) {
		for (int j = 0; j < sendcounts[i]; j++)
			send_buffer[index++] = i + rank * 10;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	for (int it=0; it < loopcount; it++) {
		double st = MPI_Wtime();
		my_spreadout_nonblocking((char*)send_buffer, sendcounts, sdispls, MPI_UNSIGNED_LONG_LONG, (char*)recv_buffer, recvcounts, rdispls, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
		double et = MPI_Wtime();
		double total_time = et - st;

		// check correctness
		int error = 0;
		for (int i=0; i < roffset; i++) {
			if ( (recv_buffer[i] % 10) != (rank % 10) ) { error++; }
		}
		if (rank == 0 && error > 0) {
			std::cout << "[spreadout] has errors" << std::endl;
			MPI_Abort(MPI_COMM_WORLD, -1);
		}

		if (warmup == 0) {
			double max_time = 0;
			MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

			if (total_time == max_time) {
				if (m == 0) { std::cout << "[RandomSP] " << nprocs << ", " << max_time << std::endl; }
				else { std::cout << "[SortSP] " << nprocs << ", " << max_time << std::endl; }
			}
		}
	}

	delete[] send_buffer;
	delete[] recv_buffer;


	return 0;
}


void my_spreadout_nonblocking(char* sendbuf, int *sendcounts, int *sdispls,
				MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls,
				MPI_Datatype recvtype, MPI_Comm comm) {

    MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));
	for (int i = 0; i < nprocs; i++) {
		int src = (rank - i + nprocs) % nprocs; // avoid always to reach first master node
		MPI_Irecv(&recvbuf[rdispls[src]], recvcounts[src], MPI_CHAR, src, 0, MPI_COMM_WORLD, &req[i]);
	}

	for (int i = 0; i < nprocs; i++) {
		int dst = (rank + i) % nprocs;
		MPI_Isend(&sendbuf[sdispls[dst]], sendcounts[dst], MPI_CHAR, dst, 0, MPI_COMM_WORLD, &req[i+nprocs]);

//		if (i == 2)
//			std::cout << "send: " << rank << ", " << dst << ", " << sendcounts[dst] << std::endl;
	}

	MPI_Waitall(2*nprocs, req, stat);
	free(req);
	free(stat);
}




