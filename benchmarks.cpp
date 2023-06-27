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

int ITE = 50;

void readinputs( int rank, std::string filename, std::vector<int> &sendsarray, std::vector<int> &recvcounts);
template<typename T> T variance(const std::vector<T> &vec);
double calculate_variance(std::vector<int> &vec, int rank, std::vector<int> &diff);


// Main entry
int main(int argc, char **argv)
{
    // Check the number of arguments
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        exit(-1);
    }

    int nprocs, rank;

    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &nprocs) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");

	// input
	std::string filename = argv[1];
	std::vector<int> sendsarray;
	std::vector<int> recvcounts;
	readinputs(rank, filename, sendsarray, recvcounts);

	std::vector<int> send_diff, recv_diff;
	double send_vari = calculate_variance(sendsarray, rank, send_diff);
	double send_stdDev = sqrt(send_vari);

	double recv_vari = calculate_variance(recvcounts, rank, recv_diff);
	double recv_stdDev = sqrt(recv_vari);

	int sdispls[nprocs], rdispls[nprocs];
	long send_tsize = 0, recv_tsize =0;
	int max_sendn = 0, max_recvn = 0;
	for (int i = 0; i < nprocs; i++) {
		sdispls[i] = send_tsize;
		rdispls[i] = recv_tsize;
		send_tsize += sendsarray[i];
		recv_tsize += recvcounts[i];
		if (sendsarray[i] > max_sendn) 
			max_sendn = sendsarray[i];
		if (recvcounts[i] > max_recvn) 
			max_recvn = recvcounts[i];
	}

	char *sendbuf = (char *)malloc(send_tsize);
    char *recvbuf = (char *)malloc(recv_tsize);

    for (int i = 0; i < send_tsize; i++)
    	sendbuf[i] = 'a' + rand() % 26;

    std::cout << rank << ", " << send_tsize << ", " << recv_tsize << ", " << send_stdDev << ", " << recv_stdDev << ", " << max_sendn << ", " << max_recvn << std::endl; 
    // for (int i = 0; i < nprocs; i++) {
    // 	std::cout << rank << ", " << i << ", " << send_diff[i] << ", " << recv_diff[i] << std::endl;
    // }

    MPI_Status *status;
    double times[ITE];
    for (int t = 0; t < ITE; t++) {

    	// double start = MPI_Wtime();
    	// MPI_Alltoallv(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
    	// double end = MPI_Wtime();
    	// times[t] = (end - start);

    	double start = MPI_Wtime();
    	MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
		MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));
		for (int i = 0; i < nprocs; i++) {
			int src = (rank + i) % nprocs; // avoid always to reach first master node
			MPI_Irecv(&recvbuf[rdispls[src]], recvcounts[src], MPI_CHAR, src, 0, MPI_COMM_WORLD, &req[i]);
		}

		for (int i = 0; i < nprocs; i++) {
			int dst = (rank - i + nprocs) % nprocs;
			MPI_Isend(&sendbuf[sdispls[dst]], sendsarray[dst], MPI_CHAR, dst, 0, MPI_COMM_WORLD, &req[i+nprocs]);
		}

		MPI_Waitall(2*nprocs, req, stat);
		free(req);
		free(stat);
		double end = MPI_Wtime();
		times[t] = (end - start);

    	// double recv_start = MPI_Wtime();
    	// for (int i = 0; i < nprocs; i++) {

    		// 
    		// MPI_Sendrecv(&sendbuf[sdispls[i]], int sendcount, MPI_Datatype sendtype,
            //      int dest, int sendtag,
            //      void &recvbuf[rdispls[i]], int recvcount, MPI_Datatype recvtype,
            //      int source, int recvtag, MPI_Comm comm, MPI_Status * status)

			// int srcp = (rank + i) % nprocs; // avoid always to reach first master node
			// MPI_Recv(&sendbuf[sdispls[i]], sendsarray[i], MPI_CHAR, (rank - 1), 0, MPI_COMM_WORLD, status);

			// // int dstp = (rank - i + nprocs) % nprocs;
			// MPI_Send(&recvbuf[rdispls[i]], recvcounts[i], MPI_CHAR, (rank + 1), 0, MPI_COMM_WORLD);
			
		// 	if (rank == 0)
		// 		std::cout << "start -- " << i << ", " << sendsarray[i] << ", " << recvcounts[i] << std::endl;

		// }
		// double recv_end = MPI_Wtime();
		// recv_times[t] = recv_end - recv_start;

		// double send_start = MPI_Wtime();
		// // for (int i = 0; i < nprocs; i++) {
		// // 	int dst = (rank - i + nprocs) % nprocs;
		// // 	MPI_Send(recvbuf, recvcounts[i], MPI_CHAR, dst, dst, MPI_COMM_WORLD);
		// // }
		// double send_end = MPI_Wtime();
		// send_times[t] = send_end - send_start;

    }

    double mean_time = 0;
    for (int i = 0; i < ITE; i++) {
    	mean_time += times[i] / ITE;

    	// MPI_Reduce(&times[i], &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    	// if (rank == 0)
    	// 	std::cout << i << " " << max_time << std::endl;
    }

    std::cout << "time, " << rank << ", " << mean_time << std::endl;

    // if (rank == 1){
	// 	// for (auto r: times)
	// 	// 	std::cout << r << std::endl;
	// }


    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}


std::string getNItem(std::string str, int k, std::string delim, int count) {
	int p = str.find(' ');
	std::string item = str.substr(0, str.find(' '));

	if (count == k) return item;
	
	str = str.substr(p + delim.length());
	count += 1;

	return getNItem(str, k, delim, count);
}


void readinputs( int rank, std::string filename, std::vector<int> &sendsarray, std::vector<int> &recvcounts) {

	std::ifstream file(filename);
	std::string str; 
	int count = 0;

    while (std::getline(file, str)) {

    	std::string item = getNItem(str, rank, " ", 0);
    	recvcounts.push_back(stol(item));

    	if (rank == count) {
    		std::stringstream ss(str); 
    		std::string number; 
    		while (ss >> number) sendsarray.push_back(stol(number));
    	}
        count++;
    } 
}


double calculate_variance(std::vector<int> &vec, int rank, std::vector<int> &diff) {
	size_t sz = vec.size();
	if (sz <= 1) return 0.0;

	int mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;

	double variance = 0;
	for (int i = 0; i < sz; i++) {
		int diffv = vec[i] - mean;
		diff.push_back(diffv);
		variance += pow(diffv, 2);
	}

	variance = variance / (sz - 1);

	return variance;
}



