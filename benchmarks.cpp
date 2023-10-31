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

// #include "fj_tool/fipp.h" 

int ITE = 50;
int nprocs, rank;
int nzero, dist;
float mb;

void readinputs( int rank, std::string filename, std::vector<int> &sendsarray, std::vector<int> &recvcounts);
template<typename T> T variance(const std::vector<T> &vec);
double calculate_variance(std::vector<int> &vec, int rank, std::vector<int> &diff);
void my_spreadout_nonblocking(char* sendbuf, int *sendcounts, int *sdispls, 
				MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, 
				MPI_Datatype recvtype, MPI_Comm comm);

void my_alltoallv_blocking(char* sendbuf, int *sendcounts, int *sdispls, 
				MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, 
				MPI_Datatype recvtype, MPI_Comm comm);

void my_sorting_nonblocking(char* sendbuf, int *sendcounts, int *sdispls, 
				MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, 
				MPI_Datatype recvtype, MPI_Comm comm);

int run(int loopcount, std::vector<int>& sendsarray, std::vector<int>& recvcounts);

void creat_randon_inputs(int nZero, int range, int maxValue, std::vector<int> &sendsarray, std::vector<int> &recvcounts);
void creat_normal_distribution_inputs(int maxValue, std::vector<int> &sendsarray, std::vector<int> &recvcounts);
void creat_Powerlaw_distribution_inputs(int maxValue, std::vector<int> &sendsarray, std::vector<int> &recvcounts);


// Main entry
int main(int argc, char **argv)
{
    // if (argc != 5) {
    //     printf("Usage: %s <Zero-ratio> <imbalance_degree> <dist> <max_value> \n", argv[0]);
    //     exit(-1);
    // }

    if (argc != 2) {
        printf("Usage: %s <filename> \n", argv[0]);
        exit(-1);
    }

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

	// float p = atof(argv[1]);
	// mb = atof(argv[2]);
	// dist = atoi(argv[3]);
	// int maxValue = atoi(argv[4]);

	// if (dist == 0){
	// 	nzero = p * nprocs; 
	// 	int range = mb * 100;
	//     creat_randon_inputs(nzero, range, maxValue, sendsarray, recvcounts);
	// }

	// if (dist == 1) {
	// 	creat_normal_distribution_inputs(maxValue, sendsarray, recvcounts);
	// }

	// // Power law distribution
	// if (dist == 2)
	// {
	// 	creat_Powerlaw_distribution_inputs(maxValue, sendsarray, recvcounts);
	// }

	std::vector<int> send_diff, recv_diff;
	double send_vari = calculate_variance(sendsarray, rank, send_diff);
	double send_stdDev = sqrt(send_vari);

	double recv_vari = calculate_variance(recvcounts, rank, recv_diff);
	double recv_stdDev = sqrt(recv_vari);

    std::cout <<  "INFO--" << rank << ", " << nzero << ", " << mb << ", " << send_stdDev << ", " << recv_stdDev << ", " << dist << std::endl; 
    // for (int i = 0; i < nprocs; i++) {
    // 	std::cout << rank << ", " << i << ", " << send_diff[i] << ", " << recv_diff[i] << std::endl;
    // }

 	// run(comm_mode, 20, sendsarray, recvcounts, 1); // warm-up
 	run(20, sendsarray, recvcounts);


    MPI_Finalize();
    return 0;
}


int run(int loopcount, std::vector<int>& sendsarray, std::vector<int>& recvcounts) {

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

    MPI_Barrier(MPI_COMM_WORLD);

    
    for (int t = 0; t < loopcount; t++) {
    	// fipp_start();   
    	double start = MPI_Wtime();
    	MPI_Alltoallv(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
    	double end = MPI_Wtime();
    	double comm_time_1 = (end - start);

    	double max_time = 0;
    	MPI_Allreduce(&comm_time_1, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    	if (max_time == comm_time_1)
	    	std::cout << "TIME-0, " << rank << ", " << comm_time_1 << ", " << send_tsize << ", " <<  recv_tsize << ", " << max_sendn << ", " << max_recvn << ", " << nzero << ", " << mb << ", " << dist << std::endl;
	    MPI_Barrier(MPI_COMM_WORLD);

    	// else if (comm_mode == 1) {
    	// 	my_spreadout_nonblocking(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
    	// }
    	// else if (comm_mode == 2) {
    	// 	my_alltoallv_blocking(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
    	// }
    	// else if (comm_mode == 3) {
	    start = MPI_Wtime();
    	my_sorting_nonblocking(sendbuf, sendsarray.data(), sdispls, MPI_CHAR, recvbuf, recvcounts.data(), rdispls, MPI_CHAR, MPI_COMM_WORLD);
    	end = MPI_Wtime();
    	double comm_time_2 = (end - start);

    	max_time = 0;
    	MPI_Allreduce(&comm_time_2, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    	if (max_time == comm_time_2)
	    	std::cout << "TIME-3, " << rank << ", " << comm_time_2 << ", " << send_tsize << ", " <<  recv_tsize << ", " << max_sendn << ", " << max_recvn << ", " << nzero << ", " << mb << std::endl;
	    MPI_Barrier(MPI_COMM_WORLD);

    	// }
    	// else {
    	// 	std::cout << "Unsupported Mode" << std::endl;
    	// 	return -1;
    	// }
    	// fipp_stop();

    	// if (warmup == 0) { 


	    // }

	    MPI_Barrier(MPI_COMM_WORLD);
    }

    free(sendbuf);
    free(recvbuf);

    MPI_Barrier(MPI_COMM_WORLD);

    return 0;
}


void my_alltoallv_blocking(char* sendbuf, int *sendcounts, int *sdispls, 
				MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, 
				MPI_Datatype recvtype, MPI_Comm comm) {

	for (int i = 0; i < nprocs; i++) {

		int recv_proc = (rank - i + nprocs) % nprocs; 
		int send_proc = (rank + i) % nprocs; 

		MPI_Sendrecv(&sendbuf[sdispls[send_proc]], sendcounts[send_proc], MPI_CHAR, 
			send_proc, 0, &recvbuf[rdispls[recv_proc]], recvcounts[recv_proc], 
			MPI_CHAR, recv_proc, 0, comm, MPI_STATUS_IGNORE);
	}
}

void my_spreadout_nonblocking(char* sendbuf, int *sendcounts, int *sdispls, 
				MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, 
				MPI_Datatype recvtype, MPI_Comm comm) {

    MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));
	for (int i = 0; i < nprocs; i++) {
		int src = (rank + i) % nprocs; // avoid always to reach first master node
		MPI_Irecv(&recvbuf[rdispls[src]], recvcounts[src], MPI_CHAR, src, 0, MPI_COMM_WORLD, &req[i]);
	}

	for (int i = 0; i < nprocs; i++) {
		int dst = (rank - i + nprocs) % nprocs;
		MPI_Isend(&sendbuf[sdispls[dst]], sendcounts[dst], MPI_CHAR, dst, 0, MPI_COMM_WORLD, &req[i+nprocs]);
	}

	MPI_Waitall(2*nprocs, req, stat);
	free(req);
	free(stat);
}

// template <typename T>
// std::vector<size_t> sort_indexes(const std::vector<T> &v) {

//   // initialize original index locations
//   std::vector<size_t> idx(v.size());
//   iota(idx.begin(), idx.end(), 0);

//   // sort indexes based on comparing values in v
//   // using std::stable_sort instead of std::sort
//   // to avoid unnecessary index re-orderings
//   // when v contains elements of equal values 
//   stable_sort(idx.begin(), idx.end(),
//        [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

//   return idx;
// }

void sort_indices(int* v, std::vector<std::pair<int, int> > &temp) {
	for (int i = 0; i < nprocs; i++)
    	temp.push_back(std::make_pair(v[i], i));
	std::sort(temp.begin(), temp.end()) ;
}

void my_sorting_nonblocking(char* sendbuf, int *sendcounts, int *sdispls, 
				MPI_Datatype sendtype, char *recvbuf, int *recvcounts, int *rdispls, 
				MPI_Datatype recvtype, MPI_Comm comm) {

    MPI_Request* req = (MPI_Request*)malloc(2*nprocs*sizeof(MPI_Request));
	MPI_Status* stat = (MPI_Status*)malloc(2*nprocs*sizeof(MPI_Status));

	std::vector< std::pair<int, int> > sendsort ;
	sort_indices(sendcounts, sendsort);

	std::vector< std::pair<int, int> > recvsort ;
	sort_indices(recvcounts, recvsort);

	// if (rank == 0){
	// 	for (int i = 0; i < nprocs; i++) {
	// 		std::cout << sendsort[i].first <<  " " << sendsort[i].second << " " << recvsort[i].first << " " << recvsort[i].second << std::endl;
	// 	}
	// }

	for (int i = 0; i < nprocs; i++) {
		/// 
		int src = sendsort[i].second;
		// int src = (rank + i) % nprocs; // avoid always to reach first master node
		MPI_Irecv(&recvbuf[rdispls[src]], recvcounts[src], MPI_CHAR, src, 0, MPI_COMM_WORLD, &req[i]);
	}

	for (int i = 0; i < nprocs; i++) {
		// int dst = (rank - i + nprocs) % nprocs;
		int dst = recvsort[i].second;
		MPI_Isend(&sendbuf[sdispls[dst]], sendcounts[dst], MPI_CHAR, dst, 0, MPI_COMM_WORLD, &req[i+nprocs]);
	}

	MPI_Waitall(2*nprocs, req, stat);
	free(req);
	free(stat);
}


std::string getNItem(std::string str, int k, std::string delim, int count) {
	int p = str.find(' ');
	std::string item = str.substr(0, str.find(' '));

	if (count == k) return item;
	
	str = str.substr(p + delim.length());
	count += 1;

	return getNItem(str, k, delim, count);
}

void creat_randon_inputs(int nZero, int range, int maxValue, std::vector<int> &sendsarray, std::vector<int> &recvcounts) {

    int random_offset = 100 - range;
	srand(time(NULL));
	// srand(rank);

	for (int i = 0; i < nZero; i++) {
		sendsarray.push_back(0);
	}

	for (int i = nZero; i < nprocs; i++) {
		int random = random_offset + rand() % range;
		int v = (maxValue * random) / 100;
		sendsarray.push_back(v);
	}

	// if (rank == 0) {
	// 	for (int i = 0; i < nprocs; i++) {
	// 		std::cout << sendsarray[i] << std::endl;
	// 	}
	// }

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(&sendsarray[0], &sendsarray[nprocs], std::default_random_engine(seed));

	recvcounts.resize(nprocs);
	MPI_Alltoall(sendsarray.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

	// if (rank == 0 ) {
	// 	for (int i = 0; i < nprocs; i++)
	// 		std::cout << sendsarray[i] << ", " << recvcounts[i] << std::endl;
	// }
}

void creat_normal_distribution_inputs(int maxValue, std::vector<int> &sendsarray, std::vector<int> &recvcounts) {


	std::default_random_engine generator;
	std::normal_distribution<double> distribution(nprocs/2, nprocs/3); // set mean and deviation

	while(true)
	{
		sendsarray.resize(nprocs);
		int p = int(distribution(generator));
		if (p >= 0 && p < nprocs) {
			if (++sendsarray[p] >= maxValue) break;
		}
	}

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(&sendsarray[0], &sendsarray[nprocs], std::default_random_engine(seed));

	recvcounts.resize(nprocs);
	MPI_Alltoall(sendsarray.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

}

void creat_Powerlaw_distribution_inputs(int maxValue, std::vector<int> &sendsarray, std::vector<int> &recvcounts) {
	double x = (double)maxValue;

	sendsarray.resize(nprocs);
	for (int i=0; i <nprocs; ++i) {
		sendsarray[i] = (int)x;
		x = x * 0.999;
	}

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(&sendsarray[0], &sendsarray[nprocs], std::default_random_engine(seed));

	recvcounts.resize(nprocs);
	MPI_Alltoall(sendsarray.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
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



