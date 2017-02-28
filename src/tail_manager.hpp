#ifndef TAIL_MANAGER__
#define TAIL_MANAGER__
#include <Eigen/Dense>
#include<vector>
#include <tuple>
#include<iostream>
#include <complex>
#include <cmath>
#include <alps/params.hpp>

using namespace std;

class TailManager {
     /*
      * A class dedicated to the calculation
      * of high-orger tails, for Green's function
      * or other quantities.
      */
public:
     TailManager(const Eigen::Ref<Eigen::MatrixXcd> sigma_0,
		 const Eigen::Ref<Eigen::MatrixXcd> sigma_1,
		 const Eigen::Ref<Eigen::VectorXcd> matsubara_frequencies,
		 int world_rank);
     void set_chemical_potential(double chemical_potential);
     void set_current_k(const Eigen::Ref<Eigen::MatrixXcd> k_hamiltonian);
     Eigen::MatrixXcd get_analytical_contribution(double beta);
     double get_squared_partial_sum(double lambda);
     
     virtual ~TailManager() {}

private:
     // Upper value of n, with omega_n = (2n + 1) pi / beta
     // such that omega_n is handled numerically.
     int n_max;
     int flavor_size;
     int full_size;
     int world_rank_;
     Eigen::MatrixXcd sigma_skeleton;
     Eigen::MatrixXcd mirror_hamiltonian;
     Eigen::MatrixXcd sigma_0_;
     Eigen::MatrixXcd sigma_1_;
     double mu_;
     Eigen::VectorXcd matsubara_frequencies_;
     static const double fermi_cutoff;     
};

#endif //TAIL_MANAGER__
