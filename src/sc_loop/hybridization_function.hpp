#pragma once

#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <iostream>
#include <complex>
#include <cmath>
#include <alps/params.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include "../shared_libs/band_structure.hpp"
#include "../shared_libs/self_energy.hpp"

using namespace std;

class HybFunction {
     /*
      * There is something tricky about this class in the multi-site
      * framework. One must keep in mind that it is a bridge between
      * the impurity-world including the self-energy, and the lattice world,
      * through lattice and local Green's function, *back* to the impurity
      * world when it delivers the hybridization function Delta. Because of this,
      * extra care must be taken when calculating the coefficients of the 
      * HF expansion, and one has to constantly monitor whether one
      * is operating in the lattice world -> all orbitals are accounted for
      * or in the impurity world -> only the relevant site's orbitals are accounted for.
      */
public:
     HybFunction(const alps::params &parms,
		 boost::shared_ptr<Bandstructure> const &lattice_bs,
		 boost::shared_ptr<Selfenergy> const &sigma,
		 complex<double> chemical_potential,
		 int world_rank, bool compute_bubble, bool verbose=false);
     
     void compute_hybridization_function(complex<double> mu);
     void display_asymptotics();
     void dump_delta();
     void dump_Gtau_for_HF();
     void dump_delta_for_matrix();
     void dump_delta_hdf5();
     void dump_bubble_hdf5();
     void compute_local_bubble();
     void compute_lattice_bubble();
     void dump_G0_hdf5(alps::hdf5::archive &h5_archive);
     void dump_G0_for_ctint_hdf5(alps::hdf5::archive &h5_archive);
     virtual ~HybFunction() {}

     static const string matsubara_bare_gf_dump_name;
     static const string mom1_dump_name;
     static const string mom2_dump_name;
     static const string imaginary_time_dump_name_for_hf;
     
private:
     bool compute_bubble;
     boost::shared_ptr<Bandstructure> lattice_bs_;
     boost::shared_ptr<Selfenergy> sigma_;
     Eigen::MatrixXcd world_bath_moment_1;
     Eigen::MatrixXcd world_bath_moment_2;
     vector<Eigen::MatrixXcd> hybridization_function;
     vector<Eigen::MatrixXcd> G0_function;
     vector<Eigen::MatrixXcd> bare_greens_function;
     vector<Eigen::MatrixXcd> no_shift_bare_greens_function;
     vector<Eigen::MatrixXcd> pure_no_shift_bare_greens_function;
     vector<Eigen::MatrixXcd> world_local_gf;
     vector<vector<Eigen::MatrixXcd> > world_local_bubble;
     // dims for lattice bubble:  boson, q_index, nu_index,
     // array of orbital indexed matrices
     vector<vector<vector<Eigen::MatrixXcd> > > lattice_bubble;
     vector<vector<vector<Eigen::MatrixXcd> > > world_lattice_bubble;
     Eigen::MatrixXcd mu_tilde;
     std::complex<double> chemical_potential;
     // The coefficients of the high-frequency expansion
     // of the Hybridization function, in power of 1 / (i omega_n)^k,
     // starting with k = 1.
     vector<Eigen::MatrixXcd> delta_tau;
     vector<Eigen::MatrixXcd> G_tau;
     vector<Eigen::MatrixXcd> hf_coeff;
     vector<Eigen::MatrixXcd> bare_g_hf_coeff;
     size_t n_sites;
     size_t per_site_orbital_size;
     size_t tot_orbital_size;
     size_t n_tau;
     size_t N_boson;
     size_t bubble_dim;
     int world_rank_;
     bool enforce_real;
     
     //void compute_delta_tau();
     void elementary_compute_delta_tau();
     void elementary_compute_G_tau();
     void compute_superior_orders(bool verbose=false);
     void compute_bare_g_superior_orders(bool verbose=false);
     std::vector<Eigen::MatrixXcd> get_greens_function(
	  Eigen::Ref<Eigen::VectorXd> k_point, int boson_index);
	  
     static const size_t tail_fit_length;
     static const size_t output_precision;
     static const size_t max_expansion_order;
     static const string matsubara_frequency_dump_name;
     static const string imaginary_time_dump_name;
     static const string imaginary_time_dump_name_for_matrix;     
     static const string imaginary_time_hdf5_root;
     static const string bubble_hdf5_root;
     static const string shift_dump_name;
     static const string hf_shift_dump_name;
     static const string shift_sq_dump_name;
};
