#ifndef DMFT_MODEL__
#define DMFT_MODEL__
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <iostream>
#include <complex>
#include <cmath>
#include <alps/params.hpp>
#include "../shared_libs/band_structure.hpp"
#include "self_energy.hpp"
#include "tail_manager.hpp"
#include <tuple>

using namespace std;

class DMFTModel {
     /*
      * class having access to lattice dispersion and local
      * self energy, able to compute lattice quantities, and elementary
      * many-body quantities, e.g. electronic density.
      */
public:
     DMFTModel(boost::shared_ptr<Bandstructure> const &lattice_bs,
	       boost::shared_ptr<Selfenergy> const &sigma,
	       const alps::params& parms, std::complex<double> chemical_potential,
               bool compute_bubble, int world_rank);
     tuple<double, double> get_particle_density(double chemical_potential,
						bool compute_derivative);
     tuple<int, double, double, double> get_mu_from_density(double initial_mu);
     tuple<int, double, double> get_mu_from_density_bisec(double initial_mu,
							   double mu_increment);
     void display_occupation_matrix();
     void display_spin_current();
     double get_kinetic_energy();
     double get_potential_energy();
     void dump_k_resolved_occupation_matrices();
     void scatter_occ_matrices();
     void scatter_xcurrent_matrices();
     void scatter_ycurrent_matrices();
     int check_density_success(double cur_density);
     void compute_order_parameter();
     void get_spin_current();
     void dump_bubble_hdf5();
     void compute_local_bubble();
     void compute_lattice_bubble();
     void set_chemical_potential(std::complex<double> new_chemical_potential) {
          chemical_potential = new_chemical_potential;
     };
     virtual ~DMFTModel() {}
	
private:
     // hf-expansion coefficients of the Green's function
     // Cf expression in Gull thesis, B.23 and B.34
     // c_1 is k-independent
     // c_2 depends on k and on mu     
     Eigen::MatrixXcd c_1;
     Eigen::MatrixXcd c_2;
     vector<Eigen::MatrixXcd> world_local_gf;
     vector<vector<Eigen::MatrixXcd> > world_local_bubble;
     // dims for lattice bubble:  boson, q_index, nu_index,
     // array of orbital indexed matrices
     vector<vector<vector<Eigen::MatrixXcd> > > lattice_bubble;
     vector<vector<vector<Eigen::MatrixXcd> > > world_lattice_bubble;

     // Dedicated object for computation of higher-order tails
     boost::shared_ptr<TailManager> tail_manager;
     void reset_occupation_matrices(size_t orbital_size);
     void reset_current_matrices(size_t orbital_size);
     void compute_tail_contribution(int k_index, size_t orbital_size,
				    double chemical_potential, double beta);
     void compute_analytical_tail(double chemical_potential, int k_index, double beta);
     double compute_derivative_tail(size_t orbital_size, double beta);
     //Eigen::MatrixXcd get_full_greens_function(double chemical_potential);
     std::vector<Eigen::MatrixXcd> get_greens_function(
	  Eigen::Ref<Eigen::VectorXd> k_point, int boson_index);
     
     boost::shared_ptr<Bandstructure> lattice_bs_;
     boost::shared_ptr<Selfenergy> sigma_;
     int world_rank_;
     size_t n_sites;
     size_t per_site_orbital_size;
     size_t tot_orbital_size;
     std::size_t N_boson;
     std::size_t bubble_dim;
     double n_tolerance;
     double target_density;
     bool exact_tail;
     bool compute_spin_current;
     vector<Eigen::MatrixXcd> k_resolved_occupation_matrices;
     vector<Eigen::MatrixXcd> spin_current_matrix;
     vector<Eigen::MatrixXcd> world_spin_current_matrix;
     vector<Eigen::MatrixXcd> k_resolved_xcurrent_matrices;
     vector<Eigen::MatrixXcd> k_resolved_ycurrent_matrices;
     vector<Eigen::MatrixXcd> world_k_resolved_occupation_matrices;
     vector<Eigen::MatrixXcd> world_k_resolved_xcurrent_matrices;
     vector<Eigen::MatrixXcd> world_k_resolved_ycurrent_matrices;
     Eigen::MatrixXcd occupation_matrix;
     Eigen::MatrixXcd world_occupation_matrix;
     std::vector<Eigen::VectorXcd> order_parameters;
     std::vector<Eigen::VectorXcd> spin_current_components;
     double kinetic_energy;
     double potential_energy;
     std::complex<double> chemical_potential;
     bool compute_bubble;

     static const std::size_t max_iter_for_bounds;     
     static const std::size_t max_iter_for_bisec;
     static const std::size_t max_iter_for_newton;     
     static const double e_max;
     static const std::size_t output_precision;
     static const std::size_t phi_output_precision;
     static const std::size_t current_output_precision;
     static const std::size_t phi_dimension;
     static const std::size_t current_dimension;
     static const std::string k_resolved_occupation_dump_name;
     static const std::string bubble_hdf5_root;
};
  
#endif //DMFT_MODEL__
