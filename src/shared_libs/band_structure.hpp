#pragma once

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <alps/params.hpp>
#include <tuple>
#include <boost/multi_array.hpp>
#include <boost/range/algorithm.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>

using namespace std;

class Bandstructure {
     /*
      * class providing interface for a bandstructure
      */
public:
  
     Bandstructure(const alps::params& parms, int world_rank, bool verbose=false);
     void check_flavor_dim_consistency(const alps::params& parms, int dimension);
     int get_lattice_size();
     int get_orbital_size();	
     double get_weight_sum(int k_min, int k_max);
     double get_weight(int k_index);
     Eigen::MatrixXcd get_local_hoppings();
     Eigen::MatrixXcd get_V_matrix(int direction_index=1);
     Eigen::MatrixXcd get_epsilon_bar();
     Eigen::MatrixXcd get_epsilon_squared_bar();
     void compute_bare_dos(double chemical_potential);
     void dump_bare_dos();
     int get_world_size();
     int get_n_points_per_proc();
     int get_real_n_points();
     void display_hoppings();
     void dump_hamilt(const alps::params& parms, double chemical_potential);
     Eigen::MatrixXcd get_k_basis_matrix(Eigen::Ref<Eigen::VectorXd> k_point);
     Eigen::VectorXd get_k_point(int k_index) {return proc_k_lattice_[k_index];};
     std::vector<double> get_world_k_point(int k_index) {return k_lattice_[k_index];};
     Eigen::VectorXd get_q_point(int q_index) {return secondary_q_lattice_[q_index];};
     Eigen::VectorXd get_k_plus_q_point(int k_index, int q_index);
     int get_nb_points_for_bseq() {return secondary_q_lattice_.size();};
     int get_nb_world_k_points() {return k_lattice_.size();};
     // void dump_crystal_field(const alps::params& parms,
     // 			     double chemical_potential);
     virtual ~Bandstructure() {}

     /*
      * This is public so dmft_model can access it. Maybe
      * good to think about an architecture that allows the structure
      * to remain protected. But that implies the passing around 
      * of Eigen arrays -- still under investigation.
      */
     std::vector<Eigen::MatrixXcd> dispersion_;
  
protected:
     void init_world_containers(int n_points);
     vector<Eigen::MatrixXcd> generate_band_from_hoppings(bool verbose,
							  Eigen::Ref<Eigen::VectorXd> weights,
							  double unique_weight);
     void read_hoppings(const alps::params& parms, bool verbose=false);
     std::vector<Eigen::MatrixXcd> read_dispersion(const alps::params& parms,
						   Eigen::Ref<Eigen::VectorXd> weights,
						   bool verbose=false);
     int read_nb_k_points(const alps::params& parms, bool verbose=false);
     void generate_bseq_lattice(int n_q_mesh, double min_xq_mesh, double min_yq_mesh,
                                double len_q_mesh, int irr_direction);
     // The average of the dispersion over the k lattice.
     Eigen::MatrixXcd epsilon_bar;
     Eigen::MatrixXcd epsilon_squared_bar;
     std::vector<std::vector<double> > k_lattice_;
     std::vector<Eigen::VectorXd> proc_k_lattice_;
     // Lattice for bseq
     std::vector<Eigen::VectorXd> secondary_q_lattice_;
     std::vector<Eigen::VectorXi> r_lattice_;
     std::vector<Eigen::MatrixXcd> hoppings_;
     Eigen::VectorXd weights_;
     int orbital_size_;
     int per_site_orbital_size;
     int n_space_sites;
     std::vector<Eigen::VectorXd> bare_dos;
     std::vector<Eigen::VectorXd> world_bare_dos;
     // Number of direct space cells linked by finite
     // hoppings to the (0, 0) unit cell, i.e.
     // number of defined hoppings in hr.dat
     int nb_r_points;
     // The number of meaningful k-points
     // additional points are added for padding, so that
     // scattering and gathering to / from multiple CPUS
     // is made easier.
     int real_n_points;
     static const std::complex<double> infinitesimal;
     // Value of the frequency for cutoff
     static const double freq_cutoff;
     // Former Im, number of numerically handled Matsubara freqs.
     static const int nb_freq_points;
     // dimensions of the reciprocal lattice.
     // TODO: make this user-controlled parameters.
     static const int x_dim;
     static const int y_dim;
     static const int z_dim;
     // Numerical precision of all file outputs
     static const int output_precision;
     // Name of the file for dumping the Hamiltonian in k-space
     static const string hamiltonian_dump_name;
     // Name of the file for dumping the bare DOS
     static const std::string bare_dos_dump_name;
     // Numerical tolerance for Hermiticity test
     static const double hermitian_tolerance;
private:
     int world_rank_;
     int world_size, n_points_per_proc;
};
