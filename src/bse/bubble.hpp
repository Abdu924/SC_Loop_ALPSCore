#pragma once

#include <Eigen/Dense>
#include <boost/multi_array.hpp>
#include <boost/range/algorithm.hpp>
#include <tuple>
#include <alps/hdf5.hpp>
#include <alps/params.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/params/convenience_params.hpp>
#include <boost/timer/timer.hpp>

#include "../shared_libs/band_structure.hpp"
#include "../shared_libs/self_energy.hpp"
#include "../shared_libs/legendre.hpp"

using namespace std;

class Bubble {
public:
     Bubble(alps::hdf5::archive &h5_archive,
                 boost::shared_ptr<Bandstructure> const &lattice_bs,
                 boost::shared_ptr<Selfenergy> const &sigma,
                 const alps::params& parms, complex<double> chemical_potential,
                 int world_rank);
     void dump_bubble_hdf5();
     void compute_local_bubble();
     void get_local_legendre_representation();
     void compute_lattice_bubble();
     void get_lattice_legendre_representation();
     virtual ~Bubble() {}

     boost::multi_array<std::complex<double>, 6> local_values_;
     boost::multi_array<std::complex<double>, 7> local_legendre_values_;
     boost::multi_array<std::complex<double>, 7> lattice_values_;
     boost::multi_array<std::complex<double>, 8> lattice_legendre_values_;
     boost::multi_array<std::complex<double>, 8> world_lattice_legendre_values_;
     
private:
     int world_rank_;
     int world_size;
     boost::shared_ptr<Bandstructure> lattice_bs_;
     boost::shared_ptr<Selfenergy> sigma_;
     std::complex<double> chemical_potential;
     int nb_q_points;
     int nb_q_points_per_proc;
     int N_l_G4;
     int N_boson;
     int bubble_dim;
     int n_sites;
     int tot_orbital_size;
     int per_site_orbital_size;
     int n_legendre;
     int dump_legendre;
     int dump_matsubara;
     boost::multi_array<complex<double> , 3> raw_full_gf;
     vector<vector<vector<Eigen::MatrixXcd> > > lattice_bubble;
     vector<vector<vector<Eigen::MatrixXcd> > > world_lattice_bubble;
     boost::shared_ptr<LegendreTransformer> legendre_trans_;
     
     std::vector<Eigen::MatrixXcd> get_greens_function(
	  Eigen::Ref<Eigen::VectorXd> k_point, int boson_index);
     Eigen::MatrixXcd get_legendre_representation(Eigen::Ref<Eigen::MatrixXcd> matsu_data);
     
     static const std::string bubble_hdf5_root;
};
