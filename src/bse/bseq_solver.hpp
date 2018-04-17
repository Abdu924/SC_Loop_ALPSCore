#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <boost/multi_array.hpp>
#include <boost/bimap.hpp>
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
#include "../shared_libs/legendre.hpp"
#include "../shared_libs/flavor_transformer.hpp"

using namespace std;

typedef Eigen::Tensor<std::complex<double>, 4> local_g2_type;
typedef Eigen::Tensor<std::complex<double>, 5> lattice_g2_type;

typedef boost::multi_array<std::complex<double>, 4> local_leg_type;
typedef boost::multi_array<std::complex<double>, 3> g1_type;

typedef boost::multi_array<std::complex<double>, 7> extended_local_leg_type;
typedef boost::multi_array<std::complex<double>, 8> extended_lattice_leg_type;

class BseqSolver {

public:
     BseqSolver(alps::hdf5::archive &g2_h5_archive,
                alps::hdf5::archive &bubble_h5_archive,
                boost::shared_ptr<Bandstructure> const &lattice_bs,
                int current_bose_freq,
                const alps::params& parms, int world_rank);
     virtual ~BseqSolver() {}

     void dump_susceptibility(const alps::params& parms);
     void inverse_bseq();
     
private:
     int world_rank_;
     int world_size;
     
     int nb_q_points;
     int nb_q_points_per_proc;
     int N_boson;
     int current_bose_freq;
     int n_legendre;
     int n_sites;
     int per_site_orbital_size;
     double beta;
     boost::shared_ptr<Bandstructure> lattice_bs_;
     boost::multi_array<complex<double> , 4> fixed_legendre_gf_;
     local_g2_type g2_data_;
     local_g2_type local_legendre_bubble_;
     lattice_g2_type lattice_chi_;
     lattice_g2_type world_lattice_chi_;
     // local_leg_type irr_vertex_;
     lattice_g2_type lattice_legendre_bubble_;
     Eigen::MatrixXcd flat_irreducible_vertex;
     boost::shared_ptr<FlavorTransformer> flavor_trans_;
     
     void dump_for_check();
     void read_local_g2(alps::hdf5::archive &g2_h5_archive);
     void read_local_bubble(alps::hdf5::archive &bubble_h5_archive);
     void read_lattice_bubble(alps::hdf5::archive &bubble_h5_archive);
     Eigen::MatrixXcd get_flattened_representation(local_g2_type& tensor);
     local_g2_type get_multidim_representation(const Eigen::Ref<Eigen::MatrixXcd> flat_data);
     void build_matrix_shuffle_map();
     void subtract_disconnected_part(alps::hdf5::archive &g2_h5_archive);
};
