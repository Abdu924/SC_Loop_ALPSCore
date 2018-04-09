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
#include "../shared_libs/legendre.hpp"

using namespace std;
typedef boost::multi_array<std::complex<double>, 8> lattice_leg_type;
typedef boost::multi_array<std::complex<double>, 6> local_leg_type;
typedef boost::multi_array<std::complex<double>, 7> extended_local_leg_type;

class BseqSolver {

public:
     BseqSolver(alps::hdf5::archive &g2_h5_archive,
                alps::hdf5::archive &bubble_h5_archive,
                boost::shared_ptr<Bandstructure> const &lattice_bs,
                int current_bose_freq,
                const alps::params& parms, int world_rank);
     virtual ~BseqSolver() {}
     
     
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
     boost::shared_ptr<Bandstructure> lattice_bs_;
     boost::multi_array<complex<double> , 4> fixed_legendre_gf_;
     local_leg_type g2_data_;
     local_leg_type local_legendre_bubble_;
     lattice_leg_type lattice_legendre_bubble_;

     void read_local_g2(alps::hdf5::archive &g2_h5_archive);
     void read_local_bubble(alps::hdf5::archive &bubble_h5_archive);
     
     static const std::string susceptibility_dump_filename;
};
