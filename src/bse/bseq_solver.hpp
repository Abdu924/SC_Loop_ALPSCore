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

class BseqSolver {
public:
     BseqSolver(alps::hdf5::archive &g2_h5_archive,
                alps::hdf5::archive &bubble_h5_archive,
                boost::shared_ptr<Bandstructure> const &lattice_bs, int world_rank);
     virtual ~BseqSolver() {}
     

     boost::multi_array<std::complex<double>, 7> local_legendre_values_;
     boost::multi_array<std::complex<double>, 8> lattice_legendre_values_;
     
private:
     int world_rank_;
     int world_size;
     int nb_q_points;
     int nb_q_points_per_proc;
     int N_boson;
     int n_legendre;
     int n_sites;
     int per_site_orbital_size;
     boost::multi_array<complex<double> , 4> fixed_legendre_gf_;
     
     static const std::string susceptibility_dump_filename;
};
