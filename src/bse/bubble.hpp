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

using namespace std;

class LocalBubble {
public:
     LocalBubble(alps::hdf5::archive &h5_archive,
                 boost::shared_ptr<Bandstructure> const &lattice_bs,
                 boost::shared_ptr<Selfenergy> const &sigma,
                 const alps::params& parms, complex<double> chemical_potential,
                 int world_rank);
     void dump_bubble_hdf5();
     void compute_local_bubble();
     virtual ~LocalBubble() {}

     boost::multi_array<std::complex<double>, 6> values_;
     
private:
     int world_rank_;
     boost::shared_ptr<Bandstructure> lattice_bs_;
     boost::shared_ptr<Selfenergy> sigma_;
     std::complex<double> chemical_potential;
     int n_orbitals;
     int N_l_G4;
     int N_boson;
     int bubble_dim;
     int n_sites;
     int tot_orbital_size;
     int per_site_orbital_size;
     boost::multi_array<complex<double> , 3> raw_full_gf;
     vector<Eigen::MatrixXcd> world_local_gf;
     vector<vector<Eigen::MatrixXcd> > world_local_bubble;

     static const std::string bubble_hdf5_root;
};

class LatticeBubble {
public:
     LatticeBubble();
     boost::multi_array<std::complex<double>, 7> values_;
     
     virtual ~LatticeBubble() {}

private:
     int n_orbitals;
     int N_l_G4;
     int N_boson;
};
