#pragma once

#include <Eigen/Dense>
#include <boost/multi_array.hpp>
#include <boost/range/algorithm.hpp>
#include <tuple>
#include "../shared_libs/band_structure.hpp"
#include "../shared_libs/self_energy.hpp"

using namespace std;

class LocalBubble {
public:
     LocalBubble(boost::shared_ptr<Bandstructure> const &lattice_bs,
                 boost::shared_ptr<Selfenergy> const &sigma,
                 const alps::params& parms, complex<double> chemical_potential,
                 int world_rank);
     boost::multi_array<std::complex<double>, 4> values_;
     
     virtual ~LocalBubble() {}
     
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
     vector<Eigen::MatrixXcd> world_local_gf;
     vector<vector<Eigen::MatrixXcd> > world_local_bubble;
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
