#include "bubble.hpp"

using namespace std;

LatticeBubble::LatticeBubble() {
     values_.resize(boost::extents[n_orbitals][n_orbitals][n_orbitals][n_orbitals][N_l_G4][N_l_G4][N_boson]);
}

LocalBubble::LocalBubble(boost::shared_ptr<Bandstructure> const &lattice_bs,
                         boost::shared_ptr<Selfenergy> const &sigma,
                         const alps::params& parms, complex<double> chemical_potential,
                         int world_rank)
     :lattice_bs_(lattice_bs), sigma_(sigma), chemical_potential(chemical_potential),
      world_rank_(world_rank) {
     int N_max = sigma_->get_n_matsubara_freqs();
     N_boson = parms["measurement.G2.n_bosonic_freq"];
     int N_Qmesh = parms["bseq.N_QBSEQ"];
     bubble_dim = parms.exists("bseq.N_NU_BSEQ") ? parms["bseq.N_NU_BSEQ"] : N_max - N_boson;
     n_sites = sigma_->get_n_sites();
     per_site_orbital_size = sigma_->get_per_site_orbital_size();
     tot_orbital_size = n_sites * per_site_orbital_size;

     world_local_gf.clear();
     for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
          world_local_gf.push_back(Eigen::MatrixXcd::Zero(
                                        tot_orbital_size, tot_orbital_size));
     }
     values_.resize(boost::extents[N_boson][bubble_dim]
                    [n_sites * per_site_orbital_size * per_site_orbital_size]
                    [n_sites * per_site_orbital_size * per_site_orbital_size]);
     std::fill(values_.origin(), values_.origin() + values_.num_elements(), 0.0);
}
