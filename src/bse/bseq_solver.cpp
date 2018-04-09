#include "bseq_solver.hpp"

using namespace std;

BseqSolver::BseqSolver(alps::hdf5::archive &g2_h5_archive,
                       alps::hdf5::archive &bubble_h5_archive,
                       boost::shared_ptr<Bandstructure> const &lattice_bs,
                       int in_current_bose_freq,
                       const alps::params& parms, int world_rank)
     : lattice_bs_(lattice_bs), current_bose_freq(in_current_bose_freq),
       world_rank_(world_rank) {
     nb_q_points = lattice_bs_->get_nb_points_for_bseq();
     per_site_orbital_size = lattice_bs_->get_orbital_size();
     n_legendre = parms["measurement.G2.n_legendre"];
     N_boson = parms["measurement.G2.n_bosonic_freq"];
     n_sites = 1;
     if (world_rank == 0) {
          MPI_Comm_size(MPI_COMM_WORLD, &world_size);
          nb_q_points_per_proc = nb_q_points / world_size;
          if (nb_q_points % world_size > 0) {
               nb_q_points_per_proc += 1;
          }
     }
     MPI_Bcast(&world_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&nb_q_points_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
     if (world_rank == 0) {
          local_legendre_bubble_.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                                        [per_site_orbital_size][per_site_orbital_size]
                                        [n_legendre][n_legendre]);
          std::fill(local_legendre_bubble_.origin(), local_legendre_bubble_.origin() + local_legendre_bubble_.num_elements(), 0.0);
          read_local_bubble(bubble_h5_archive);
          read_local_g2(g2_h5_archive);
     }
}

void BseqSolver::read_local_g2(alps::hdf5::archive &g2_h5_archive) {
     extended_local_leg_type temp_g2_data;
     temp_g2_data.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                         [per_site_orbital_size][per_site_orbital_size]
                         [n_legendre][n_legendre][N_boson]);
     g2_h5_archive["G2_LEGENDRE"] >> temp_g2_data;
     g2_data_.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                     [per_site_orbital_size][per_site_orbital_size]
                     [n_legendre][n_legendre]);
     for (int orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
          for (int orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
               for (int orb3 = 0; orb3 < per_site_orbital_size; orb3++) {
                    for (int orb4 = 0; orb4 < per_site_orbital_size; orb4++) {
                         for (int l1 = 0; l1 < n_legendre; l1++) {
                              for (int l2 = 0; l2 < n_legendre; l2++) {
                                   g2_data_[orb1][orb2][orb3][orb4][l1][l2] =
                                        temp_g2_data[orb1][orb2][orb3][orb4][l1][l2][current_bose_freq];
                              }
                         }
                    }
               }
          }
     }
}

void BseqSolver::read_local_bubble(alps::hdf5::archive &bubble_h5_archive) {
     
     bubble_h5_archive["/legendre_local_bubble/site_0/data"] >> local_legendre_bubble_;
}

const std::string BseqSolver::susceptibility_dump_filename = "c_susceptibility.h5";
