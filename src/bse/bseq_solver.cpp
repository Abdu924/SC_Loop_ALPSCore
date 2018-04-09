#include "bseq_solver.hpp"

using namespace std;

BseqSolver::BseqSolver(alps::hdf5::archive &g2_h5_archive,
                       alps::hdf5::archive &bubble_h5_archive,
                       boost::shared_ptr<Bandstructure> const &lattice_bs,
                       const alps::params& parms, int world_rank)
     : lattice_bs_(lattice_bs), world_rank_(world_rank) {
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
                                        [n_legendre][n_legendre][N_boson]);
          std::fill(local_legendre_bubble_.origin(), local_legendre_bubble_.origin() + local_legendre_bubble_.num_elements(), 0.0);
          read_local_bubble(bubble_h5_archive);
     }
}

void BseqSolver::read_local_g2(alps::hdf5::archive &g2_h5_archive) {
     //h5_archive["/legendre_gf/data"] >> raw_full_gf;
}

void BseqSolver::read_local_bubble(alps::hdf5::archive &bubble_h5_archive) {
     bubble_h5_archive["/legendre_local_bubble/site_0/data"] >> local_legendre_bubble_;
}

const std::string BseqSolver::susceptibility_dump_filename = "c_susceptibility.h5";
