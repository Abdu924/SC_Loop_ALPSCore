#include "bseq_solver.hpp"

using namespace std;

BseqSolver::BseqSolver(alps::hdf5::archive &g2_h5_archive, alps::hdf5::archive &bubble_h5_archive,
                        boost::shared_ptr<Bandstructure> const &lattice_bs,
                        int world_rank)
     : world_rank_(world_rank) {
     if (world_rank == 0) {
          MPI_Comm_size(MPI_COMM_WORLD, &world_size);
          nb_q_points_per_proc = nb_q_points / world_size;
          if (nb_q_points % world_size > 0) {
               nb_q_points_per_proc += 1;
          }
     }
     MPI_Bcast(&world_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&nb_q_points_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

const std::string BseqSolver::susceptibility_dump_filename = "c_susceptibility";
