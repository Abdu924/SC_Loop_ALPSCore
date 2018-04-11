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
     beta = parms["model.beta"];
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
     build_matrix_shuffle_map();
     if (world_rank == 0) {
          read_local_bubble(bubble_h5_archive);
          read_local_g2(g2_h5_archive);
          subtract_disconnected_part(g2_h5_archive);
          Eigen::MatrixXcd flat_view;
          get_flattened_representation(g2_data_, flat_view);
     }
}

void BseqSolver::get_flattened_representation(
     Eigen::Tensor<std::complex<double>, 4>& tensor,
     Eigen::Ref<Eigen::MatrixXcd> result) {
     assert(tensor.dimension(0) = tensor.dimension(1));
     assert(tensor.dimension(2) = tensor.dimension(3));
     int orb_dim = tensor.dimension(0);
     int leg_dim = tensor.dimension(2);
     int flattened_dim = orb_dim * leg_dim;
     result = Eigen::MatrixXcd::Zero(flattened_dim, flattened_dim);
     for (int l1 = 0; l1 < leg_dim; l1++) {
          for (int l2 = 0; l2 < leg_dim; l2++) {
               Eigen::array<int, 2> offsets = {1, 0};
               Eigen::array<int, 2> extents = {2, 2};
               Eigen::Tensor<std::complex<double>, 2> test = (tensor.chip(l1, 3)).chip(l2, 4);
               for (int orb1 = 0; orb1 < orb_dim; orb1++) {
                    for (int orb2 = 0; orb2 < orb_dim; orb2++) {
                         //Eigen::Tensor<int, 1> slice = a.slice(offsets, extents);
                         result.block(l1 * orb_dim, l2 * orb_dim, orb_dim, orb_dim)(orb1, orb2) =
                              test(orb1, orb2);
                    }
               }
          }
     }
}

void BseqSolver::build_matrix_shuffle_map() {
     std::map<int, int> corresp1, corresp2;
     // we go from
     // aup bdown adown bup
     // to
     // aup bup adown bdown
     line_from_orbital_pair.clear();
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 0), 0));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 3), 1));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 0), 2));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 3), 3));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 2), 4));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 1), 5));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 2), 6));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 1), 7));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 2), 8));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 1), 9));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 2), 10));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 1), 11));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 0), 12));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 3), 13));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 0), 14));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 3), 15));
     col_from_orbital_pair.clear();
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 0), 0));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 0), 1));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 3), 2));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 3), 3));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 2), 4));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 2), 5));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 1), 6));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 1), 7));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 0), 8));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 0), 9));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 3), 10));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 3), 11));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 2), 12));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 2), 13));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 1), 14));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 1), 15));     
}

// We transform to the orbital order used by J. Kunes and others (ijkl) vs (ijlk).
// we also transform the 4-orbital indexing scheme to the 2-orbital-pair indexing scheme
// Block diagonalization will be introduced as an option at a later stage. (maybe :) )
// The matrix format is given in Boehnke's PhD thesis.
void BseqSolver::read_local_g2(alps::hdf5::archive &g2_h5_archive) {
     extended_local_leg_type temp_g2_data;
     temp_g2_data.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                         [per_site_orbital_size][per_site_orbital_size]
                         [n_legendre][n_legendre][N_boson]);
     g2_h5_archive["G2_LEGENDRE"] >> temp_g2_data;
     g2_data_ = Eigen::Tensor<std::complex<double>, 4>(per_site_orbital_size * per_site_orbital_size,
                                                       per_site_orbital_size * per_site_orbital_size,
                                                       n_legendre,n_legendre);
     g2_data_.setZero();
     int line_idx, col_idx;
     for (int orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
          for (int orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
               for (int orb3 = 0; orb3 < per_site_orbital_size; orb3++) {
                    for (int orb4 = 0; orb4 < per_site_orbital_size; orb4++) {
                         for (int l1 = 0; l1 < n_legendre; l1++) {
                              for (int l2 = 0; l2 < n_legendre; l2++) {
                                   line_idx = line_from_orbital_pair.left.at(std::make_pair(orb1, orb2));
                                   col_idx = col_from_orbital_pair.left.at(std::make_pair(orb3, orb4));
                                   g2_data_(line_idx, col_idx, l1, l2) =
                                        -temp_g2_data[orb1][orb2][orb3][orb4][l1][l2][current_bose_freq] / beta;
                              }
                         }
                    }
               }
          }
     }
}

void BseqSolver::subtract_disconnected_part(alps::hdf5::archive &g2_h5_archive) {
     if (current_bose_freq == 0) {
          g1_type temp_g1_data;
          temp_g1_data.resize(boost::extents[per_site_orbital_size][per_site_orbital_size][n_legendre]);
          g2_h5_archive["legendre_gf_fixed/data"] >> temp_g1_data;
          local_g2_type disconnected_part;
          disconnected_part = Eigen::Tensor<std::complex<double>, 4>(per_site_orbital_size * per_site_orbital_size,
                                                                     per_site_orbital_size * per_site_orbital_size,
                                                                     n_legendre,n_legendre);
          disconnected_part.setZero();
          int line_idx, col_idx;
          for (int orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
               for (int orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
                    for (int orb3 = 0; orb3 < per_site_orbital_size; orb3++) {
                         for (int orb4 = 0; orb4 < per_site_orbital_size; orb4++) {
                              for (int l1 = 0; l1 < n_legendre; l1++) {
                                   double sign_factor = -1.0;
                                   for (int l2 = 0; l2 < n_legendre; l2++) {
                                        line_idx = line_from_orbital_pair.left.at(std::make_pair(orb1, orb2));
                                        col_idx = col_from_orbital_pair.left.at(std::make_pair(orb3, orb4));
                                        disconnected_part(line_idx, col_idx, l1, l2) =
                                             temp_g1_data[orb1][orb2][l1] *
                                             temp_g1_data[orb3][orb4][l2] * sign_factor;
                                        sign_factor *= -1.0;
                                   }
                              }
                         }
                    }
               }
          }
          g2_data_ = g2_data_ + disconnected_part;
     }
}

void BseqSolver::read_local_bubble(alps::hdf5::archive &bubble_h5_archive) {
     extended_local_leg_type temp_local_bubble;
     bubble_h5_archive["/legendre_local_bubble/site_0/data"] >> temp_local_bubble;
     local_legendre_bubble_ = Eigen::Tensor<std::complex<double>, 4>(per_site_orbital_size * per_site_orbital_size,
                                                                     per_site_orbital_size * per_site_orbital_size,
                                                                     n_legendre,n_legendre);
     local_legendre_bubble_.setZero();
     int line_idx, col_idx;
     for (int orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
          for (int orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
               for (int orb3 = 0; orb3 < per_site_orbital_size; orb3++) {
                    for (int orb4 = 0; orb4 < per_site_orbital_size; orb4++) {
                         for (int l1 = 0; l1 < n_legendre; l1++) {
                              for (int l2 = 0; l2 < n_legendre; l2++) {
                                   line_idx = line_from_orbital_pair.left.at(std::make_pair(orb1, orb2));
                                   col_idx = col_from_orbital_pair.left.at(std::make_pair(orb3, orb4));
                                   local_legendre_bubble_(line_idx, col_idx, l1, l2) =
                                        temp_local_bubble[orb1][orb2][orb3][orb4][l1][l2][current_bose_freq];
                              }
                         }
                    }
               }
          }
     }
}


const std::string BseqSolver::susceptibility_dump_filename = "c_susceptibility.h5";
