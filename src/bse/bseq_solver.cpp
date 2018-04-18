#include "bseq_solver.hpp"

using namespace std;

BseqSolver::BseqSolver(alps::hdf5::archive &g2_h5_archive,
                       alps::hdf5::archive &bubble_h5_archive,
                       boost::shared_ptr<Bandstructure> const &lattice_bs,
                       int in_current_bose_freq,
                       const alps::params& parms, int world_rank)
     : lattice_bs_(lattice_bs), current_bose_freq(in_current_bose_freq),
       world_rank_(world_rank)
{
     nb_q_points = lattice_bs_->get_nb_points_for_bseq();
     per_site_orbital_size = lattice_bs_->get_orbital_size();
     n_legendre = parms["bseq.inversion.n_legendre"];
     N_boson = parms["bseq.inversion.n_bosonic_freq"];
     assert (N_boson <= parms["bseq.bubbles.n_bosonic_freq"]);
     assert (N_boson <= parms["measurement.G2.n_bosonic_freq"]);
     beta = parms["model.beta"];
     n_sites = 1;
     if (world_rank_ == 0) {
          MPI_Comm_size(MPI_COMM_WORLD, &world_size);
          nb_q_points_per_proc = nb_q_points / world_size;
          if (nb_q_points % world_size > 0) {
               nb_q_points_per_proc += 1;
          }
     }
     MPI_Bcast(&world_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&nb_q_points_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
     // build correspondence map between orbitals, rows and columns, in order
     // to go from the excitonic convention to the usual susceptibility convention
     flavor_trans_.reset(new FlavorTransformer());
     if (world_rank_ == 0) {
          {
               boost::timer::auto_cpu_timer bubble_calc;
               read_local_bubble(bubble_h5_archive);
               cout << "Local bubble read: ";
          }
          {
               boost::timer::auto_cpu_timer bubble_calc;
               read_local_g2(g2_h5_archive);
               cout << "Local G2 read: ";
          }
          if (current_bose_freq == 0) {
               boost::timer::auto_cpu_timer bubble_calc;
               subtract_disconnected_part(g2_h5_archive);
               cout << "subtract disconnect: ";
          }




          
          cout << "test value " << beta * real(g2_data_(0,0, 2, 2)) << endl;
          Eigen::MatrixXcd flat_g2 = get_flattened_representation(g2_data_);
          cout << "test value " << beta * real(flat_g2(2 * 16, 2 * 16)) << endl;
          Eigen::MatrixXcd flat_bubble = get_flattened_representation(local_legendre_bubble_);
          local_g2_type retour = get_multidim_representation(flat_g2);
          Eigen::MatrixXcd reretour = get_flattened_representation(retour);
          cout << "consistency " << (flat_g2 - reretour).cwiseAbs().sum() << endl;
          cout << "flat_g2 "  << " abs som "  << flat_g2.cwiseAbs().sum() << endl;
          cout << "flat_g2 det " << abs(flat_g2.determinant()) << endl;
          // for (int i = 0; i < n_legendre; i++) {
          //      for (int j = 0; j < n_legendre; j++) {
          //           cout << "flat_g2 block " << i << ", " << j << " det " <<
          //                flat_g2.block(i*per_site_orbital_size, j*per_site_orbital_size,
          //                              n_legendre, n_legendre).determinant() << endl;
          //      }
          // }
          cout << "flat_g2 h5 mimck" << endl;
          for (int i = 0; i < n_legendre; i++) {
               for (int j = 0; j < n_legendre; j++) {
                    cout << beta * real(flat_g2.block(i * per_site_orbital_size*per_site_orbital_size,
                                                      j * per_site_orbital_size*per_site_orbital_size,
                                                      per_site_orbital_size*per_site_orbital_size,
                                                      per_site_orbital_size*per_site_orbital_size)(0,0)) << " ";
               }
               cout << endl;
          }
          cout << "flat_g2 h5 mimck dets" << endl;
          for (int i = 0; i < n_legendre; i++) {
               for (int j = 0; j < n_legendre; j++) {
                    cout << abs(flat_g2.block(i * per_site_orbital_size*per_site_orbital_size,
                                          j * per_site_orbital_size*per_site_orbital_size,
                                          per_site_orbital_size*per_site_orbital_size,
                                              per_site_orbital_size*per_site_orbital_size).determinant()) << " ";
               }
               cout << endl;
          }

          cout << "flat_bubble h5 mimck" << endl;
          for (int i = 0; i < n_legendre; i++) {
               for (int j = 0; j < n_legendre; j++) {
                    cout << beta * real(flat_bubble.block(i * per_site_orbital_size*per_site_orbital_size,
                                                      j * per_site_orbital_size*per_site_orbital_size,
                                                      per_site_orbital_size*per_site_orbital_size,
                                                      per_site_orbital_size*per_site_orbital_size)(0,0)) << " ";
               }
               cout << endl;
          }
          cout << "flat_bubble h5 mimck dets" << endl;
          for (int i = 0; i < n_legendre; i++) {
               for (int j = 0; j < n_legendre; j++) {
                    cout << abs(flat_bubble.block(i * per_site_orbital_size*per_site_orbital_size,
                                          j * per_site_orbital_size*per_site_orbital_size,
                                          per_site_orbital_size*per_site_orbital_size,
                                              per_site_orbital_size*per_site_orbital_size).determinant()) << " ";
               }
               cout << endl;
          }

          
          // helper function for checks
          if (false)
               dump_for_check();
          {
               boost::timer::auto_cpu_timer bubble_calc;
               flat_irreducible_vertex = flat_g2.inverse() - flat_bubble.inverse();
               for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                         Eigen::array<int, 4> offsets = {4 * i, 4 * j, 0, 0};
                         Eigen::array<int, 4> extents = {4, 4, n_legendre, n_legendre};
                         local_g2_type slice = g2_data_.slice(offsets, extents);
                         cout << "SLICE 4!!!!!!!!!!!" << endl;
                         Eigen::MatrixXcd flat_slice = get_flattened_representation(slice);
                         cout << "block " << i << ", " << j << ", abs som "  << flat_slice.cwiseAbs().sum() << endl;
                         cout << "flat_slice block " << i << ", " << j << " det "  << flat_slice.determinant() << endl;
                         cout << "flat_slice block " << i << ", " << j << endl  << flat_slice.real() << endl;
                         cout << "flat_slice block " << i << ", " << j << endl  << flat_slice.imag() << endl;
                    }
               }
               for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                         Eigen::array<int, 4> offsets = {8 * i, 8 * j, 0, 0};
                         Eigen::array<int, 4> extents = {8, 8, n_legendre, n_legendre};
                         local_g2_type slice = g2_data_.slice(offsets, extents);
                         Eigen::MatrixXcd flat_slice = get_flattened_representation(slice);
                         cout << "block " << i << ", " << j << ", abs som "  << flat_slice.cwiseAbs().sum() << endl;
                         cout << "flat_slice block " << i << ", " << j << " det "  << flat_slice.determinant() << endl;
                         cout << "flat_slice block " << i << ", " << j << " inv sum "  <<
                              flat_slice.inverse().cwiseAbs().sum() << endl;
                    }
               }
               cout << "flat_slice global "<< " det "  <<
                    flat_g2.inverse().inverse().cwiseAbs().sum() << endl;

               

               

               // cout << "flat_g2 row 8 " << flat_g2.row(8).segment(8, 4) << endl;
               // cout << "flat_g2 row 9 " << flat_g2.row(9).segment(8, 4) << endl;
               // cout << "flat_g2 row 10 " << flat_g2.row(10).segment(8, 4) << endl;
               // cout << "flat_g2 row 11 " << (flat_g2.row(11)).segment(8, 4) << endl;
               cout << "flat_bubble det " << flat_bubble.determinant() << endl;
               cout << "flat_g2 det " << flat_g2.determinant() << endl;
               cout << "flat_g2(0, 0, 0, 0, 0) " << flat_g2(0, 0) << endl;
               cout << "flat_bubble(0, 0, 0, 0, 0) " << flat_bubble(0, 0) << endl;
               cout << "flat_irreducible_vertex(0, 0, 0, 0, 0) " << flat_irreducible_vertex(0, 0) << endl;
               cout << "Get irreducible vertex (inversion): ";
          }
     } else {
          // resize target objects before mpi broadcast
          int vertex_size = per_site_orbital_size * per_site_orbital_size * n_legendre;
          flat_irreducible_vertex = Eigen::MatrixXcd::Zero(vertex_size, vertex_size);
     }
     MPI_Barrier(MPI_COMM_WORLD);
     try {
          MPI_Bcast(flat_irreducible_vertex.data(), flat_irreducible_vertex.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
     } catch(std::exception& exc){
	  std::cerr<<exc.what()<<std::endl;
     } catch(...){
	  std::cerr << "Fatal Error: Unknown Exception!\n";
     }
     
     {
          boost::timer::auto_cpu_timer bubble_calc;
          read_lattice_bubble(bubble_h5_archive);
          cout << "read lattice bubble";
     }
}

void BseqSolver::inverse_bseq() {     
     {
          boost::timer::auto_cpu_timer inverse_calc;
          if (world_rank_ == 0)
          {
               world_lattice_chi_ = lattice_g2_type(
                    per_site_orbital_size * per_site_orbital_size,
                    per_site_orbital_size * per_site_orbital_size,
                    n_legendre, n_legendre, nb_q_points);
                    world_lattice_chi_.setZero();
          }
          lattice_chi_ = lattice_g2_type(
               per_site_orbital_size * per_site_orbital_size,
               per_site_orbital_size * per_site_orbital_size,
               n_legendre, n_legendre, nb_q_points_per_proc);
          lattice_chi_.setZero();
          for (int q_index = 0; q_index < nb_q_points_per_proc; q_index++) {
               int world_q_index = q_index + nb_q_points_per_proc * world_rank_;
               if (world_q_index >= nb_q_points)
                    continue;
               local_g2_type temp_lattice_bubble = lattice_legendre_bubble_.chip(q_index, 4);
               Eigen::MatrixXcd flat_lattice_chi = (flat_irreducible_vertex +
                                                    (get_flattened_representation(temp_lattice_bubble)).inverse()).inverse();
               //local_g2_type temp_lattice_chi = lattice_chi_.chip(q_index, 4);
               //temp_lattice_chi = get_multidim_representation(flat_lattice_chi);
               cout << "flat_irreducible_vertex(0, 0, 0, 0, 0) " << flat_irreducible_vertex(0, 0) << endl;
               cout << "get_flattened_representation(temp_lattice_bubble)(0, 3, 0, 0, 0) " <<
                    get_flattened_representation(temp_lattice_bubble)(0, 3) << endl;

               cout << "flat_lattice_chi(0, 0, 0, 0, 0) " << flat_lattice_chi(0, 0) << endl;
               cout << "flat_lattice_chi(0, 3, 0, 0, 0) " << flat_lattice_chi(0, 3) << endl;
               lattice_chi_.chip(q_index, 4) = get_multidim_representation(flat_lattice_chi);
               cout << "lattice_chi_(0, 0, 0, 0, 0) " << lattice_chi_(0, 0, 0, 0, 0) << endl;
               cout << "lattice_chi_(0, 3, 0, 0, 0) " << lattice_chi_(0, 3, 0, 0, 0) << endl;
          }
          cout << "Invert lattice BSE";
     }
     if (world_rank_ == 0) {
          for (int proc_index = 1; proc_index < world_size; proc_index++) {
               for (size_t q_index = 0; q_index < nb_q_points_per_proc; q_index++) {
                    int world_q_index = q_index + nb_q_points_per_proc * proc_index;
                    if (world_q_index >= nb_q_points)
                         continue;
                    local_g2_type temp_world_lattice_chi(world_lattice_chi_.dimensions()[0],
                                                         world_lattice_chi_.dimensions()[1],
                                                         world_lattice_chi_.dimensions()[2],
                                                         world_lattice_chi_.dimensions()[3]);
                    temp_world_lattice_chi.setZero();
                    MPI_Recv(temp_world_lattice_chi.data(),
                             temp_world_lattice_chi.size(),
                             MPI_DOUBLE_COMPLEX, proc_index, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    world_lattice_chi_.chip(world_q_index, 4) = temp_world_lattice_chi;
               }
          }
     } else {
          for (size_t q_index = 0; q_index < nb_q_points_per_proc; q_index++) {
               int world_q_index = q_index + nb_q_points_per_proc * world_rank_;
               if (world_q_index < nb_q_points) {
                    local_g2_type temp_lattice_chi = lattice_chi_.chip(q_index, 4);
                    MPI_Send(temp_lattice_chi.data(),
                             temp_lattice_chi.size(),
                             MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
               } else {
                    continue;
               }
          }
     }
     if (world_rank_ == 0) {
          for (size_t q_index = 0; q_index < nb_q_points_per_proc; q_index++) {
               world_lattice_chi_.chip(q_index, 4) = lattice_chi_.chip(q_index, 4);
          }
          cout << "world_lattice_chi_(0, 3, 0, 0, 0) " << world_lattice_chi_(0, 3, 0, 0, 0) << endl;
     }
}

void BseqSolver::dump_susceptibility(const alps::params& parms) {
     if (world_rank_ == 0) {
          const string archive_name = parms["bseq.inversion.filename"].as<string>();
          alps::hdf5::archive bseq_output(archive_name, "a");
          std::string h5_group_name("/lattice_chi");
          boost::multi_array<std::complex<double>, 5> temp_lattice_chi;
          temp_lattice_chi.resize(boost::extents
                                  [world_lattice_chi_.dimensions()[0]]
                                  [world_lattice_chi_.dimensions()[1]]
                                  [world_lattice_chi_.dimensions()[2]]
                                  [world_lattice_chi_.dimensions()[3]]
                                  [world_lattice_chi_.dimensions()[4]]);
          for (int site_index = 0; site_index < n_sites; site_index++) {
               std::stringstream site_path;
               site_path << h5_group_name + "/site_" +
                    boost::lexical_cast<std::string>(site_index) + "/data";
               for (int orb1 = 0; orb1 < world_lattice_chi_.dimensions()[0]; orb1++) {
                    for (int orb2 = 0; orb2 < world_lattice_chi_.dimensions()[1]; orb2++) {
                         for (int l1 = 0; l1 < world_lattice_chi_.dimensions()[2]; l1++) {
                              for (int l2 = 0; l2 < world_lattice_chi_.dimensions()[3]; l2++) {
                                   for (int q_index = 0; q_index < nb_q_points; q_index++) {
                                        temp_lattice_chi[orb1][orb2][l1][l2][q_index] =
                                             world_lattice_chi_(orb1, orb2, l1, l2, q_index);
                                   }
                              }
                         }
                    }
               }
               bseq_output[site_path.str()] << temp_lattice_chi;
          }
          // q point list
          h5_group_name = "/lattice_chi/q_point_list";
          std::vector<std::complex<double>> temp_data;
          temp_data.resize(nb_q_points);
          Eigen::VectorXd q_point;
          for (int q_index = 0; q_index < nb_q_points; q_index++) {
               q_point = lattice_bs_->get_q_point(q_index);
               temp_data[q_index] = std::complex<double>(q_point(0), q_point(1));
          }
          bseq_output[h5_group_name] = temp_data;
          // Close file
	  bseq_output.close();
     }
     MPI_Barrier(MPI_COMM_WORLD);
}


Eigen::MatrixXcd BseqSolver::get_flattened_representation(local_g2_type& tensor) {
     assert(tensor.dimension(0) = tensor.dimension(1));
     assert(tensor.dimension(2) = tensor.dimension(3));
     int orb_dim = tensor.dimension(0);
     int leg_dim = tensor.dimension(2);
     int flattened_dim = orb_dim * leg_dim;
     Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(flattened_dim, flattened_dim);
     for (int l1 = 0; l1 < leg_dim; l1++) {
          for (int l2 = 0; l2 < leg_dim; l2++) {
               Eigen::Tensor<std::complex<double>, 2> sub_matrix = (tensor.chip(l1, 2)).chip(l2, 2);
               for (int orb1 = 0; orb1 < orb_dim; orb1++) {
                    for (int orb2 = 0; orb2 < orb_dim; orb2++) {
                         result.block(l1 * orb_dim, l2 * orb_dim, orb_dim, orb_dim)(orb1, orb2) =
                              sub_matrix(orb1, orb2);
                    }
               }
          }
     }
     cout << "result.determinant " << result.determinant() << endl;
     return result;
}

local_g2_type BseqSolver::get_multidim_representation(const Eigen::Ref<Eigen::MatrixXcd> flat_data) {
     local_g2_type result(per_site_orbital_size * per_site_orbital_size,
                          per_site_orbital_size * per_site_orbital_size,
                          n_legendre,n_legendre);
     result.setZero();
     assert(per_site_orbital_size * per_site_orbital_size * n_legendre = flat_data.rows());
     int orb_dim = per_site_orbital_size * per_site_orbital_size;
     for (int l1 = 0; l1 < n_legendre; l1++) {
          for (int l2 = 0; l2 < n_legendre; l2++) {
               Eigen::Tensor<std::complex<double>, 2> sub_matrix(orb_dim, orb_dim);
               sub_matrix.setZero();
               for (int orb1 = 0; orb1 < orb_dim; orb1++) {
                    for (int orb2 = 0; orb2 < orb_dim; orb2++) {
                         sub_matrix(orb1, orb2) =
                              flat_data.block(l1 * orb_dim, l2 * orb_dim, orb_dim, orb_dim)(orb1, orb2);
                    }
               }
               (result.chip(l1, 2)).chip(l2, 2) = sub_matrix;
          }
     }
     return result;
}

void BseqSolver::dump_for_check() {
     // DUMP FOR check
     const string archive_name("test_flat.h5");
     alps::hdf5::archive test_output(archive_name, "a");
     std::stringstream site_path;
     std::string h5_group_name("/test_flat");
     site_path << h5_group_name + "/data";
     boost::multi_array<std::complex<double>, 2>  temp_flat_data;
     Eigen::MatrixXcd flat_g2 = get_flattened_representation(g2_data_);
     temp_flat_data.resize(boost::extents[flat_g2.rows()][flat_g2.rows()]);
     for (int i = 0; i < flat_g2.rows(); i++) {
          for (int j = 0; j < flat_g2.rows(); j++) {
               temp_flat_data[i][j] = flat_g2(i, j);
          }
     }                   
     test_output[site_path.str()] << temp_flat_data;
     std::stringstream site_path2;
     std::string h5_group_name2("/test_g2");
     site_path2 << h5_group_name2 + "/data";
     boost::multi_array<std::complex<double>, 4>  temp_g2_data;
     temp_g2_data.resize(boost::extents[per_site_orbital_size * per_site_orbital_size]
                         [per_site_orbital_size * per_site_orbital_size][n_legendre][n_legendre]);
     for (int i = 0; i < per_site_orbital_size * per_site_orbital_size; i++) {
          for (int j = 0; j < per_site_orbital_size * per_site_orbital_size; j++) {
               for (int k = 0; k < n_legendre; k++) {
                    for (int l = 0; l < n_legendre; l++) {
                         temp_g2_data[i][j][k][l] = g2_data_(i, j, k, l);
                    }
               }
          }
     }
     test_output[site_path2.str()] << temp_g2_data;
}

// We transform to the orbital order used by J. Kunes and others (ijkl) vs (ijlk).
// we also transform the 4-orbital indexing scheme to the 2-orbital-pair indexing scheme
// Block diagonalization will be introduced as an option at a later stage. (maybe :) )
// The matrix format is given in Boehnke's PhD thesis.
void BseqSolver::read_local_g2(alps::hdf5::archive &g2_h5_archive) {
     boost::multi_array<double, 8>  real_temp_g2_data;
     real_temp_g2_data.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                              [per_site_orbital_size][per_site_orbital_size]
                              [n_legendre][n_legendre][N_boson][2]);
     g2_h5_archive["G2_LEGENDRE"] >> real_temp_g2_data;
     g2_data_ = local_g2_type(per_site_orbital_size * per_site_orbital_size,
                              per_site_orbital_size * per_site_orbital_size,
                              n_legendre, n_legendre);
     g2_data_.setZero();
     int line_idx, col_idx;
     for (int orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
          for (int orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
               for (int orb3 = 0; orb3 < per_site_orbital_size; orb3++) {
                    for (int orb4 = 0; orb4 < per_site_orbital_size; orb4++) {
                         for (int l1 = 0; l1 < n_legendre; l1++) {
                              for (int l2 = 0; l2 < n_legendre; l2++) {
                                   line_idx = flavor_trans_->get_line_from_pair(orb1, orb2);
                                   col_idx = flavor_trans_->get_col_from_pair(orb3, orb4);
                                   g2_data_(line_idx, col_idx, l1, l2) =
                                        -std::complex<double>(
                                             real_temp_g2_data[orb1][orb2][orb3][orb4][l1][l2][current_bose_freq][0],
                                             real_temp_g2_data[orb1][orb2][orb3][orb4][l1][l2][current_bose_freq][1])
                                        / beta;
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
          //temp_g1_data.resize(boost::extents[per_site_orbital_size][per_site_orbital_size][n_legendre]);
          g2_h5_archive["legendre_gf_fixed/data"] >> temp_g1_data;
          assert(n_legendre <= temp_g1_data.shape()[2]);
          local_g2_type disconnected_part;
          disconnected_part = local_g2_type(per_site_orbital_size * per_site_orbital_size,
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
                                        line_idx = flavor_trans_->get_line_from_pair(orb1, orb2);
                                        col_idx = flavor_trans_->get_col_from_pair(orb3, orb4);
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
     extended_local_leg_type_compact temp_local_bubble;
     bubble_h5_archive["/legendre_local_bubble_compact/site_0/data"] >> temp_local_bubble;
     local_legendre_bubble_ = local_g2_type(temp_local_bubble.shape()[0],
                                            temp_local_bubble.shape()[1],
                                            n_legendre,n_legendre);
     local_legendre_bubble_.setZero();
     for (int orb1 = 0; orb1 < temp_local_bubble.shape()[0]; orb1++) {
          for (int orb2 = 0; orb2 < temp_local_bubble.shape()[1]; orb2++) {
               for (int l1 = 0; l1 < n_legendre; l1++) {
                    for (int l2 = 0; l2 < n_legendre; l2++) {
                         local_legendre_bubble_(orb1, orb2, l1, l2) =
                              temp_local_bubble[orb1][orb2][l1][l2][current_bose_freq];
                    }
               }
          }
     }
}

void BseqSolver::read_lattice_bubble(alps::hdf5::archive &bubble_h5_archive) {
     extended_lattice_leg_type_compact temp_lattice_bubble;
     bubble_h5_archive["/legendre_lattice_bubble_compact/site_0/data"] >> temp_lattice_bubble;
     lattice_legendre_bubble_ = lattice_g2_type(
          temp_lattice_bubble.shape()[0],
          temp_lattice_bubble.shape()[1],
          n_legendre,n_legendre, nb_q_points_per_proc);
     lattice_legendre_bubble_.setZero();
     for (int q_index = 0; q_index < nb_q_points_per_proc; q_index++) {
          int world_q_index = q_index + nb_q_points_per_proc * world_rank_;
          if (world_q_index >= nb_q_points)
               continue;
          for (int orb1 = 0; orb1 < temp_lattice_bubble.shape()[0]; orb1++) {
               for (int orb2 = 0; orb2 < temp_lattice_bubble.shape()[1]; orb2++) {
                    for (int l1 = 0; l1 < n_legendre; l1++) {
                         for (int l2 = 0; l2 < n_legendre; l2++) {
                              lattice_legendre_bubble_(orb1, orb2, l1, l2, q_index) =
                                   temp_lattice_bubble[orb1][orb2][l1][l2][current_bose_freq][q_index];
                         }
                    }
               }
          }
     }
}
