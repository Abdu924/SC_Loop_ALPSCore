#include "bubble.hpp"

using namespace std;
typedef boost::multi_array_types::index_range range;

Bubble::Bubble(alps::hdf5::archive &h5_archive,
                         boost::shared_ptr<Bandstructure> const &lattice_bs,
                         boost::shared_ptr<Selfenergy> const &sigma,
                         const alps::params& parms, complex<double> chemical_potential,
                         int world_rank)
     :lattice_bs_(lattice_bs), sigma_(sigma), chemical_potential(chemical_potential),
      world_rank_(world_rank) {
     int N_max = sigma_->get_n_matsubara_freqs();
     nb_q_points = lattice_bs_->get_nb_points_for_bseq();
     if (world_rank == 0) {
          MPI_Comm_size(MPI_COMM_WORLD, &world_size);
          nb_q_points_per_proc = nb_q_points / world_size;
          if (nb_q_points % world_size > 0) {
               nb_q_points_per_proc += 1;
          }
     }
     MPI_Bcast(&world_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&nb_q_points_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
     N_boson = parms["bseq.bubbles.n_bosonic_freq"];
     n_legendre = parms["bseq.bubbles.n_legendre"];
     assert(n_legendre <= parms["measurement.G2.n_legendre"]);
     dump_matsubara = parms["bseq.bubbles.dump_matsubara"];
     dump_legendre = parms["bseq.bubbles.dump_legendre"];
     bubble_dim = parms.exists("bseq.N_NU_BSEQ") ? parms["bseq.N_NU_BSEQ"] : N_max - N_boson;
     n_sites = sigma_->get_n_sites();
     if (n_sites > 1) {
          cout << "More than 1 site - not supported . " << endl;
          throw runtime_error("More than one site is not supported !");
     }
     per_site_orbital_size = sigma_->get_per_site_orbital_size();
     tot_orbital_size = n_sites * per_site_orbital_size;
     std::cout << "Computing bubbles for " << bubble_dim << " fermionic frequencies, "
               << std::endl << " q mesh based on " << nb_q_points << " points, i.e. "
               << nb_q_points << " q points, and " 
               << N_boson << " bosonic frequencies" << std::endl;     
     // n_matsubara is parms[N_MATSUBARA]
     // FIXME check if this is consistent with N_max... 
     int n_matsubara = parms["N_MATSUBARA"];
     legendre_trans_.reset(new LegendreTransformer(bubble_dim, n_legendre));
     flavor_trans_.reset(new FlavorTransformer());
     raw_full_gf.resize(boost::extents
                        [per_site_orbital_size][per_site_orbital_size][N_max]);
     h5_archive["/legendre_gf/data"] >> raw_full_gf;
     local_values_.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                          [per_site_orbital_size][per_site_orbital_size]
                          [bubble_dim][N_boson]);
     std::fill(local_values_.origin(), local_values_.origin() + local_values_.num_elements(), 0.0);
     local_legendre_values_.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                                   [per_site_orbital_size][per_site_orbital_size]
                                   [n_legendre][n_legendre][N_boson]);
     std::fill(local_legendre_values_.origin(), local_legendre_values_.origin() + local_legendre_values_.num_elements(), 0.0);
     lattice_values_.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                            [per_site_orbital_size][per_site_orbital_size]
                            [bubble_dim][N_boson][nb_q_points]);
     std::fill(lattice_values_.origin(), lattice_values_.origin() + lattice_values_.num_elements(), 0.0);
     lattice_legendre_values_.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                                     [per_site_orbital_size][per_site_orbital_size]
                                     [n_legendre][n_legendre][N_boson][nb_q_points_per_proc]);
     std::fill(lattice_legendre_values_.origin(), lattice_legendre_values_.origin() + lattice_legendre_values_.num_elements(), 0.0);
     world_lattice_legendre_values_.resize(boost::extents[per_site_orbital_size][per_site_orbital_size]
                                           [per_site_orbital_size][per_site_orbital_size]
                                           [n_legendre][n_legendre][N_boson][nb_q_points_per_proc * world_size]);
     std::fill(world_lattice_legendre_values_.origin(),
               world_lattice_legendre_values_.origin() + world_lattice_legendre_values_.num_elements(), 0.0);
}

std::vector<Eigen::MatrixXcd> Bubble::get_greens_function(Eigen::Ref<Eigen::VectorXd> k_point,
                                                          int boson_index) {
     size_t N_max = sigma_->get_n_matsubara_freqs();
     std::vector<Eigen::MatrixXcd> output;
     output.clear();
     output.resize(N_max);
     Eigen::MatrixXcd inverse_gf(tot_orbital_size, tot_orbital_size);
     for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	  Eigen::VectorXcd mu_plus_iomega = Eigen::VectorXcd::Constant
	       (tot_orbital_size, chemical_potential +
		sigma_->get_matsubara_frequency(freq_index));
	  Eigen::MatrixXcd self_E = sigma_->values_[freq_index];
	  inverse_gf = -lattice_bs_->get_k_basis_matrix(k_point) - self_E;
	  inverse_gf.diagonal() += mu_plus_iomega;
	  output[freq_index] = inverse_gf.inverse();
     }
     return output;
}

void Bubble::dump_bubble_hdf5(const alps::params& parms) {
     if (world_rank_ == 0) {
          int line_idx, col_idx;
          const string archive_name = parms["bseq.bubbles.filename"].as<string>();
          alps::hdf5::archive bubble_output(archive_name, "a");
          if (dump_matsubara == 1) {
               // // Lattice bubble          
               std::string h5_group_name("/lattice_bubble_compact");
               // Lattice bubble, new flavor order
               boost::multi_array<std::complex<double>, 4> lattice_values_compact;
               lattice_values_compact.resize(boost::extents
                                             [lattice_values_.shape()[0] * lattice_values_.shape()[1]]
                                             [lattice_values_.shape()[2] * lattice_values_.shape()[3]]
                                             [lattice_values_.shape()[4]] // fermionic matsu, diagonal
                                             [lattice_values_.shape()[6]]);  // q index
               for (int nb = 0; nb < lattice_values_.shape()[5]; nb++) {
                    std::fill(lattice_values_compact.origin(),
                              lattice_values_compact.origin() + lattice_values_compact.num_elements(), 0.0);
                    for (int orb1 = 0; orb1 < lattice_values_.shape()[0]; orb1++) {
                         for (int orb2 = 0; orb2 < lattice_values_.shape()[1]; orb2++) {
                              for (int orb3 = 0; orb3 < lattice_values_.shape()[2]; orb3++) {
                                   for (int orb4 = 0; orb4 < lattice_values_.shape()[3]; orb4++) {
                                        line_idx = flavor_trans_->get_bubble_line_from_pair(orb1, orb2);
                                        col_idx = flavor_trans_->get_bubble_col_from_pair(orb3, orb4);
                                        for (int nf = 0; nf < lattice_values_.shape()[4]; nf++) {
                                             for (int nq = 0; nq < lattice_values_.shape()[6]; nq++) {
                                                  lattice_values_compact[line_idx][col_idx][nf][nq] =
                                                       lattice_values_[orb1][orb2][orb3][orb4][nf][nb][nq];
                                             }
                                        }
                                   }
                              }
                         }
                    }
                    for (int site_index = 0; site_index < n_sites; site_index++) {
                         std::stringstream site_path;
                         site_path << h5_group_name + "/site_" +
                              boost::lexical_cast<std::string>(site_index) + "/bose_" +
                              boost::lexical_cast<std::string>(nb) + "/data";
                         bubble_output[site_path.str()] << lattice_values_compact;
                    }
               }
          }
          if (dump_legendre == 1) {          
               // Local bubble Legendre
               std::string h5_group_name("/legendre_local_bubble_compact");
               boost::multi_array<std::complex<double>, 4> local_values_compact;
               local_values_compact.resize(boost::extents
                                           [local_legendre_values_.shape()[0] * local_legendre_values_.shape()[1]]
                                           [local_legendre_values_.shape()[2] * local_legendre_values_.shape()[3]]
                                           [local_legendre_values_.shape()[4]] // l1
                                           [local_legendre_values_.shape()[5]]); // l1
                    for (int nb = 0; nb < local_legendre_values_.shape()[6]; nb++) {
                         std::fill(local_values_compact.origin(),
                                   local_values_compact.origin() + local_values_compact.num_elements(), 0.0);
                         for (int orb1 = 0; orb1 < local_legendre_values_.shape()[0]; orb1++) {
                              for (int orb2 = 0; orb2 < local_legendre_values_.shape()[1]; orb2++) {
                                   for (int orb3 = 0; orb3 < local_legendre_values_.shape()[2]; orb3++) {
                                        for (int orb4 = 0; orb4 < local_legendre_values_.shape()[3]; orb4++) {
                                             line_idx = flavor_trans_->get_bubble_line_from_pair(orb1, orb2);
                                             col_idx = flavor_trans_->get_bubble_col_from_pair(orb3, orb4);
                                             for (int l1 = 0; l1 < local_legendre_values_.shape()[4]; l1++) {
                                                  for (int l2 = 0; l2 < local_legendre_values_.shape()[5]; l2++) {
                                                       local_values_compact[line_idx][col_idx][l1][l2] =
                                                            local_legendre_values_[orb1][orb2][orb3][orb4][l1][l2][nb];
                                                  }
                                             }
                                        }
                                   }
                              }
                         }
                         for (int site_index = 0; site_index < n_sites; site_index++) {
                              std::stringstream site_path;
                              site_path << h5_group_name + "/site_" +
                                   boost::lexical_cast<std::string>(site_index) + "/bose_" +
                                   boost::lexical_cast<std::string>(nb) + "/data";
                              bubble_output[site_path.str()] << local_values_compact;
                         }
                    }
               std::string h5_group_name_2("/legendre_lattice_bubble_compact");
               // Lattice bubble, new flavor order
               boost::multi_array<std::complex<double>, 5> lattice_values_compact;
               lattice_values_compact.resize(boost::extents
                                             [world_lattice_legendre_values_.shape()[0] * world_lattice_legendre_values_.shape()[1]]
                                             [world_lattice_legendre_values_.shape()[2] * world_lattice_legendre_values_.shape()[3]]
                                             [world_lattice_legendre_values_.shape()[4]] // l1
                                             [world_lattice_legendre_values_.shape()[5]]  // l2
                                             [nb_q_points]);               // q index
               for (int nb = 0; nb < world_lattice_legendre_values_.shape()[6]; nb++) {
                    for (int orb1 = 0; orb1 < world_lattice_legendre_values_.shape()[0]; orb1++) {
                         for (int orb2 = 0; orb2 < world_lattice_legendre_values_.shape()[1]; orb2++) {
                              for (int orb3 = 0; orb3 < world_lattice_legendre_values_.shape()[2]; orb3++) {
                                   for (int orb4 = 0; orb4 < world_lattice_legendre_values_.shape()[3]; orb4++) {
                                        line_idx = flavor_trans_->get_bubble_line_from_pair(orb1, orb2);
                                        col_idx = flavor_trans_->get_bubble_col_from_pair(orb3, orb4);
                                        for (int l1 = 0; l1 < world_lattice_legendre_values_.shape()[4]; l1++) {
                                             for (int l2 = 0; l2 < world_lattice_legendre_values_.shape()[5]; l2++) {
                                                  for (int nq = 0; nq < nb_q_points; nq++) {
                                                       lattice_values_compact[line_idx][col_idx][l1][l2][nq] =
                                                            world_lattice_legendre_values_[orb1][orb2][orb3][orb4][l1][l2][nb][nq];
                                                  }
                                             }
                                        }
                                   }
                              }
                         }
                    }
                    for (int site_index = 0; site_index < n_sites; site_index++) {
                         std::stringstream site_path;
                         site_path << h5_group_name_2 + "/site_" +
                              boost::lexical_cast<std::string>(site_index) + "/bose_" +
                              boost::lexical_cast<std::string>(nb) + "/data";
                         bubble_output[site_path.str()] << lattice_values_compact;
                    }
               }
          }
          // q point list
          std::string h5_group_name("/lattice_bubble/q_point_list");
          std::vector<std::complex<double>> temp_data;
          temp_data.resize(nb_q_points);
          Eigen::VectorXd q_point;
          for (int q_index = 0; q_index < nb_q_points; q_index++) {
               q_point = lattice_bs_->get_q_point(q_index);
               temp_data[q_index] = std::complex<double>(q_point(0), q_point(1));
          }
          bubble_output[h5_group_name] = temp_data;
          // Close file
	  bubble_output.close();
     }
     MPI_Barrier(MPI_COMM_WORLD);
}

void Bubble::compute_local_bubble() {     
     if (world_rank_ == 0)
     {
	  cout << "***********************************************" << endl;
	  cout << "** LOCAL BUBBLE CALCULATION                 ***" << endl;
	  cout << "***********************************************" << endl << endl;
	  boost::timer::auto_cpu_timer bubble_calc;
	  int orbital_size = per_site_orbital_size;
	  for (int boson_index = 0; boson_index < N_boson; boson_index++) {
	       for (int freq_index = 0; freq_index < bubble_dim; freq_index++) {
		    for(size_t site_index = 0; site_index < n_sites; site_index++) {
			 for (int part_index_1 = 0; part_index_1 < orbital_size;
			      part_index_1++) {
			      for (int hole_index_2 = 0; hole_index_2 < orbital_size;
				   hole_index_2++) {
				   for (int part_index_2 = 0; part_index_2 < orbital_size;
					part_index_2++) {
					for (int hole_index_1 = 0; hole_index_1 < orbital_size; hole_index_1++) {
                                             if ((boson_index == 0) and
                                                 (freq_index == 0) and
                                                 (part_index_1 == 2)
                                                 and (hole_index_2 == 3)
                                                 and (part_index_2 == 0)
                                                 and (hole_index_1 == 1)) {
                                             }
					     local_values_[part_index_1][hole_index_2]
                                                  [part_index_2][hole_index_1][freq_index][boson_index] =
						  raw_full_gf[part_index_1][part_index_2][freq_index]
                                                  * raw_full_gf[hole_index_1][hole_index_2][freq_index + boson_index];
					} // hole_index_1
				   }  // part_index_2
			      }  // hole_index_2
			 }  // part_index_1
		    }  // site_index
	       } // freq_index
	  } // boson
          get_local_legendre_representation();
          std::cout << "local bubble time : " << std::endl;
     } // world_rank_
     MPI_Barrier(MPI_COMM_WORLD);
}

Eigen::MatrixXcd Bubble::get_legendre_representation(Eigen::Ref<Eigen::MatrixXcd> matsu_data) {
     const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &Tnl(legendre_trans_->Tnl());
     const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &Tnl_neg(legendre_trans_->Tnl_neg());
     Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> tmp_mat_neg(bubble_dim, bubble_dim),
          tmp_mat_leg(n_legendre, n_legendre);
     tmp_mat_neg = Eigen::MatrixXcd::Zero(bubble_dim, bubble_dim);
     // The negative freq contribution
     tmp_mat_neg = matsu_data.conjugate();
     tmp_mat_leg = Tnl.adjoint() * (matsu_data * Tnl) +
          Tnl_neg.adjoint() * (tmp_mat_neg * Tnl_neg);
     return tmp_mat_leg;
}

void Bubble::get_local_legendre_representation() {
     // NOte: only rank 0 has knowledge of the local bubble!
     Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> tmp_mat(bubble_dim, bubble_dim),
          tmp_mat_leg(n_legendre, n_legendre);
     const	int orbital_size = per_site_orbital_size;
     tmp_mat = Eigen::MatrixXcd::Zero(bubble_dim, bubble_dim);
     for (int boson_index = 0; boson_index < N_boson; boson_index++) {
          for(size_t site_index = 0; site_index < n_sites; site_index++) {
               for(int orb1 = 0; orb1 < orbital_size; orb1++) {
                    for(int orb2 = 0; orb2 < orbital_size; orb2++) {
                         for(int orb3 = 0; orb3 < orbital_size; orb3++) {
                              for(int orb4 = 0; orb4 < orbital_size; orb4++) {
                                   for (int n1 = 0; n1 < bubble_dim; n1++) {
                                        tmp_mat(n1, n1) = local_values_[orb1][orb2][orb3][orb4][n1][boson_index];
                                   }
                                   tmp_mat_leg = get_legendre_representation(tmp_mat);
                                   for (int l1 = 0; l1 < n_legendre; l1++) {
                                        for (int l2 = 0; l2 < n_legendre; l2++) {
                                             local_legendre_values_[orb1][orb2][orb3][orb4][l1][l2][boson_index] =
                                                  tmp_mat_leg(l1, l2);
                                        }
                                   }
                              }
                         }
                    }
               }
          }
     }
}

void Bubble::compute_lattice_bubble() {
     if (world_rank_ == 0)
     {
	  cout << "***********************************************" << endl;
	  cout << "** LATTICE BUBBLE CALCULATION               ***" << endl;
	  cout << "***********************************************" << endl << endl;
     }
     boost::timer::auto_cpu_timer lattice_bubble_calc;
     size_t k_min(0);
     size_t k_max(lattice_bs_->get_lattice_size());
     int orbital_size(per_site_orbital_size);
     int new_i(0);
     int new_j(0);
     std::vector<std::vector<Eigen::MatrixXcd> > partial_sum;
     lattice_bubble.clear();
     lattice_bubble.resize(N_boson);
     std::vector<Eigen::MatrixXcd> gf_kq, gf_k;
     int block_size = per_site_orbital_size * per_site_orbital_size;
     for (int boson_index = 0; boson_index < N_boson; boson_index++) {
	  boost::timer::auto_cpu_timer boson_calc; 
	  partial_sum.clear();
	  partial_sum.resize(nb_q_points);
	  for(int q_index = 0; q_index < nb_q_points; q_index++) {
	       for(int freq_index = 0; freq_index < bubble_dim; freq_index++) {
		    partial_sum[q_index].push_back(
			 Eigen::MatrixXcd::Zero(n_sites * per_site_orbital_size * per_site_orbital_size,
						n_sites * per_site_orbital_size * per_site_orbital_size));
	       }
	  }
	  lattice_bubble[boson_index].resize(nb_q_points);
	  for (int k_index = k_min; k_index < k_max; k_index++) {
	       double l_weight = lattice_bs_->get_weight(k_index);
	       if (abs(l_weight) < 1e-6) {
		    if (world_rank_ == 0) {
			 cout << "skipping k point in lattice bubble" << endl;
		    }
		    continue;
	       } else {
		    Eigen::VectorXd k_point = lattice_bs_->get_k_point(k_index);
		    gf_k = get_greens_function(k_point, boson_index);
		    for(int q_index = 0; q_index < nb_q_points; q_index++) {
			 Eigen::VectorXd k_plus_q_point = lattice_bs_->get_k_plus_q_point(k_index, q_index);
			 gf_kq = get_greens_function(k_plus_q_point, boson_index);
			 for (int freq_index = 0; freq_index < bubble_dim; freq_index++) {
			      for(size_t site_index = 0; site_index < n_sites;
				  site_index++) {
				   // block start for full system greens function
				   int block_index = site_index * per_site_orbital_size * per_site_orbital_size;
				   for (int part_index_1 = 0; part_index_1 < orbital_size;
					part_index_1++) {
					for (int hole_index_2 = 0; hole_index_2 < orbital_size;
					     hole_index_2++) {
					     for (int part_index_2 = 0; part_index_2 < orbital_size;
						  part_index_2++) {
						  for (int hole_index_1 = 0; hole_index_1 < orbital_size;
						       hole_index_1++) {
						       new_i = part_index_1 *
							    orbital_size + hole_index_2;
						       new_j = part_index_2 *
							    orbital_size + hole_index_1;
						       //Careful here: greens functions are based
						       //on the full orbital dimension of the
						       // system
						       partial_sum[q_index][freq_index].block(
							    block_index, block_index,
							    block_size, block_size)(new_i, new_j) +=
							    l_weight *
							    gf_k[freq_index].block(
								 block_index, block_index,
								 block_size, block_size)(
								      part_index_1, part_index_2) *
							    gf_kq[freq_index + boson_index].block(
								 block_index, block_index,
								 block_size, block_size)(
								      hole_index_1, hole_index_2);
						  } // hole_index_1
					     }  // part_index_2
					}  // hole_index_2
				   }  // part_index_1
			      }  // site_index
			 } // freq
		    } // q_index
	       } // if weight
	  } // k_index
	  for (int q_index = 0; q_index < nb_q_points; q_index++) {
	       for (int freq_index = 0; freq_index < bubble_dim; freq_index++) {
                    Eigen::MatrixXcd tmp_lattice_bubble;
                    tmp_lattice_bubble = Eigen::MatrixXcd::Zero(n_sites * per_site_orbital_size * per_site_orbital_size,
                                                                n_sites * per_site_orbital_size * per_site_orbital_size);
		    MPI_Allreduce(partial_sum[q_index][freq_index].data(),
				  tmp_lattice_bubble.data(),
				  partial_sum[q_index][freq_index].size(),
				  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
                    for(int line_idx = 0; line_idx < per_site_orbital_size * per_site_orbital_size; ++line_idx) {
                         for(int col_idx = 0; col_idx < per_site_orbital_size * per_site_orbital_size; ++col_idx) {
                              int part_index_1 = line_idx / per_site_orbital_size;
                              int hole_index_2 = line_idx % per_site_orbital_size;
                              int part_index_2 = col_idx / per_site_orbital_size;
                              int hole_index_1 = col_idx % per_site_orbital_size;
                              lattice_values_[part_index_1][hole_index_2]
                                   [part_index_2][hole_index_1][freq_index][boson_index][q_index] =
                                   tmp_lattice_bubble(line_idx, col_idx);
                         }
                    }
               }
	  }
	  std::cout << "Time for boson freq " << boson_index
		    << ": " << std::endl;
     } // boson
     get_lattice_legendre_representation();
}

void Bubble::get_lattice_legendre_representation() {
     if (world_rank_ == 0) {
          cout << "***********************************************" << endl;
          cout << "** LATTICE BUBBLE LEGENDRE REP              ***" << endl;
          cout << "***********************************************" << endl << endl;
     }
     int orbital_size(per_site_orbital_size);
     Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> tmp_mat(bubble_dim, bubble_dim),
          tmp_mat_leg(n_legendre, n_legendre), tmp_mpi_mat(n_legendre, n_legendre);
     for (int boson_index = 0; boson_index < N_boson; boson_index++) {
          std::cout << "boson freq " << boson_index << std::endl;               
          boost::timer::auto_cpu_timer lattice_bubble_leg;
          for (int q_index = 0; q_index < nb_q_points_per_proc; q_index++) {
               int world_q_index = q_index + nb_q_points_per_proc * world_rank_;
               if (world_q_index >= nb_q_points)
                    continue;
               std::cout << "world_q_index " << world_q_index << std::endl;
               for(int orb1 = 0; orb1 < orbital_size; orb1++) {
                    for(int orb2 = 0; orb2 < orbital_size; orb2++) {
                         for(int orb3 = 0; orb3 < orbital_size; orb3++) {
                              for(int orb4 = 0; orb4 < orbital_size; orb4++) {
                                   tmp_mat = Eigen::MatrixXcd::Zero(bubble_dim, bubble_dim);
                                   for (int n1 = 0; n1 < bubble_dim; n1++) {
                                        tmp_mat(n1, n1) = lattice_values_[orb1][orb2][orb3][orb4]
                                             [n1][boson_index][world_q_index];
                                   }
                                   tmp_mat_leg = get_legendre_representation(tmp_mat);
                                   for (int l1 = 0; l1 < n_legendre; l1++) {
                                        for (int l2 = 0; l2 < n_legendre; l2++) {
                                             lattice_legendre_values_[orb1][orb2][orb3][orb4]
                                                  [l1][l2][boson_index][q_index] = tmp_mat_leg(l1, l2);
                                        }
                                   }
                              }
                         }
                    }
               }
          }
          std::cout << "Time for boson freq " << boson_index << ": " << std::endl;
     }
     // MPI_Barrier(MPI_COMM_WORLD);     
     // Gather all results
     for (int boson_index = 0; boson_index < N_boson; boson_index++) {
          for(int orb1 = 0; orb1 < orbital_size; orb1++) {
               for(int orb2 = 0; orb2 < orbital_size; orb2++) {
                    for(int orb3 = 0; orb3 < orbital_size; orb3++) {
                         for(int orb4 = 0; orb4 < orbital_size; orb4++) {
                              for (int q_index = 0; q_index < nb_q_points_per_proc; q_index++) {
                                   for (int l1 = 0; l1 < n_legendre; l1++) {
                                        for (int l2 = 0; l2 < n_legendre; l2++) {
                                             tmp_mat_leg(l1, l2) =
                                                  lattice_legendre_values_[orb1][orb2][orb3][orb4]
                                                  [l1][l2][boson_index][q_index];
                                        }
                                   }
                                   if (world_rank_ == 0) {
                                        for (int proc_index = 1; proc_index < world_size; proc_index++) {
                                             MPI_Recv(
                                                  tmp_mpi_mat.data(), tmp_mpi_mat.size(),
                                                  MPI_DOUBLE_COMPLEX, proc_index, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                             int world_q_index = q_index + nb_q_points_per_proc * proc_index;
                                             for (int l1 = 0; l1 < n_legendre; l1++) {
                                                  for (int l2 = 0; l2 < n_legendre; l2++) {
                                                       world_lattice_legendre_values_[orb1][orb2][orb3][orb4]
                                                            [l1][l2][boson_index][world_q_index] =
                                                            tmp_mpi_mat(l1, l2);
                                                  }
                                             }
                                        }
                                   } else {
                                        MPI_Send(tmp_mat_leg.data(), tmp_mat_leg.size(),
                                                 MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);	       
                                   }
                                   if (world_rank_ == 0) {
                                        for (int l1 = 0; l1 < n_legendre; l1++) {
                                             for (int l2 = 0; l2 < n_legendre; l2++) {
                                                  world_lattice_legendre_values_[orb1][orb2][orb3][orb4]
                                                       [l1][l2][boson_index][q_index] =
                                                       tmp_mat_leg(l1, l2);
                                             }
                                        }
                                   }
                              }
                         }
                    }
               }
          }
     }
}
