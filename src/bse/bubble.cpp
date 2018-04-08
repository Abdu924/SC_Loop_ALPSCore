#include "bubble.hpp"

using namespace std;
typedef boost::multi_array<complex<double> , 3> cplx_array_type;

Bubble::Bubble(alps::hdf5::archive &h5_archive,
                         boost::shared_ptr<Bandstructure> const &lattice_bs,
                         boost::shared_ptr<Selfenergy> const &sigma,
                         const alps::params& parms, complex<double> chemical_potential,
                         int world_rank)
     :lattice_bs_(lattice_bs), sigma_(sigma), chemical_potential(chemical_potential),
      world_rank_(world_rank) {
     int N_max = sigma_->get_n_matsubara_freqs();
     int N_Qmesh = parms["bseq.N_QBSEQ"];
     nb_q_points = lattice_bs_->get_nb_points_for_bseq();
     N_boson = parms["measurement.G2.n_bosonic_freq"];
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
               << std::endl << " q mesh based on " << N_Qmesh << " points, i.e. "
               << nb_q_points << " q points, and " 
               << N_boson << " bosonic frequencies" << std::endl;     
     // n_matsubara is parms[N_MATSUBARA]
     // FIXME check if this is consistent with N_max... 
     int n_matsubara = parms["N_MATSUBARA"];
     n_legendre = parms["measurement.G2.n_legendre"];
     legendre_trans_.reset(new LegendreTransformer(bubble_dim, n_legendre));
     raw_full_gf.resize(boost::extents
                        [per_site_orbital_size][per_site_orbital_size][N_max]);
     h5_archive["/legendre_gf/data"] >> raw_full_gf;     
     local_values_.resize(boost::extents[N_boson][bubble_dim]
                    [per_site_orbital_size][per_site_orbital_size]
                    [per_site_orbital_size][per_site_orbital_size]);
     std::fill(local_values_.origin(), local_values_.origin() + local_values_.num_elements(), 0.0);
     local_legendre_values_.resize(boost::extents[N_boson][n_legendre][n_legendre]
                                   [per_site_orbital_size][per_site_orbital_size]
                                   [per_site_orbital_size][per_site_orbital_size]);
     std::fill(local_legendre_values_.origin(), local_legendre_values_.origin() + local_legendre_values_.num_elements(), 0.0);
     lattice_values_.resize(boost::extents[N_boson][nb_q_points][bubble_dim]
                    [per_site_orbital_size][per_site_orbital_size]
                    [per_site_orbital_size][per_site_orbital_size]);
     std::fill(lattice_values_.origin(), lattice_values_.origin() + lattice_values_.num_elements(), 0.0);
     lattice_legendre_values_.resize(boost::extents[N_boson][nb_q_points][n_legendre][n_legendre]
                                     [per_site_orbital_size][per_site_orbital_size]
                                     [per_site_orbital_size][per_site_orbital_size]);
     std::fill(lattice_legendre_values_.origin(), lattice_legendre_values_.origin() + lattice_legendre_values_.num_elements(), 0.0);
     // The Eigen matrix objects are useful for the MPI gather operation
     // The boost multi array are used for hdf5 interface...
     world_lattice_bubble.clear();
     world_lattice_bubble.resize(N_boson);
     for (size_t boson_freq = 0; boson_freq < N_boson; boson_freq++) {
          world_lattice_bubble[boson_freq].resize(nb_q_points);
          for (size_t q_index = 0; q_index < nb_q_points; q_index++) {
               for (size_t freq_index = 0; freq_index < bubble_dim; freq_index++) {
                    world_lattice_bubble[boson_freq][q_index].push_back(
                         Eigen::MatrixXcd::Zero(n_sites * per_site_orbital_size * per_site_orbital_size,
                                                n_sites * per_site_orbital_size * per_site_orbital_size));
               }
          }
     }
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

void Bubble::dump_bubble_hdf5() {
     if (world_rank_ == 0) {
          std::string archive_name = bubble_hdf5_root + ".h5";
          alps::hdf5::archive bubble_output(archive_name, "a");
          if (dump_matsubara == 1) {
               // Local bubble
               std::string h5_group_name("/local_bubble");
               for (int site_index = 0; site_index < n_sites; site_index++) {
                    std::stringstream site_path;
                    site_path << h5_group_name + "/site_" +
                         boost::lexical_cast<std::string>(site_index) + "/data";
                    bubble_output[site_path.str()] << local_values_;
               }
               // Lattice bubble          
               std::string h5_group_name_2("/lattice_bubble");
               for (int site_index = 0; site_index < n_sites; site_index++) {
                    std::stringstream site_path;
                    site_path << h5_group_name_2 + "/site_" +
                         boost::lexical_cast<std::string>(site_index) + "/data";
                    bubble_output[site_path.str()] << lattice_values_;
               }
          }
          if (dump_legendre == 1) {          
               // Local bubble Legendre
               std::string h5_group_name("/legendre_local_bubble");
               for (int site_index = 0; site_index < n_sites; site_index++) {
                    std::stringstream site_path;
                    site_path << h5_group_name + "/site_" +
                         boost::lexical_cast<std::string>(site_index) + "/data";
                    bubble_output[site_path.str()] << local_legendre_values_;
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
					for (int hole_index_1 = 0;
					     hole_index_1 < orbital_size; hole_index_1++) {
					     local_values_[boson_index][freq_index][part_index_1][hole_index_2]
                                                  [part_index_2][hole_index_1] =
						  raw_full_gf[part_index_1][part_index_2][freq_index]
                                                  * raw_full_gf[hole_index_1][hole_index_2][freq_index + boson_index];
					} // hole_index_1
				   }  // part_index_2
			      }  // hole_index_2
			 }  // part_index_1
		    }  // site_index
	       } // freq_index
	  } // boson
	  std::cout << "local bubble time : " << std::endl;
          get_local_legendre_representation();
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
     // Only rank 0 has knowledge of the local bubble!
     if (world_rank_ == 0)
     {
          boost::timer::auto_cpu_timer local_legendre_rep;
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
                                             tmp_mat(n1, n1) = local_values_[boson_index][n1][orb1][orb2][orb3][orb4];
                                        }
                                        tmp_mat_leg = get_legendre_representation(tmp_mat);
                                        for (int l1 = 0; l1 < n_legendre; l1++) {
                                             for (int l2 = 0; l2 < n_legendre; l2++) {
                                                  local_legendre_values_[boson_index][l1][l2]
                                                       [orb1][orb2][orb3][orb4] = tmp_mat_leg(l1, l2);
                                             }
                                        }
                                   }
                              }
                         }
                    }
               }
          }
          std::cout << "local bubble legendre time : " << std::endl;
     } // world_rank_
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
		    MPI_Allreduce(partial_sum[q_index][freq_index].data(),
				  world_lattice_bubble[boson_index][q_index][freq_index].data(),
				  partial_sum[q_index][freq_index].size(),
				  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
                    for(int line_idx = 0; line_idx < per_site_orbital_size * per_site_orbital_size; ++line_idx) {
                         for(int col_idx = 0; col_idx < per_site_orbital_size * per_site_orbital_size; ++col_idx) {
                              int part_index_1 = line_idx / per_site_orbital_size;
                              int hole_index_2 = line_idx % per_site_orbital_size;
                              int part_index_2 = col_idx / per_site_orbital_size;
                              int hole_index_1 = col_idx % per_site_orbital_size;
                              lattice_values_[boson_index][q_index][freq_index]
                                   [part_index_1][hole_index_2]
                                   [part_index_2][hole_index_1] =
                                   world_lattice_bubble[boson_index][q_index][freq_index](line_idx, col_idx);
                         }
                    }
	       }
	  }
	  std::cout << "Time for boson freq " << boson_index
		    << ": " << std::endl;
     } // boson
}

void Bubble::get_lattice_legendre_representation() {
     if (world_rank_ == 0)
     {
          boost::timer::auto_cpu_timer lattice_legendre_rep;
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
                                             tmp_mat(n1, n1) = local_values_[boson_index][n1][orb1][orb2][orb3][orb4];
                                        }
                                        tmp_mat_leg = get_legendre_representation(tmp_mat);
                                        for (int l1 = 0; l1 < n_legendre; l1++) {
                                             for (int l2 = 0; l2 < n_legendre; l2++) {
                                                  local_legendre_values_[boson_index][l1][l2]
                                                       [orb1][orb2][orb3][orb4] = tmp_mat_leg(l1, l2);
                                             }
                                        }
                                   }
                              }
                         }
                    }
               }
          }
          std::cout << "local bubble legendre time : " << std::endl;
     } // world_rank_
     MPI_Barrier(MPI_COMM_WORLD);
}
     
const std::string Bubble::bubble_hdf5_root = "c_bubble_new";
