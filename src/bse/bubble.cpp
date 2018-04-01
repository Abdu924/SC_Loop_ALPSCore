#include "bubble.hpp"

using namespace std;
typedef boost::multi_array<complex<double> , 3> cplx_array_type;

LatticeBubble::LatticeBubble() {
     values_.resize(boost::extents[n_orbitals][n_orbitals][n_orbitals][n_orbitals][N_l_G4][N_l_G4][N_boson]);
}

LocalBubble::LocalBubble(alps::hdf5::archive &h5_archive,
                         boost::shared_ptr<Bandstructure> const &lattice_bs,
                         boost::shared_ptr<Selfenergy> const &sigma,
                         const alps::params& parms, complex<double> chemical_potential,
                         int world_rank)
     :lattice_bs_(lattice_bs), sigma_(sigma), chemical_potential(chemical_potential),
      world_rank_(world_rank) {
     int N_max = sigma_->get_n_matsubara_freqs();
     int N_Qmesh = parms["bseq.N_QBSEQ"];
     N_boson = parms["measurement.G2.n_bosonic_freq"];
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
               << lattice_bs->get_nb_points_for_bseq() << " q points, and " 
               << N_boson << " bosonic frequencies" << std::endl;

     
     world_local_gf.clear();
     for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
          world_local_gf.push_back(Eigen::MatrixXcd::Zero(
                                        tot_orbital_size, tot_orbital_size));
     }

     // n_matsubara is parms[N_MATSUBARA]
     // FIXME check if this is consistent with N_max... 
     int n_matsubara = parms["N_MATSUBARA"];
     raw_full_gf.resize(boost::extents
                        [per_site_orbital_size][per_site_orbital_size][N_max]);
     h5_archive["/legendre_gf/data"] >> raw_full_gf;     
     values_.resize(boost::extents[N_boson][bubble_dim]
                    [n_sites * per_site_orbital_size * per_site_orbital_size]
                    [n_sites * per_site_orbital_size * per_site_orbital_size]);
     std::fill(values_.origin(), values_.origin() + values_.num_elements(), 0.0);
}

void LocalBubble::dump_bubble_hdf5() {
     if (world_rank_ == 0) {
	  std::string archive_name = bubble_hdf5_root + ".h5";
	  alps::hdf5::archive bubble_output(archive_name, "a");
	  std::string h5_group_name("/local_bubble");
	  for (int site_index = 0; site_index < n_sites; site_index++) {
               std::stringstream site_path;
               site_path << h5_group_name + "/site_" +
                    boost::lexical_cast<std::string>(site_index) + "/data";
               bubble_output[site_path.str()] << values_;
	  }
	  // std::string h5_group_name_2("/lattice_bubble");
	  // for (int site_index = 0; site_index < n_sites; site_index++) {
	  //      for(int boson_index = 0; boson_index < N_boson; boson_index++) {
	  //           for(int q_index = 0;
	  //       	q_index < lattice_bs_->get_nb_points_for_bseq(); q_index++) {
	  //       	 std::stringstream site_path;
	  //       	 site_path <<
	  //       	      h5_group_name_2 +
	  //       	      "/site_" + boost::lexical_cast<std::string>(site_index) + "/" +
	  //       	      "boson_" + boost::lexical_cast<std::string>(boson_index) + "/" +
	  //       	      "q_" + boost::lexical_cast<std::string>(q_index) + "/";
	  //       	 for(int line_idx = 0;
	  //       	     line_idx < per_site_orbital_size * per_site_orbital_size;
	  //       	     ++line_idx) {
	  //       	      for(int col_idx = 0;
	  //       		  col_idx < per_site_orbital_size * per_site_orbital_size;
	  //       		  ++col_idx) {
	  //       		   std::stringstream orbital_path;
	  //       		   int part_index_1 = line_idx / per_site_orbital_size;
	  //       		   int hole_index_2 = line_idx % per_site_orbital_size;
	  //       		   int part_index_2 = col_idx / per_site_orbital_size;
	  //       		   int hole_index_1 = col_idx % per_site_orbital_size;
	  //       		   orbital_path << site_path.str() <<
	  //       			boost::lexical_cast<std::string>(part_index_1) + "/"
	  //       			+ boost::lexical_cast<std::string>(hole_index_2) + "/"
	  //       			+ boost::lexical_cast<std::string>(part_index_2) + "/"
	  //       			+ boost::lexical_cast<std::string>(hole_index_1) + "/value";
	  //       		   std::vector<std::complex<double>> temp_data;
	  //       		   temp_data.resize(bubble_dim);
	  //       		   for (int freq_index = 0; freq_index < bubble_dim; freq_index++) {
	  //       			temp_data[freq_index] =
	  //       			     world_lattice_bubble[boson_index][q_index][freq_index].block(
	  //       				  site_index * per_site_orbital_size *
	  //       				  per_site_orbital_size,
	  //       				  site_index * per_site_orbital_size *
	  //       				  per_site_orbital_size,
	  //       				  per_site_orbital_size *
	  //       				  per_site_orbital_size,
	  //       				  per_site_orbital_size *
	  //       				  per_site_orbital_size)(line_idx, col_idx);
	  //       		   }
	  //       		   bubble_output << alps::make_pvp(orbital_path.str(), temp_data);
	  //       	      }
	  //       	 }
	  //           }
	  //      }
	  // }
	  // h5_group_name_2 = "/lattice_bubble/q_point_list";
	  // std::vector<std::complex<double>> temp_data;
	  // temp_data.resize(lattice_bs_->get_nb_points_for_bseq());
	  // int nb_q_points = lattice_bs_->get_nb_points_for_bseq();
	  // Eigen::VectorXd q_point;
	  // for (int q_index = 0; q_index < nb_q_points; q_index++) {
	  //      q_point = lattice_bs_->get_q_point(q_index);
	  //      temp_data[q_index] = std::complex<double>(q_point(0), q_point(1));
	  // }
	  // bubble_output << alps::make_pvp(h5_group_name_2, temp_data);
	  bubble_output.close();
     }
     MPI_Barrier(MPI_COMM_WORLD);
}


void LocalBubble::compute_local_bubble() {

     // NEED to CHECK thiS!!!!
     // TODO FIXME
     // if (compute_bubble) {
     //      // We are interested in the linear response
     //      // in the paramagentic phase, thus we do
     //      // not need to separate spins anymore.
     //      world_local_gf[freq_index].block(
     //           site_index * per_site_orbital_size,
     //           site_index * per_site_orbital_size,
     //           per_site_orbital_size,
     //           per_site_orbital_size) =
     //           world_local_greens_function.block(
     //                site_index * per_site_orbital_size,
     //                site_index * per_site_orbital_size,
     //                per_site_orbital_size,
     //                per_site_orbital_size);
     // }
     
     if (world_rank_ == 0)
     {
	  cout << "***********************************************" << endl;
	  cout << "** LOCAL BUBBLE CALCULATION                 ***" << endl;
	  cout << "***********************************************" << endl << endl;
	  boost::timer::auto_cpu_timer bubble_calc;
	  int orbital_size(lattice_bs_->get_orbital_size());
	  int new_i(0);
	  int new_j(0);
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
					     new_i = part_index_1 * orbital_size + hole_index_2;
					     new_j = part_index_2 * orbital_size + hole_index_1;
					     values_[boson_index][freq_index][new_i][new_j] =
						  raw_full_gf[part_index_1][part_index_2][freq_index]
                                                  * raw_full_gf[hole_index_1][hole_index_2][freq_index];
					} // hole_index_1
				   }  // part_index_2
			      }  // hole_index_2
			 }  // part_index_1
		    }  // site_index
	       } // freq_index
	  } // boson
	  std::cout << "local bubble time : " << std::endl;
     } // world_rank_
     MPI_Barrier(MPI_COMM_WORLD);
}

const std::string LocalBubble::bubble_hdf5_root = "c_bubble";
