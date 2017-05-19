#include "gf_base.hpp"

GfBase::GfBase(const alps::params &parms, int world_rank,
	       boost::shared_ptr<Chemicalpotential> chempot,
	       boost::shared_ptr<Bandstructure> const &lattice_bs):
     world_rank_(world_rank), chempot_(chempot), lattice_bs_(lattice_bs) {
     read_params(parms);
}

GfBase::GfBase(const alps::params &parms, int world_rank):world_rank_(world_rank) {
     read_params(parms);
}

void GfBase::read_params(const alps::params &parms) {
     n_blocks = static_cast<size_t>(parms["N_BLOCKS"]);
     n_sites = parms.exists("N_SITES") ?
	  static_cast<size_t>(parms["N_SITES"]) : 1;
     per_site_orbital_size = parms.exists("N_ORBITALS") ?
	  static_cast<size_t>(parms["N_ORBITALS"]) : 2;
     // Careful - this is the actual number of Matsubara frequencies.
     // not the actual value of N for the max value of omega such that
     // max_freq = 2N + 1 pi / beta, because n = 0 is also a frequency,
     // so that n_matsubara = N + 1
     n_matsubara = parms["N_MATSUBARA"];
     beta = static_cast<double>(parms["BETA"]);
     tot_orbital_size = per_site_orbital_size * n_sites;
     matsubara_frequencies_ = Eigen::VectorXcd::Zero(n_matsubara);
     for (size_t freq_index = 0; freq_index < n_matsubara; freq_index++) {
	  matsubara_frequencies_(freq_index) =
	       complex<double>(0.0, (2.0 * freq_index + 1) * M_PI / beta);
     }
     interaction_matrix =
	  Eigen::MatrixXcd::Constant(tot_orbital_size, tot_orbital_size, 0.0);
     site_symmetry_matrix = Eigen::MatrixXcd::Constant(tot_orbital_size, tot_orbital_size, 1.0);
     density_density_correl = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     a_dagger_b = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     if (n_blocks < per_site_orbital_size) {
	  std::string base_name = "BLOCK_";
	  std::string blocks_file_name = parms["BLOCKS"];
	  alps::hdf5::archive ar(blocks_file_name, alps::hdf5::archive::READ);
	  blocks.resize(n_blocks, std::vector<size_t>());
	  for(std::size_t i=0; i < blocks.size(); ++i) {
	       std::string block_name;
	       block_name = base_name + boost::lexical_cast<std::string>(i);
	       blocks[i].resize(4, 0);
	       ar >> alps::make_pvp(block_name, blocks[i]);
	  }
     } else {
	  // diagonal hopping - equivalent to each block holding one orbital
	  blocks.resize(n_blocks, std::vector<size_t>());
	  for(std::size_t i=0; i < blocks.size(); ++i) {
	       std::vector<size_t> temp{i};
	       blocks[i] = temp;
	  }
     }
     measured_c1 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     measured_c2 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     measured_c3 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     target_c1 = Eigen::MatrixXcd::Identity(per_site_orbital_size, per_site_orbital_size);
     target_c2 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     target_c3 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
}

void GfBase::get_interaction_matrix(int ref_site_index, const alps::params &parms) {
     // TODO: get this uniform between alps3 and alps2
     // Convention issue: no need to worry about it with interaction matrix,
     // since it is symmetric for density-density interactions.
     std::vector<double> u_elements;
     u_elements.resize(per_site_orbital_size * per_site_orbital_size);
     if (parms.exists("U_MATRIX")) {
	  std::string ufilename = boost::lexical_cast<std::string>(parms["U_MATRIX"]);
	  if (parms.exists("UMATRIX_IN_HDF5") &&
	      boost::lexical_cast<bool>(parms["UMATRIX_IN_HDF5"])) {
	       //attempt to read from h5 archive
	       alps::hdf5::archive u_archive(ufilename, alps::hdf5::archive::READ);
	       u_archive >> alps::make_pvp("/Umatrix", u_elements);
	  } else {
	       std::cerr << "U matrix is not defined properly in hdf5 input file in function "
		    + std::string(__FUNCTION__);
	       throw std::runtime_error("pb reading U_MATRIX");	       
	  }
     } else {
	  std::cerr << "U_MATRIX param is absent from hdf5 input file in function "
	       + std::string(__FUNCTION__);
	  throw std::runtime_error("pb reading U_MATRIX");
     }
     interaction_matrix =
	  Eigen::MatrixXcd::Constant(tot_orbital_size, tot_orbital_size, 0.0);
     int cur_index = 0;
     for (int orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
	  for (int orb2 = 0; orb2 < per_site_orbital_size; orb2++, cur_index++) {
	       interaction_matrix.block(ref_site_index * per_site_orbital_size,
					ref_site_index * per_site_orbital_size,
					per_site_orbital_size,
					per_site_orbital_size)(orb1, orb2) =
		    u_elements[cur_index];
	  }
     }
}

void GfBase::get_density_density_correl(int ref_site_index,
					const alps::params &parms,
					alps::hdf5::archive &h5_archive) {
     // density density correlation
     // is available in observables.dat or in the hdf5 file.
     if (parms["from_alps3"].as<bool>()) {
	  std::string density_density_result_name = "DENSITY_DENSITY_CORRELATION_FUNCTIONS";
	  // density density data
	  typedef boost::multi_array<double, 3> array_type;
	  int n_tau_for_density = boost::lexical_cast<int>(parms["measurement.nn_corr.n_tau"]);
	  int n_orbital_pairs_for_density = per_site_orbital_size * (per_site_orbital_size + 1) / 2;
	  array_type density_data(
	       boost::extents[n_orbital_pairs_for_density][n_tau_for_density][2]);
	  h5_archive[density_density_result_name] >> density_data;
	  // Build density density data
	  std::string fname = boost::lexical_cast<std::string>(parms["measurement.nn_corr.def"]);
	  std::ifstream infile(fname.c_str());
	  if(!infile.good()) {
	       cout << "Could not find file " << fname <<
		    " for definition of density density data" << endl;
	       throw runtime_error("Bad input for density density calcs !");
	  } else {
	       double cur_dd_correl;
	       int data_index, orb1, orb2, nb_lines;
	       std::string line;
	       std::getline(infile, line);
	       std::istringstream iss(line);
	       iss >> nb_lines;
	       for (int i = 0; i < nb_lines; i++) {
		    std::string line2;
		    std::getline(infile, line2);
		    std::istringstream iss2(line2);
		    iss2 >> data_index >> orb1 >> orb2;
		    cur_dd_correl = density_data[data_index][0][0];
		    density_density_correl.block(ref_site_index * per_site_orbital_size,
						 ref_site_index * per_site_orbital_size,
						 per_site_orbital_size,
						 per_site_orbital_size)(orb1, orb2)
			 = cur_dd_correl;
		    density_density_correl.block(ref_site_index * per_site_orbital_size,
						 ref_site_index * per_site_orbital_size,
						 per_site_orbital_size,
						 per_site_orbital_size)(orb2, orb1)
			 = cur_dd_correl;
	       }
	  }
     } else {
	  // TODO use uniform N_TAU
	  int n_tau = boost::lexical_cast<int>(parms["N_TAU"]) + 1;
	  std::string gtau_path("/od_hyb_G_tau/");
	  double cur_dd_correl;
	  for(int block_index = 0; block_index < n_blocks; ++block_index) {
	       int cur_index = 0;
	       for(int line_idx = 0; line_idx < blocks[block_index].size(); ++line_idx) {
		    for(int col_idx = 0; col_idx < blocks[block_index].size();
			++col_idx) {
			 std::stringstream density_path;
			 // ATTENTION here: convention of QMC is F_ij = -T<c_i c^dag_j>,
			 // but DMFT is looking for  c^dag_i c_j
			 // No - misleading comment. Just check the way the quantities
			 // are dumped in hdf5 by QMC - this is fine.
			 cur_index = line_idx + col_idx * blocks[block_index].size();
			 if (line_idx == col_idx) {
			      density_path << "/simulation/results/density_" <<
				   boost::lexical_cast<std::string>(line_idx) + "/mean/value";
			 } else {
			      if (line_idx > col_idx) {
				   density_path << "/simulation/results/nn_" <<
					boost::lexical_cast<std::string>(line_idx) + "_" +
					boost::lexical_cast<std::string>(col_idx)
					+ "/mean/value";
			      } else {
				   density_path << "/simulation/results/nn_" <<
					boost::lexical_cast<std::string>(col_idx) + "_" +
					boost::lexical_cast<std::string>(line_idx)
					+ "/mean/value";
			      }
			 }
			 // VERY CAREFUL HERE
			 // What comes out of QMC is G_{kl} =  -<T c_k(tau) c^dagger_l(tau')>
			 // we need c^\dagger_k c_l here, so transpose -
			 // see above in the construction!!
			 h5_archive >> alps::make_pvp(density_path.str(), cur_dd_correl);
			 density_density_correl.block(ref_site_index * per_site_orbital_size,
						      ref_site_index * per_site_orbital_size,
						      per_site_orbital_size,
						      per_site_orbital_size)
			      (blocks[block_index][line_idx],
			       blocks[block_index][col_idx]) = cur_dd_correl;
		    }
	       }
	  }
     }
     // Print out some log for convergence check against
     // similar output from dmft
     cout << "<n_i n_j> :" << endl << density_density_correl.block(
	  ref_site_index * per_site_orbital_size,
	  ref_site_index * per_site_orbital_size,
	  per_site_orbital_size,
	  per_site_orbital_size).real() << endl;
}

void GfBase::get_a_dagger_b(int ref_site_index,
			    const alps::params &parms,
			    alps::hdf5::archive &h5_archive) {
     // a_dagger_b is picked from the
     // values of G(tau = beta)
     // density density correlation
     // is available in observables.dat or in the hdf5 file.
     // HERE - maybe not
     // Convention issue: no need to worry about it with interaction matrix,
     // since it is symmetric for density-density interactions.
     // TODO : get rid of u_pd in Alps3
     if (parms["from_alps3"].as<bool>()) {
	  int n_tau = boost::lexical_cast<int>(parms["measurement.G1.n_tau"]) + 1;
	  typedef boost::multi_array<complex<double> , 3> cplx_array_type;
	  cplx_array_type raw_full_gf(
	       boost::extents[n_tau][per_site_orbital_size][per_site_orbital_size]);	  
	  h5_archive["/gtau/data"] >> raw_full_gf;
	  // Build a_dagger_b data
	  for(int block_index = 0; block_index < n_blocks; ++block_index) {
	       for(int line_idx = 0; line_idx < blocks[block_index].size(); ++line_idx) {
		    for(int col_idx = 0; col_idx < blocks[block_index].size(); ++col_idx) {
			 // ATTENTION here: convention of QMC Alps2 is F_ij = -T<c_i c^dag_j>,
			 // but DMFT is looking for  c^dag_i c_j
			 // VERY CAREFUL HERE -- With Alps3 we are fine though!!
			 // What comes out of QMC is G_{kl} =  -<T c_k(tau) c^dagger_l(tau')>
			 // we need c^\dagger_k c_l here, so transpose -
			 a_dagger_b.block(ref_site_index * per_site_orbital_size,
					  ref_site_index * per_site_orbital_size,
					  per_site_orbital_size,
					  per_site_orbital_size)
			      (blocks[block_index][line_idx],
			       blocks[block_index][col_idx]) =
			      // ATTENTION: DIFFERENT SIGN CONVENTION FROM FORTRAN
			      // HERE
			      // AND NEED TO SYMMETRIZE FOR DATA FROM ALPS3
			      -0.5 * (raw_full_gf[n_tau - 1][line_idx][col_idx]
				      + std::conj(raw_full_gf[n_tau - 1][col_idx][line_idx]));
		    }
	       }
	  }
     } else { // now Alps2 version
	  int n_tau = boost::lexical_cast<int>(parms["N_TAU"]) + 1;
	  std::string gtau_path("/od_hyb_G_tau/");
	  std::vector<std::complex<double> > temp_data;
	  temp_data.resize(n_tau);
	  for(int block_index = 0; block_index < n_blocks; ++block_index) {
	       int cur_index = 0;
	       for(int line_idx = 0; line_idx < blocks[block_index].size(); ++line_idx) {
		    for(int col_idx = 0; col_idx < blocks[block_index].size();
			++col_idx) {
			 std::stringstream orbital_path;
			 // ATTENTION here: convention of QMC is F_ij = -T<c_i c^dag_j>,
			 // but DMFT is looking for  c^dag_i c_j
			 //cur_index = line_idx + col_idx * blocks[block_index].size();
			 cur_index = col_idx + line_idx * blocks[block_index].size();
			 orbital_path << gtau_path << boost::lexical_cast<std::string>(block_index) +
			      "/" + boost::lexical_cast<std::string>(cur_index) + "/mean/value";
			 // VERY CAREFUL HERE
			 // What comes out of QMC is G_{kl} =  -<T c_k(tau) c^dagger_l(tau')>
			 // we need c^\dagger_k c_l here, so transpose -
			 // see above in the construction!!
			 h5_archive >> alps::make_pvp(orbital_path.str(), temp_data);
			 a_dagger_b.block(ref_site_index * per_site_orbital_size,
					  ref_site_index * per_site_orbital_size,
					  per_site_orbital_size,
					  per_site_orbital_size)
			      (blocks[block_index][line_idx],
			       blocks[block_index][col_idx]) =
			      // ATTENTION: DIFFERENT SIGN CONVENTION FROM FORTRAN
			      // HERE
			      -temp_data.back();
		    }
	       }
	  } 
     }
     Eigen::MatrixXcd temp_matrix = a_dagger_b;
     temp_matrix.adjointInPlace();
     a_dagger_b = 0.5 * (a_dagger_b + temp_matrix);
}

void GfBase::feed_tail_params(int ref_site_index,
			      const alps::params &parms,
			      alps::hdf5::archive &h5_archive) {
     get_interaction_matrix(ref_site_index, parms);
     get_density_density_correl(ref_site_index, parms, h5_archive);
     get_a_dagger_b(ref_site_index, parms, h5_archive);
     get_target_c2(ref_site_index);
     get_target_c3(ref_site_index);
}

Eigen::MatrixXcd GfBase::get_measured_c2() {
     return measured_c2;
}

Eigen::MatrixXcd GfBase::get_measured_c3() {
     return measured_c3;
}

void GfBase::get_target_c2(int ref_site_index) {
     //std::cout << "Compute target c2 for GF" << std::endl << std::endl;
     target_c2 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     // Get the local hoppings
     Eigen::MatrixXcd bath_m1 = lattice_bs_->get_local_hoppings();
     // and fix the diagonal elements (chempot includes atomic level energy:
     // it is -epsilon + mu...)
     for(int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx)
	  bath_m1(line_idx, line_idx) = -(*chempot_)[line_idx];
     Eigen::MatrixXcd U_matrix = 0.5 * (interaction_matrix.block(ref_site_index * per_site_orbital_size,
							  ref_site_index * per_site_orbital_size,
							  per_site_orbital_size,
							  per_site_orbital_size) +
					interaction_matrix.block(ref_site_index * per_site_orbital_size,
							  ref_site_index * per_site_orbital_size,
							  per_site_orbital_size,
								 per_site_orbital_size).transpose());
     Eigen::MatrixXcd dd_matrix = density_density_correl.block(ref_site_index * per_site_orbital_size,
							       ref_site_index * per_site_orbital_size,
							       per_site_orbital_size,
							       per_site_orbital_size);
     Eigen::VectorXcd density_vector = dd_matrix.diagonal();
     Eigen::MatrixXcd ab_matrix = a_dagger_b.block(ref_site_index * per_site_orbital_size,
						   ref_site_index * per_site_orbital_size,
						   per_site_orbital_size,
						   per_site_orbital_size);
     target_c2 -= U_matrix.transpose().cwiseProduct(ab_matrix);
     target_c2 += bath_m1;
     target_c2.diagonal() += U_matrix * density_vector;
}

void GfBase::get_target_c3(int ref_site_index) {
     // 0.5 factor stems from the fact that the formal expression
     // of the Hamiltonian is without order in the thesis, while it has
     // order (a before b) in the papers ==> equivalent to specifying U/2
     // in the papers
     // For details of derivation of formulae, see Gull thesis,
     // Appendix B.4, or hopefully even better, my own thesis, appendices.
     // Sell also Ferber's thesis for order 3 coeff.
     //std::cout << "Compute target c3 for GF" << std::endl << std::endl;
     target_c3 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     // Get the local hoppings
     Eigen::MatrixXcd bath_m1 = lattice_bs_->get_local_hoppings();
     // and fix the diagonal elements (chempot includes atomic level energy:
     // it is -epsilon + mu...)
     for(int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx)
	  bath_m1(line_idx, line_idx) = -(*chempot_)[line_idx];
     Eigen::MatrixXcd V_matrix = lattice_bs_->get_V_matrix();
     Eigen::MatrixXcd bath_m2 = bath_m1 * bath_m1.adjoint() +
	  8.0 * (V_matrix * V_matrix.adjoint() + V_matrix.adjoint() * V_matrix);
     target_c3 += bath_m2;
     Eigen::MatrixXcd U_matrix = interaction_matrix.block(ref_site_index * per_site_orbital_size,
							  ref_site_index * per_site_orbital_size,
							  per_site_orbital_size,
							  per_site_orbital_size);
     Eigen::MatrixXcd dd_matrix = density_density_correl.block(ref_site_index * per_site_orbital_size,
							       ref_site_index * per_site_orbital_size,
							       per_site_orbital_size,
							       per_site_orbital_size);
     Eigen::VectorXcd density_vector = dd_matrix.diagonal();
     Eigen::MatrixXcd ab_matrix = a_dagger_b.block(ref_site_index * per_site_orbital_size,
						   ref_site_index * per_site_orbital_size,
						   per_site_orbital_size,
						   per_site_orbital_size);
     Eigen::MatrixXcd temp = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     for (int col_index = 0; col_index < per_site_orbital_size; col_index++) {
	  temp.col(col_index) += U_matrix * density_vector;
     }
     for (int row_index = 0; row_index < per_site_orbital_size; row_index++) {
	  temp.row(row_index) += density_vector.transpose() * U_matrix;
     }     
     target_c3 += bath_m1.cwiseProduct(temp);
     target_c3 -= (bath_m1 * (U_matrix.cwiseProduct(ab_matrix)).transpose());
     target_c3 -= (bath_m1.transpose() * ((U_matrix.transpose()).cwiseProduct(ab_matrix))).transpose();
     Eigen::MatrixXcd factor_3 =
	  -U_matrix.cwiseProduct((ab_matrix.transpose()).cwiseProduct(temp));
     // Eigen::MatrixXcd factor_4 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     // for (int row_index = 0; row_index < per_site_orbital_size; row_index++) {
     // 	  for (int col_index = 0; col_index < per_site_orbital_size; col_index++) {
     // 	       for (int l_index = 0; l_index < per_site_orbital_size; l_index++) {
     // 		    factor_4(row_index, col_index) += (U_matrix(row_index, l_index) +
     // 						       U_matrix(col_index, l_index)) *
     // 			 ab_matrix(col_index, l_index) *
     // 			 ab_matrix(l_index, row_index);
     // 	       }
     // 	  }
     // }
     // factor_4 = U_matrix.cwiseProduct(factor_4);
     //std::cout << "factor 4 " << std::endl << factor_4 << std::endl << std::endl;
     //std::cout << "factor 3 " << std::endl << factor_3 << std::endl << std::endl;
     // TODO : we need to neglect factor 4 in order to match Fortran
     // TODO : moreover, the decomposition of the 4 point correlator into
     // factor3 + factor4 relies on Wick's theorem, which is not applicable
     // as far as I understand.
     //std::cout << "factor_4" << std::endl << U_matrix.cwiseProduct(factor_4) << std::endl << std::endl;
     //target_c3 += (factor_3 + U_matrix.cwiseProduct(factor_4));
     // temp = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     // for (int row_index = 0; row_index < per_site_orbital_size; row_index++) {
     // 	  for (int col_index = 0; col_index < per_site_orbital_size; col_index++) {
     // 	       for (int l_index = 0; l_index < per_site_orbital_size; l_index++) {
     // 		    temp(row_index, col_index) += (U_matrix(row_index, l_index) *
     // 						       U_matrix(col_index, l_index)) *
     // 			    ab_matrix(col_index, l_index) *
     // 			    ab_matrix(l_index, row_index);
     // 	       }
     // 	  }
     // }
     // std::cout << "additonal: " << std::endl << temp << std::endl << std::endl;
     target_c3 += (factor_3);
     temp = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     for(int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx) {
	  for(int k = 0; k < per_site_orbital_size; ++k) {
	       for(int l = 0; l < per_site_orbital_size; ++l) {
		    temp(line_idx, line_idx) += U_matrix(line_idx, k) * U_matrix(line_idx, l) *
			 dd_matrix(k, l);
	       }
	  }
     }
     target_c3.diagonal() += temp.diagonal();
     //std::cout << "TARGET C3" << std::endl << target_c3 << std::endl << std::endl;
}

void GfBase::get_new_target_c3(int ref_site_index) {
     get_new_target_c3(ref_site_index);
     target_c3 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     Eigen::MatrixXcd temp = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     for(int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx) {
	  for(int col_idx = 0; col_idx < per_site_orbital_size; ++col_idx) {
	       // HERE Note that there is some possible discrepancy to remove between
	       // dmft and qmc: on the dmft side, the crystal field is
	       // input via the diagonal elements of the hopping matrix,
	       // while on the qmc side it is blent with mu in MUvector.
	       // as a consequence, the energy of each orbital in the present code
	       // is indeed the \epsilon_k needed in the formula from my thesis.
	       for(int ter_idx = 0; ter_idx < per_site_orbital_size; ++ter_idx) {
		    temp(line_idx, line_idx) +=
			 interaction_matrix.block(ref_site_index * per_site_orbital_size,
						  ref_site_index * per_site_orbital_size,
						  per_site_orbital_size,
						  per_site_orbital_size)(line_idx, col_idx) *
			 interaction_matrix.block(ref_site_index * per_site_orbital_size,
						  ref_site_index * per_site_orbital_size,
						  per_site_orbital_size,
						  per_site_orbital_size)(line_idx, ter_idx) *
			 (density_density_correl.block(ref_site_index * per_site_orbital_size,
						       ref_site_index * per_site_orbital_size,
						       per_site_orbital_size,
						       per_site_orbital_size)(ter_idx, col_idx) -
			  density_density_correl.block(ref_site_index * per_site_orbital_size,
						       ref_site_index * per_site_orbital_size,
						       per_site_orbital_size,
						       per_site_orbital_size)(ter_idx, ter_idx) *
			  density_density_correl.block(ref_site_index * per_site_orbital_size,
						       ref_site_index * per_site_orbital_size,
						       per_site_orbital_size,
						       per_site_orbital_size)(col_idx, col_idx));
	       }
	  }
     }
     Eigen::MatrixXcd ab_matrix = a_dagger_b.block(ref_site_index * per_site_orbital_size,
						   ref_site_index * per_site_orbital_size,
						   per_site_orbital_size,
						   per_site_orbital_size);
     Eigen::MatrixXcd U_matrix = interaction_matrix.block(ref_site_index * per_site_orbital_size,
							  ref_site_index * per_site_orbital_size,
							  per_site_orbital_size,
							  per_site_orbital_size);
     Eigen::MatrixXcd temp2 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     for(int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx) {
	  for(int col_idx = 0; col_idx < per_site_orbital_size; ++col_idx) {
	       for(int ter_idx = 0; ter_idx < per_site_orbital_size; ++ter_idx) {
		    temp2(line_idx, col_idx) += U_matrix(line_idx, ter_idx) *
			 U_matrix(ter_idx, col_idx) * ab_matrix(ter_idx, line_idx) *
			 ab_matrix(col_idx, ter_idx);
	       }
	  }
     }     
     target_c3 = target_c2 * target_c2 + temp;
     std::cout << "FAKE TARGET C3" << std::endl << target_c3 << std::endl << std::endl;
}
