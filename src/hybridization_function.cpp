#include "hybridization_function.hpp"
#include <boost/timer/timer.hpp>
#include <alps/hdf5/multi_array.hpp>

using namespace std;
typedef boost::multi_array<complex<double> , 3> cplx_array_type;

HybFunction::HybFunction(const alps::params &parms,
			 boost::shared_ptr<Bandstructure> const &lattice_bs,
			 boost::shared_ptr<Selfenergy> const &sigma,
			 complex<double> chemical_potential, int world_rank,
			 bool compute_bubble, bool verbose):
     lattice_bs_(lattice_bs), sigma_(sigma), world_rank_(world_rank),
     enforce_real(sigma->get_enforce_real()),
     chemical_potential(chemical_potential),
     compute_bubble(compute_bubble) {
     boost::timer::auto_cpu_timer hyb_calc; 
     n_tau = static_cast<size_t>(parms["N_TAU"]);
     n_sites = sigma_->get_n_sites();
     per_site_orbital_size = sigma_->get_per_site_orbital_size();
     tot_orbital_size = n_sites * per_site_orbital_size;
     N_boson = parms["measurement.G2.n_bosonic_freq"];
     int N_Qmesh = static_cast<size_t>(parms["N_QBSEQ"]);
     size_t N_max = sigma_->get_n_matsubara_freqs();
     bubble_dim = parms.exists("N_NU_BSEQ") ? static_cast<int>(parms["N_NU_BSEQ"]) : N_max - N_boson;
     if (compute_bubble) {
	  std::cout << "Computing bubbles for " << bubble_dim << " fermionic frequencies, "
		    << std::endl << " q mesh based on " << N_Qmesh << " points, i.e. "
		    << lattice_bs->get_nb_points_for_bseq() << " q points, and " 
		    << N_boson << " bosonic frequencies" << std::endl;
	  world_local_bubble.clear();
	  world_local_bubble.resize(N_boson);
	  for (size_t boson_freq = 0; boson_freq < N_boson; boson_freq++) {
	       for (size_t freq_index = 0; freq_index < bubble_dim; freq_index++) {
		    world_local_bubble[boson_freq].push_back(
			 Eigen::MatrixXcd::Zero(n_sites * per_site_orbital_size * per_site_orbital_size,
						n_sites * per_site_orbital_size * per_site_orbital_size));
	       }
	  }
	  world_lattice_bubble.clear();
	  world_lattice_bubble.resize(N_boson);
	  for (size_t boson_freq = 0; boson_freq < N_boson; boson_freq++) {
	       world_lattice_bubble[boson_freq].resize(lattice_bs_->get_nb_points_for_bseq());
	       for (size_t q_index = 0; q_index < lattice_bs_->get_nb_points_for_bseq(); q_index++) {
		    for (size_t freq_index = 0; freq_index < bubble_dim; freq_index++) {
			 world_lattice_bubble[boson_freq][q_index].push_back(
			      Eigen::MatrixXcd::Zero(n_sites * per_site_orbital_size * per_site_orbital_size,
						     n_sites * per_site_orbital_size * per_site_orbital_size));
		    }
	       }
	  }
     }
     world_bath_moment_1 = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     world_bath_moment_2 = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     // Compute the site-level-averaged contribution of
     // <epsilon>^2 --> project out the sitewise off-diagonal blocks.
     Eigen::MatrixXcd site_wise_epsilon_bar =
	  Eigen::MatrixXcd::Zero(tot_orbital_size,
				 tot_orbital_size);
     for (size_t site_index = 0; site_index < n_sites; site_index++) {
	  site_wise_epsilon_bar.block(
	       site_index * per_site_orbital_size,
	       site_index * per_site_orbital_size,
	       per_site_orbital_size, per_site_orbital_size) =
	       lattice_bs_->get_epsilon_bar().block(
		    site_index * per_site_orbital_size,
		    site_index * per_site_orbital_size,
		    per_site_orbital_size, per_site_orbital_size);
     }
     Eigen::MatrixXcd first_order =  lattice_bs_->get_epsilon_squared_bar() -
	  site_wise_epsilon_bar * site_wise_epsilon_bar;
     hf_coeff.clear();
     // Store the first order coefficient
     hf_coeff.push_back(first_order);
     // Then compute the full hybridization function...
     // do *NOT* remove this computation from the constructor -
     //
     compute_hybridization_function(chemical_potential);
     // And extract the 2nd and 3rd order coefficients from the numerics.
     compute_superior_orders(verbose);
     elementary_compute_delta_tau();
     dump_delta();
     dump_delta_hdf5();
     dump_delta_for_matrix();
     cout << "hybridization time: ";
}

/* The coefficients of order 2 and 3 are computed numerically
 * using the hf values of the numerical form of Delta. Here again,
 * this is done on a site-projected form of the quantities.
 */
void HybFunction::compute_superior_orders(bool verbose) {
     Eigen::MatrixXcd second_order = Eigen::MatrixXcd::Zero(tot_orbital_size,
							    tot_orbital_size);
     Eigen::MatrixXcd third_order = Eigen::MatrixXcd::Zero(tot_orbital_size,
							   tot_orbital_size);
     Eigen::MatrixXcd temp = Eigen::MatrixXcd::Zero(tot_orbital_size,
						    tot_orbital_size);     
     size_t N_max = sigma_->get_n_matsubara_freqs();
     for (size_t freq_index = N_max - tail_fit_length;
	  freq_index < N_max; freq_index++) {
	  second_order += 0.5 *
	       pow(sigma_->get_matsubara_frequency(freq_index), 2) *
	       (hybridization_function[freq_index] +
		hybridization_function[freq_index].transpose().conjugate());
	  temp = 0.5 * sigma_->get_matsubara_frequency(freq_index) *
	       (hybridization_function[freq_index] -
		hybridization_function[freq_index].transpose().conjugate());
	  temp -= hf_coeff[0];
	  third_order += temp * pow(sigma_->get_matsubara_frequency(freq_index), 2);
     }
     second_order /= tail_fit_length;
     third_order /= tail_fit_length;
     hf_coeff.push_back(second_order);
     hf_coeff.push_back(third_order);
     if (verbose) {
	  display_asymptotics();
     }
}

void HybFunction::display_asymptotics() {
     if (world_rank_ == 0) {
	  auto old_precision = cout.precision(5);
	  for (size_t site_index = 0; site_index < n_sites; site_index++) {
	       cout << "HYBRIDIZATION FUNCTION ASYMPTOTICS: " << site_index << endl << endl;;
	       cout << "order 1: " << endl;
	       cout << hf_coeff[0].block(site_index * per_site_orbital_size,
					 site_index * per_site_orbital_size,
					 per_site_orbital_size, per_site_orbital_size)
		    << endl << endl;
	       cout << "order 2: " << endl;
	       cout << hf_coeff[1].block(site_index * per_site_orbital_size,
					 site_index * per_site_orbital_size,
					 per_site_orbital_size, per_site_orbital_size)
		    << endl << endl;
	       cout << "order 3: " << endl;
	       cout << hf_coeff[2].block(site_index * per_site_orbital_size,
					 site_index * per_site_orbital_size,
					 per_site_orbital_size, per_site_orbital_size)
		    << endl << endl;
	  }
	  cout.precision(old_precision);
     }
}

void HybFunction::compute_hybridization_function(complex<double> mu) {
     if (world_rank_ == 0) {
	  cout << "***********************************************" << endl;
	  cout << "** HYBRIDIZATION FUNCTION CALCULATION      ***" << endl;
	  cout << "***********************************************" << endl << endl;
     }
     // Cf Ferber thesis, A.15 and A.26
     // This is the actual calculation of:
     // F(iomega_n)
     // G_0(iomega_n)
     // as stated in A.26, taking into account the necessary
     // projections on sites for the multi-site case.
     size_t k_min(0);
     size_t k_max = lattice_bs_->get_lattice_size();
     double beta = sigma_->get_beta();
     size_t N_max = sigma_->get_n_matsubara_freqs();
     // initialize quantities to be computed
     hybridization_function.clear();
     G0_function.clear();
     bare_greens_function.clear();
     no_shift_bare_greens_function.clear();
     pure_no_shift_bare_greens_function.clear();
     Eigen::MatrixXcd local_greens_function =
	  Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     Eigen::MatrixXcd pure_local_bare_gf =
	  Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     Eigen::MatrixXcd world_local_greens_function =
	  Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
          Eigen::MatrixXcd  world_pure_local_bare_gf =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     Eigen::MatrixXcd inverse_gf(tot_orbital_size, tot_orbital_size);
     Eigen::MatrixXcd bath_moment_1(tot_orbital_size, tot_orbital_size);
     Eigen::MatrixXcd bath_moment_2(tot_orbital_size, tot_orbital_size);
     Eigen::MatrixXcd pure_inverse_bare_gf(tot_orbital_size, tot_orbital_size);
     if (compute_bubble) {
	  world_local_gf.clear();
	  for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	       world_local_gf.push_back(Eigen::MatrixXcd::Zero(
						tot_orbital_size, tot_orbital_size));
	  }
     }
     world_bath_moment_1 = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     world_bath_moment_2 = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     // mu_tilde is the opposite of xm1 in the original Fortran code.
     mu_tilde = -lattice_bs_->get_epsilon_bar();
     mu_tilde.diagonal() += Eigen::VectorXcd::Constant(tot_orbital_size, mu);
     // TODO
     // It is probably very sub optimal to have a MPI_Allreduce inside
     // a frequency loop....
     Eigen::MatrixXcd eta_inf = sigma_->get_sigma_0();
     for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	  pure_local_bare_gf = Eigen::MatrixXcd::Zero(tot_orbital_size,
						      tot_orbital_size);	  
	  local_greens_function = Eigen::MatrixXcd::Zero(tot_orbital_size,
							 tot_orbital_size);
	  Eigen::VectorXcd mu_plus_iomega = Eigen::VectorXcd::Constant
	       (tot_orbital_size, mu + sigma_->get_matsubara_frequency(freq_index));
	  Eigen::MatrixXcd self_E = sigma_->values_[freq_index];
	  for (size_t k_index = k_min; k_index < k_max; k_index++) {
	       double l_weight = lattice_bs_->get_weight(k_index);
	       if (abs(l_weight) < 1e-6) {
		    if ((freq_index == 0) && (world_rank_ == 0)) {
			 cout << "skipping k point" << endl;
		    }
		    continue;
	       } else {
		    pure_inverse_bare_gf = -lattice_bs_->dispersion_[k_index];
		    inverse_gf = -lattice_bs_->dispersion_[k_index] - self_E;
		    pure_inverse_bare_gf.diagonal() += mu_plus_iomega;
		    inverse_gf.diagonal() += mu_plus_iomega;
                    if (freq_index == 0) {
                         Eigen::MatrixXcd temp_comp = lattice_bs_->dispersion_[k_index] + eta_inf;
                         temp_comp.diagonal() -= Eigen::VectorXcd::Constant(tot_orbital_size, mu);
                         bath_moment_1 += temp_comp * l_weight;
                         bath_moment_2 += temp_comp * temp_comp * l_weight;
                    }
		    pure_local_bare_gf += pure_inverse_bare_gf.inverse() * l_weight;
		    local_greens_function += inverse_gf.inverse() * l_weight;
	       }
	  }
	  MPI_Allreduce(local_greens_function.data(),
			world_local_greens_function.data(),
			local_greens_function.size(),
			MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
	  MPI_Allreduce(pure_local_bare_gf.data(),
			world_pure_local_bare_gf.data(),
			pure_local_bare_gf.size(),
			MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
	  // local_greens_function is inversed, but only after it is projected
	  // out on each site. xm1 is projected on site as well
	  inverse_gf = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       if (compute_bubble) {
		    // We are interested in the linear response
		    // in the paramagentic phase, thus we do
		    // not need to separate spins anymore.
		    world_local_gf[freq_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) =
			 world_local_greens_function.block(
			      site_index * per_site_orbital_size,
			      site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size);
	       }
	       Eigen::MatrixXcd world_inverse =
		    world_local_greens_function.block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size).inverse();
	       inverse_gf.block(site_index * per_site_orbital_size,
				site_index * per_site_orbital_size,
				per_site_orbital_size,
				per_site_orbital_size) =
		    -world_inverse;
	  }
	  // No need to site-project - sigma is site diagonal
	  inverse_gf -= sigma_->values_[freq_index];
	  no_shift_bare_greens_function.push_back(-inverse_gf.inverse());
	  pure_no_shift_bare_greens_function.push_back(world_pure_local_bare_gf);
	  inverse_gf.diagonal() += mu_tilde.diagonal();
	  // TODO This follows the Fortran code, which uses a
	  // "shift" quantity, saved in a specified file shift.tmp,
	  // but is not compliant with either Hafferman
	  // PHYSICAL REVIEW B 85, 205106 (2012), Eq. (8), or
	  // equivalently, A.26 in Ferber, or Gull B.38.
	  // Since the point of this is to calculate the self-energy
	  // after the QMC step, it works because the shift is applied
	  // backwards in mix.f, where sigma is computed, but why use such
	  // a convoluted implementation?
	  // Morevoer, in that case, why save G0 and not G0^-1? This would
	  // save two inversions.
	  bare_greens_function.push_back(-inverse_gf.inverse());
	  inverse_gf.diagonal() += Eigen::VectorXcd::Constant
	       (tot_orbital_size,
		sigma_->get_matsubara_frequency(freq_index));
	  hybridization_function.push_back(inverse_gf);
	  inverse_gf.diagonal() -= Eigen::VectorXcd::Constant
	       (tot_orbital_size,
		sigma_->get_matsubara_frequency(freq_index));
	  //G0_function.push_back(-inverse_gf.inverse());
	  inverse_gf.diagonal() -= mu_tilde.diagonal();
	  G0_function.push_back(-inverse_gf.inverse());
     }
     MPI_Allreduce(bath_moment_1.data(),
                   world_bath_moment_1.data(),
                   bath_moment_1.size(),
                   MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(bath_moment_2.data(),
                   world_bath_moment_2.data(),
                   bath_moment_2.size(),
                   MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
     Eigen::MatrixXcd temp_qty = world_bath_moment_1;
     world_bath_moment_1 = world_bath_moment_1 - sigma_->get_sigma_0();
     world_bath_moment_2 = -temp_qty * temp_qty +
          world_bath_moment_1 * world_bath_moment_1 + world_bath_moment_2;
     world_bath_moment_2 = world_bath_moment_2 - world_bath_moment_1 * world_bath_moment_1;
     world_bath_moment_1.diagonal() = Eigen::VectorXcd::Zero(n_sites * per_site_orbital_size);
     world_bath_moment_2 = world_bath_moment_2 + world_bath_moment_1 * world_bath_moment_1;
}

std::vector<Eigen::MatrixXcd> HybFunction::get_greens_function(
     Eigen::Ref<Eigen::VectorXd> k_point, int boson_index) {
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

void HybFunction::compute_local_bubble() {
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
					     new_i = part_index_1 *
						  orbital_size + hole_index_2;
					     new_j = part_index_2 *
						  orbital_size + hole_index_1;
					     world_local_bubble[boson_index]
						  [freq_index].block(
						       site_index *
						       per_site_orbital_size *
						       per_site_orbital_size,
						       site_index *
						       per_site_orbital_size *
						       per_site_orbital_size,
						       per_site_orbital_size *
						       per_site_orbital_size,
						       per_site_orbital_size *
						       per_site_orbital_size)(new_i,
									      new_j) =
						  world_local_gf[
						       freq_index].block(
							    site_index *
							    per_site_orbital_size,
							    site_index *
							    per_site_orbital_size,
							    per_site_orbital_size,
							    per_site_orbital_size)(
								 part_index_1,
								 part_index_2) *
						  world_local_gf[freq_index + boson_index].block(
						       site_index *
						       per_site_orbital_size,
						       site_index *
						       per_site_orbital_size,
						       per_site_orbital_size,
						       per_site_orbital_size)(
							    hole_index_1, hole_index_2);
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

void HybFunction::compute_lattice_bubble() {
     boost::timer::auto_cpu_timer lattice_bubble_calc;
     size_t k_min(0);
     size_t k_max(lattice_bs_->get_lattice_size());
     int orbital_size(lattice_bs_->get_orbital_size());
     int nb_q_points(lattice_bs_->get_nb_points_for_bseq());
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
	       }
	  }
	  std::cout << "Time for boson freq " << boson_index
		    << ": " << std::endl;
     } // boson
}

void HybFunction::dump_delta() {
     if (world_rank_ == 0) {
	  size_t N_max = sigma_->get_n_matsubara_freqs();     
	  ofstream out(matsubara_frequency_dump_name);
	  out.precision(output_precision);
	  out << fixed << setprecision(output_precision);
	  
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
		    for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			 for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
			      out << imag(sigma_->get_matsubara_frequency(freq_index)) << "     "
				  << real(hybridization_function[freq_index].block(
					       site_index * per_site_orbital_size,
					       site_index * per_site_orbital_size,
					       per_site_orbital_size,
					       per_site_orbital_size)(orb1, orb2)) << "     "
				  << imag(hybridization_function[freq_index].block(
					       site_index * per_site_orbital_size,
					       site_index * per_site_orbital_size,
					       per_site_orbital_size,
					       per_site_orbital_size)(orb1, orb2)) << endl;
			 }
			 out << endl;
		    }
	       }
	  }
	  out.close();
	  double beta = sigma_->get_beta();
	  out.open(imaginary_time_dump_name, ofstream::out);
	  out << fixed << setprecision(output_precision);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
		    for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			 for (size_t tau_index = 0; tau_index < delta_tau.size(); tau_index++) {
			      // CAREFUL! delta_tau has been extended to include tau = beta
			      // it has one more element than should be for a proper definition of
			      // tau_value:
			      out << beta * static_cast<double>(tau_index) / (delta_tau.size() - 1) << "     "
				  << -real(delta_tau[tau_index].block(
						site_index * per_site_orbital_size,
						site_index * per_site_orbital_size,
						per_site_orbital_size,
						per_site_orbital_size)(orb1, orb2))
				  << -imag(delta_tau[tau_index].block(
						site_index * per_site_orbital_size,
						site_index * per_site_orbital_size,
						per_site_orbital_size,
						per_site_orbital_size)(orb1, orb2)) << endl;
			 }
			 out << endl;
		    }
	       }
	  }
	  out.close();
	  // dump bare GF in Matsubara frequencies
	  out.open(matsubara_bare_gf_dump_name, ofstream::out);
	  out.precision(output_precision);
	  out << fixed << setprecision(output_precision);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
		    for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			 for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
			      out << imag(sigma_->get_matsubara_frequency(freq_index)) << "     "
				  << real(bare_greens_function[freq_index].block(
					       site_index * per_site_orbital_size,
					       site_index * per_site_orbital_size,
					       per_site_orbital_size,
					       per_site_orbital_size)(orb1, orb2)) << "     "
				  << imag(bare_greens_function[freq_index].block(
					       site_index * per_site_orbital_size,
					       site_index * per_site_orbital_size,
					       per_site_orbital_size,
					       per_site_orbital_size)(orb1, orb2)) << endl;
			 }
		    }
	       }
	  }
	  out.close();
	  // dump non shifted bare GF in Matsubara frequencies
	  out.open(bare_gf_no_shift_dump_name, ofstream::out);
	  out.precision(output_precision);
	  out << fixed << setprecision(output_precision);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
		    for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			 for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
			      out << imag(sigma_->get_matsubara_frequency(freq_index)) << "     "
				  << real(no_shift_bare_greens_function[freq_index].block(
					       site_index * per_site_orbital_size,
					       site_index * per_site_orbital_size,
					       per_site_orbital_size,
					       per_site_orbital_size)(orb1, orb2)) << "     "
				  << imag(no_shift_bare_greens_function[freq_index].block(
					       site_index * per_site_orbital_size,
					       site_index * per_site_orbital_size,
					       per_site_orbital_size,
					       per_site_orbital_size)(orb1, orb2)) << endl;
			 }
		    }
	       }
	  }
	  out.close();
	  // dump shift...
	  // dump -mu_tilde, so that the fortran code is happy...
	  out.open(shift_dump_name, ofstream::out);
	  out.precision(output_precision);
	  out << fixed << setprecision(output_precision);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
		    out << -real(mu_tilde.block(
				      site_index * per_site_orbital_size,
				      site_index * per_site_orbital_size,
				      per_site_orbital_size,
				      per_site_orbital_size)(orb1, orb1)) << endl;
	       }
	  }
	  out.close();
	  out.open(shift_sq_dump_name, ofstream::out);
	  out.precision(output_precision);
	  out << fixed << setprecision(output_precision);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
		    out << real(lattice_bs_->get_epsilon_squared_bar().block(
				     site_index * per_site_orbital_size,
				     site_index * per_site_orbital_size,
				     per_site_orbital_size,
				     per_site_orbital_size)(orb1, orb1)) << endl;
	       }
	  }
	  out.close();
	  // dump 1st and 2nd moments of the local bath function
	  out.open(mom1_dump_name, ofstream::out);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
                    for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
                         out << real(world_bath_moment_1.block(
                                          site_index * per_site_orbital_size,
                                          site_index * per_site_orbital_size,
                                          per_site_orbital_size,
                                          per_site_orbital_size)(orb1, orb2)) <<
                              "   " << imag(world_bath_moment_1.block(
                                                 site_index * per_site_orbital_size,
                                                 site_index * per_site_orbital_size,
                                                 per_site_orbital_size,
                                                 per_site_orbital_size)(orb1, orb2))<< endl;
                    }
	       }
	  }
	  out.close();
          out.open(mom2_dump_name, ofstream::out);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
                    for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
                         out << real(world_bath_moment_2.block(
                                          site_index * per_site_orbital_size,
                                          site_index * per_site_orbital_size,
                                          per_site_orbital_size,
                                          per_site_orbital_size)(orb1, orb2)) <<
                              "   " << imag(world_bath_moment_2.block(
                                                 site_index * per_site_orbital_size,
                                                 site_index * per_site_orbital_size,
                                                 per_site_orbital_size,
                                                 per_site_orbital_size)(orb1, orb2)) << endl;
                    }
	       }
	  }
	  out.close();

     }
     MPI_Barrier(MPI_COMM_WORLD);
}

void HybFunction::dump_delta_for_matrix() {
	if (world_rank_ == 0) {
		ofstream out(imaginary_time_dump_name_for_matrix, ofstream::out);
		out.precision(output_precision);
		out << fixed << setprecision(output_precision);
		//out.open(imaginary_time_dump_name_for_matrix);
		out << fixed << setprecision(output_precision);
		for(size_t site_index = 0; site_index < n_sites; site_index++) {
			for (size_t tau_index = 0; tau_index < delta_tau.size(); tau_index++) {
				for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
					for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
						// CAREFUL! delta_tau has been extended to include tau = beta
						// it has one more element than should be for a proper definition of
						// tau_value:
						// Also note: sign convention is different between our QMC,
						// and ALps3 QMC for hybridization function, hence the sign difference
						// below
						out << tau_index << "  " << orb1 << "  "
						    << orb2 << "  "  
						    << real(delta_tau[tau_index].block(
								     site_index * per_site_orbital_size,
								     site_index * per_site_orbital_size,
								     per_site_orbital_size,
								     per_site_orbital_size)(orb1, orb2)) << "  "
						    << imag(delta_tau[tau_index].block(
								     site_index * per_site_orbital_size,
								     site_index * per_site_orbital_size,
								     per_site_orbital_size,
								     per_site_orbital_size)(orb1, orb2)) << endl;
					}
				}
			}
		}
		out.close();
		// dump bare GF in Matsubara frequencies
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void HybFunction::dump_G0_hdf5(alps::hdf5::archive &h5_archive) {
     if (world_rank_ == 0) {
	  size_t N_max = sigma_->get_n_matsubara_freqs();
	  cplx_array_type temp_g0(boost::extents[tot_orbital_size][tot_orbital_size][N_max]);
	  for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	       for(size_t site_index = 0; site_index < n_sites; site_index++) {
		    Eigen::MatrixXcd temp = no_shift_bare_greens_function[freq_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size);
		    for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
			 for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			      temp_g0[orb1 + site_index * per_site_orbital_size]
				   [orb2 + site_index * per_site_orbital_size][freq_index] =
				   temp(orb1, orb2);
			 }
		    }
	       }
	  }
	  h5_archive["/G0/data"] = temp_g0;
     }
}

void HybFunction::dump_G0_for_ctint_hdf5(alps::hdf5::archive &h5_archive) {
     if (world_rank_ == 0) {
	  size_t N_max = sigma_->get_n_matsubara_freqs();
	  cplx_array_type temp_g0(boost::extents[tot_orbital_size][tot_orbital_size][N_max]);
	  for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	       for(size_t site_index = 0; site_index < n_sites; site_index++) {
		    Eigen::MatrixXcd temp = G0_function[freq_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size);
		    for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
			 for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			      temp_g0[orb1 + site_index * per_site_orbital_size]
				   [orb2 + site_index * per_site_orbital_size][freq_index] =
				   -temp(orb1, orb2);
			 }
		    }
	       }
	  }
	  h5_archive["/G0_CTINT"] = temp_g0;
	  for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	       for(size_t site_index = 0; site_index < n_sites; site_index++) {
		    Eigen::MatrixXcd temp = pure_no_shift_bare_greens_function[freq_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size);
		    for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
			 for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			      temp_g0[orb1 + site_index * per_site_orbital_size]
				   [orb2 + site_index * per_site_orbital_size][freq_index] =
				   temp(orb1, orb2);
			 }
		    }
	       }
	  }
	  h5_archive["/G0_lattice_CTINT"] = temp_g0;

     }
}

void HybFunction::dump_delta_hdf5() {
     if (world_rank_ == 0) {
	  double beta = sigma_->get_beta();
	  if (enforce_real) {
	       cout << "Enforcing REAL DELTA function" << endl;
	  }
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       int orb_index = 0;
	       std::string archive_name = imaginary_time_hdf5_root + "_"
		    + boost::lexical_cast<std::string>(site_index) + ".h5";
	       alps::hdf5::archive delta_output(archive_name, "a");
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
		    for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++, orb_index++) {
			 std::vector<complex<double> > delta_function;
			 for (size_t tau_index = 0; tau_index < delta_tau.size(); tau_index++) {
			      // CAREFUL! delta_tau has been extended to include tau = beta
			      // it has one more element than should be for a proper definition of
			      // tau_value:
			      if (enforce_real) {
				   delta_function.push_back(
					-delta_tau[tau_index].
					block(site_index * per_site_orbital_size,
					      site_index * per_site_orbital_size,
					      per_site_orbital_size,
					      per_site_orbital_size).real()(orb1, orb2));
			      } else {
				   delta_function.push_back(
					-delta_tau[tau_index].
					block(site_index * per_site_orbital_size,
					      site_index * per_site_orbital_size,
					      per_site_orbital_size,
					      per_site_orbital_size)(orb1, orb2));
			      }
			 }
			 std::stringstream h5_group_name;
			 h5_group_name << "/Delta_" << orb_index;
			 delta_output << alps::make_pvp(h5_group_name.str(), delta_function);
		    }
	       }
	       delta_output.close();
	  }
     }
     MPI_Barrier(MPI_COMM_WORLD);
}

void HybFunction::dump_bubble_hdf5() {
     if (world_rank_ == 0) {
	  std::string archive_name = bubble_hdf5_root + ".h5";
	  alps::hdf5::archive bubble_output(archive_name, "a");
	  std::string h5_group_name("/local_bubble");
	  for (int site_index = 0; site_index < n_sites; site_index++) {
	       for(int boson_index = 0; boson_index < N_boson; boson_index++) {
		    std::stringstream site_path;
		    site_path << h5_group_name + "/site_" +
			 boost::lexical_cast<std::string>(site_index) + "/" +
			 "boson_" + boost::lexical_cast<std::string>(boson_index) + "/";
		    for(int line_idx = 0;
			line_idx < per_site_orbital_size * per_site_orbital_size;
			++line_idx) {
			 for(int col_idx = 0;
			     col_idx < per_site_orbital_size * per_site_orbital_size;
			     ++col_idx) {
			      std::stringstream orbital_path;
			      int part_index_1 = line_idx / per_site_orbital_size;
			      int hole_index_2 = line_idx % per_site_orbital_size;
			      int part_index_2 = col_idx / per_site_orbital_size;
			      int hole_index_1 = col_idx % per_site_orbital_size;
			      orbital_path << site_path.str() <<
				   boost::lexical_cast<std::string>(part_index_1) + "/"
				   + boost::lexical_cast<std::string>(hole_index_2) + "/"
				   + boost::lexical_cast<std::string>(part_index_2) + "/"
				   + boost::lexical_cast<std::string>(hole_index_1) + "/value";
			      std::vector<std::complex<double>> temp_data;
			      temp_data.resize(bubble_dim);
			      for (int freq_index = 0; freq_index < bubble_dim; freq_index++) {
				   temp_data[freq_index] =
					world_local_bubble[boson_index][freq_index].block(
					     site_index * per_site_orbital_size *
					     per_site_orbital_size,
					     site_index * per_site_orbital_size *
					     per_site_orbital_size,
					     per_site_orbital_size *
					     per_site_orbital_size,
					     per_site_orbital_size *
					     per_site_orbital_size)(line_idx, col_idx);
			      }
			      bubble_output << alps::make_pvp(orbital_path.str(), temp_data);
			 }
		    }
	       }
	  }
	  std::string h5_group_name_2("/lattice_bubble");
	  for (int site_index = 0; site_index < n_sites; site_index++) {
	       for(int boson_index = 0; boson_index < N_boson; boson_index++) {
		    for(int q_index = 0;
			q_index < lattice_bs_->get_nb_points_for_bseq(); q_index++) {
			 std::stringstream site_path;
			 site_path <<
			      h5_group_name_2 +
			      "/site_" + boost::lexical_cast<std::string>(site_index) + "/" +
			      "boson_" + boost::lexical_cast<std::string>(boson_index) + "/" +
			      "q_" + boost::lexical_cast<std::string>(q_index) + "/";
			 for(int line_idx = 0;
			     line_idx < per_site_orbital_size * per_site_orbital_size;
			     ++line_idx) {
			      for(int col_idx = 0;
				  col_idx < per_site_orbital_size * per_site_orbital_size;
				  ++col_idx) {
				   std::stringstream orbital_path;
				   int part_index_1 = line_idx / per_site_orbital_size;
				   int hole_index_2 = line_idx % per_site_orbital_size;
				   int part_index_2 = col_idx / per_site_orbital_size;
				   int hole_index_1 = col_idx % per_site_orbital_size;
				   orbital_path << site_path.str() <<
					boost::lexical_cast<std::string>(part_index_1) + "/"
					+ boost::lexical_cast<std::string>(hole_index_2) + "/"
					+ boost::lexical_cast<std::string>(part_index_2) + "/"
					+ boost::lexical_cast<std::string>(hole_index_1) + "/value";
				   std::vector<std::complex<double>> temp_data;
				   temp_data.resize(bubble_dim);
				   for (int freq_index = 0; freq_index < bubble_dim; freq_index++) {
					temp_data[freq_index] =
					     world_lattice_bubble[boson_index][q_index][freq_index].block(
						  site_index * per_site_orbital_size *
						  per_site_orbital_size,
						  site_index * per_site_orbital_size *
						  per_site_orbital_size,
						  per_site_orbital_size *
						  per_site_orbital_size,
						  per_site_orbital_size *
						  per_site_orbital_size)(line_idx, col_idx);
				   }
				   bubble_output << alps::make_pvp(orbital_path.str(), temp_data);
			      }
			 }
		    }
	       }
	  }
	  h5_group_name_2 = "/lattice_bubble/q_point_list";
	  std::vector<std::complex<double>> temp_data;
	  temp_data.resize(lattice_bs_->get_nb_points_for_bseq());
	  int nb_q_points = lattice_bs_->get_nb_points_for_bseq();
	  Eigen::VectorXd q_point;
	  for (int q_index = 0; q_index < nb_q_points; q_index++) {
	       q_point = lattice_bs_->get_q_point(q_index);
	       temp_data[q_index] = std::complex<double>(q_point(0), q_point(1));
	  }
	  bubble_output << alps::make_pvp(h5_group_name_2, temp_data);
	  bubble_output.close();
     }
     MPI_Barrier(MPI_COMM_WORLD);
}

void HybFunction::elementary_compute_delta_tau() {
     double beta = sigma_->get_beta();
     size_t N_max = sigma_->get_n_matsubara_freqs();
     // We hold of tau-index vector of
     // Eigen matrices in the orbital dimension for the sum
     // of analytical HF contributions
     vector<Eigen::MatrixXcd> hf_analytical_contribs;
     vector<Eigen::MatrixXcd> tail_adjustments;
     vector<Eigen::MatrixXcd> neg_tail_adjustments;
     vector<double> analytical_values;
     vector<double> tau_values;
     // Initialize the output Eigen object, and pre-compute the analytical
     // expression of the FT of the HF expansion of Delta(i omega_n)
     analytical_values.resize(max_expansion_order);
     for (size_t tau_index = 0; tau_index < n_tau; tau_index++) {
	  double tau_value = beta * static_cast<double>(tau_index) / n_tau;
	  tau_values.push_back(tau_value);
	  delta_tau.push_back(Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size));
	  analytical_values[0] = -0.5;
	  analytical_values[1] = (2.0 * tau_value  - beta) / 4.0;
	  analytical_values[2] = tau_value * (beta - tau_value) / 4.0;

	  hf_analytical_contribs.push_back(Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size));
	  for (size_t i = 0; i < max_expansion_order; i++) { 
	       for (size_t site_index = 0; site_index < n_sites; site_index++) {
		    hf_analytical_contribs[tau_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) +=
			 hf_coeff[i].block(
			      site_index * per_site_orbital_size,
			      site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size) * analytical_values[i];
	       }
	  }
     }

     // precompute the hf contribution to the finite sum over
     // matsubara frequencies.
     for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	  tail_adjustments.push_back(Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size));
	  neg_tail_adjustments.push_back(Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size));
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {	  
	       for (size_t i = 0; i < max_expansion_order; i++) {
		    tail_adjustments[freq_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) += hf_coeff[i].block(
			      site_index * per_site_orbital_size,
			      site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size) /
			 pow(sigma_->get_matsubara_frequency(freq_index), i+1);
		    neg_tail_adjustments[freq_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) += hf_coeff[i].block(
			      site_index * per_site_orbital_size,
			      site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size) /
			 pow(-sigma_->get_matsubara_frequency(freq_index), i+1);
	       }
	  }
     }

     for (size_t tau_index = 0; tau_index < n_tau; tau_index++) {
	  for (size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
		    double phase = tau_values[tau_index] *
			 imag(sigma_->get_matsubara_frequency(freq_index));
		    complex<double> phase_factor = exp(-complex<double>(0.0, 1.0) * phase);
		    delta_tau[tau_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) +=
			 (((hybridization_function[freq_index].block(
				 site_index * per_site_orbital_size,
				 site_index * per_site_orbital_size,
				 per_site_orbital_size,
				 per_site_orbital_size) -
			    tail_adjustments[freq_index].block(
				 site_index * per_site_orbital_size,
				 site_index * per_site_orbital_size,
				 per_site_orbital_size,
				 per_site_orbital_size)) * phase_factor
			   +
			   (hybridization_function[freq_index].block(
				site_index * per_site_orbital_size,
				site_index * per_site_orbital_size,
				per_site_orbital_size,
				per_site_orbital_size).transpose().conjugate() -
			    neg_tail_adjustments[freq_index].block(
				 site_index * per_site_orbital_size,
				 site_index * per_site_orbital_size,
				 per_site_orbital_size,
				 per_site_orbital_size)) / phase_factor) / beta);
	       }
	       delta_tau[tau_index].block(
		    site_index * per_site_orbital_size,
		    site_index * per_site_orbital_size,
		    per_site_orbital_size,
		    per_site_orbital_size) +=
		    hf_analytical_contribs[tau_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size);
	  }
     }

     // Add the value for tau = beta.
     // Careful, we are dealing with the hybridization function, not a standard Green's function.
     // Using the A.5 - A.8 decomposition from Ferber as an infinite sum, and noticing that all
     // powers of n are continuous at tau = 0, except power 1
     // For the power 1 contribution, we get F_1(tau = 0-) = -F_1(tau = 0+),
     // Hence F(0-) - F(0+) = 2 F(0-) = 2*1/2*c_1 = c_1
     // Then, F(beta-) = -F(0-) = -(c_1 + F(0+))
     delta_tau.push_back(-hf_coeff[0] - delta_tau[0]);
}

const size_t HybFunction::output_precision = 16;
const size_t HybFunction::tail_fit_length = 10;
const size_t HybFunction::max_expansion_order = 3;
const string HybFunction::matsubara_frequency_dump_name = "c_delta.w";
const string HybFunction::matsubara_bare_gf_dump_name = "c_gw";
const string HybFunction::bare_gf_no_shift_dump_name = "c_bare_gf";
const string HybFunction::imaginary_time_dump_name = "c_delta.tau";
const string HybFunction::imaginary_time_dump_name_for_matrix = "ec_delta.tau";
const string HybFunction::imaginary_time_hdf5_root = "c_delta";
const string HybFunction::bubble_hdf5_root = "c_bubble";
const string HybFunction::shift_dump_name = "c_shift.tmp";
const string HybFunction::shift_sq_dump_name = "c_shift_sq.tmp";
const string HybFunction::matsubara_self_energy_name = "current_sigma";
const string HybFunction::legendre_self_energy_name = "current_legendre_sigma";
const string HybFunction::mom1_dump_name = "c_mom1.tmp";
const string HybFunction::mom2_dump_name = "c_mom2.tmp";
