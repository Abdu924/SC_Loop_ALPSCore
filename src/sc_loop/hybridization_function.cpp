#include "hybridization_function.hpp"
#include <boost/timer/timer.hpp>
#include <alps/hdf5/multi_array.hpp>

using namespace std;
typedef boost::multi_array<complex<double> , 3> cplx_array_type;

HybFunction::HybFunction(const alps::params &parms,
			 boost::shared_ptr<Bandstructure> const &lattice_bs,
			 boost::shared_ptr<Selfenergy> const &sigma,
			 complex<double> chemical_potential, int world_rank,
			 bool verbose):
     lattice_bs_(lattice_bs), sigma_(sigma), world_rank_(world_rank),
     enforce_real(sigma->get_enforce_real()),
     chemical_potential(chemical_potential) {
     boost::timer::auto_cpu_timer hyb_calc; 
     n_tau = static_cast<size_t>(parms["N_TAU"]);
     n_sites = sigma_->get_n_sites();
     per_site_orbital_size = sigma_->get_per_site_orbital_size();
     tot_orbital_size = n_sites * per_site_orbital_size;
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
     bare_g_hf_coeff.clear();
     // Store the first order coefficient
     hf_coeff.push_back(first_order);
     // Then compute the full hybridization function...
     // do *NOT* remove this computation from the constructor -
     compute_hybridization_function(chemical_potential);
     // And extract the 2nd and 3rd order coefficients from the numerics.
     compute_superior_orders(verbose);
     elementary_compute_delta_tau();
     compute_bare_g_superior_orders(verbose);
     elementary_compute_G_tau();
     dump_delta();
     dump_Gtau_for_HF();
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

void HybFunction::compute_bare_g_superior_orders(bool verbose) {
     Eigen::MatrixXcd bare_first_order = Eigen::VectorXcd::Constant(tot_orbital_size, 1.0).asDiagonal();
     bare_g_hf_coeff.push_back(bare_first_order);     
     bare_g_hf_coeff.push_back(world_bath_moment_1);
     bare_g_hf_coeff.push_back(world_bath_moment_2);
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
				  << "   " <<
                                   -imag(delta_tau[tau_index].block(
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
	  out.open(GfBase::bare_gf_no_shift_dump_name, ofstream::out);
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
          out.open(hf_shift_dump_name, ofstream::out);
	  out.precision(output_precision);
	  out << fixed << setprecision(output_precision);
	  for(size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size / 2; orb1++) {
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
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size / 2; orb1++) {
                    for (size_t orb2 = 0; orb2 < per_site_orbital_size / 2; orb2++) {
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
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size / 2; orb1++) {
                    for (size_t orb2 = 0; orb2 < per_site_orbital_size / 2; orb2++) {
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

void HybFunction::dump_Gtau_for_HF() {
     double beta = sigma_->get_beta();
     if (world_rank_ == 0) {
          ofstream out(imaginary_time_dump_name_for_hf, ofstream::out);
          out.precision(output_precision);
          out << fixed << setprecision(output_precision);
          //out.open(imaginary_time_dump_name_for_matrix);
          out << fixed << setprecision(output_precision);
          for(size_t site_index = 0; site_index < n_sites; site_index++) {
               for (size_t orb1 = 0; orb1 < per_site_orbital_size / 2; orb1++) {
                    for (size_t orb2 = 0; orb2 < per_site_orbital_size / 2; orb2++) {
                         for (size_t tau_index = 0; tau_index < G_tau.size(); tau_index++) {
                              // CAREFUL! delta_tau has been extended to include tau = beta
                              // it has one more element than should be for a proper definition of
                              // tau_value:
                              // Also note: sign convention is different between our QMC,
                              // and ALps3 QMC for hybridization function, hence the sign difference
                              // below
                              out << beta * static_cast<double>(tau_index) / (G_tau.size() - 1)  <<  "  "  
                                  << -real(G_tau[tau_index].block(
                                               site_index * per_site_orbital_size,
                                               site_index * per_site_orbital_size,
                                               per_site_orbital_size,
                                               per_site_orbital_size)(orb1, orb2)) << "  "
                                  << -imag(G_tau[tau_index].block(
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

void HybFunction::elementary_compute_G_tau() {
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
	  G_tau.push_back(Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size));
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
			 bare_g_hf_coeff[i].block(
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
			 per_site_orbital_size) += bare_g_hf_coeff[i].block(
			      site_index * per_site_orbital_size,
			      site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size) /
			 pow(sigma_->get_matsubara_frequency(freq_index), i+1);
		    neg_tail_adjustments[freq_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) += bare_g_hf_coeff[i].block(
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
		    G_tau[tau_index].block(
			 site_index * per_site_orbital_size,
			 site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) +=
			 (((bare_greens_function[freq_index].block(
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
			   (bare_greens_function[freq_index].block(
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
	       G_tau[tau_index].block(
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
}

const size_t HybFunction::output_precision = 16;
const size_t HybFunction::tail_fit_length = 10;
const size_t HybFunction::max_expansion_order = 3;
const string HybFunction::matsubara_frequency_dump_name = "c_delta.w";
const string HybFunction::matsubara_bare_gf_dump_name = "c_gw";
const string HybFunction::imaginary_time_dump_name = "c_delta.tau";
const string HybFunction::imaginary_time_dump_name_for_matrix = "ec_delta.tau";
const string HybFunction::imaginary_time_dump_name_for_hf = "gtau";
const string HybFunction::imaginary_time_hdf5_root = "c_delta";
const string HybFunction::shift_dump_name = "c_shift.tmp";
const string HybFunction::hf_shift_dump_name = "shift.tmp";
const string HybFunction::shift_sq_dump_name = "c_shift_sq.tmp";
const string HybFunction::mom1_dump_name = "mom1.tmp";
const string HybFunction::mom2_dump_name = "mom2.tmp";
