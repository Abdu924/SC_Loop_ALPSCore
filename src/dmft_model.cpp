#include "dmft_model.hpp"
#include <Eigen/LU>
#include<math.h>
#include <boost/timer/timer.hpp>

using namespace std;

Eigen::MatrixXcd get_pauli_matrix(int i) {
     std::complex<double> cim(0.0, 1.0);
     if (i == 0)
	  return 0.5 * (Eigen::MatrixXcd(2,2)
			<< 1.0, 0.0, 0.0, 1.0).finished();
     else if (i == 1)
	  return 0.5 * (Eigen::MatrixXcd(2,2)
			<< 0.0, 1.0, 1.0, 0.0).finished();
     else if (i == 2)
	  return 0.5 * (Eigen::MatrixXcd(2,2)
			<< 0.0, -cim, cim, 0.0).finished();
     else if (i == 3)
	  return 0.5 * (Eigen::MatrixXcd(2,2)
			<< 1.0, 0.0, 0.0, -1.0).finished();
     else
	  throw runtime_error("invalid Pauli matrix index !");
}

DMFTModel::DMFTModel(boost::shared_ptr<Bandstructure> const &lattice_bs,
		     boost::shared_ptr<Selfenergy> const &sigma,
		     const alps::params& parms, int world_rank)
     : lattice_bs_(lattice_bs), sigma_(sigma), world_rank_(world_rank) {
     size_t N_max = sigma_->get_n_matsubara_freqs();
     double omega_max = abs(sigma_->get_matsubara_frequency(N_max - 1));
     string ref_exact ("exact");
     string tail_style = parms["TAIL_STYLE"];
     compute_spin_current = parms["model.compute_spin_current"].as<bool>();
     n_tolerance = 0.1 * pow(e_max / omega_max, 3);
     target_density = static_cast<double>(parms["N_ELECTRONS"]);
     if (ref_exact.compare(tail_style) == 0) {
	  exact_tail = true;
     } else {
	  exact_tail = false;
     }
     if (world_rank_ == 0) {
	  cout << "Exact tail treatment for density: "
	       << exact_tail << endl;
     }
     size_t orbital_size = lattice_bs_->get_orbital_size();
     occupation_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     order_parameters.clear();
     int world_size = lattice_bs_->get_world_size();
     int n_points_per_proc = lattice_bs_->get_n_points_per_proc();
     world_k_resolved_occupation_matrices.resize(n_points_per_proc * world_size);
     world_k_resolved_xcurrent_matrices.resize(n_points_per_proc * world_size);
     world_k_resolved_ycurrent_matrices.resize(n_points_per_proc * world_size);
     for (int k_index = 0; k_index < n_points_per_proc * world_size; k_index++) {
	  world_k_resolved_occupation_matrices[k_index].resize(
	       orbital_size, orbital_size);
	  world_k_resolved_xcurrent_matrices[k_index].resize(
	       orbital_size, orbital_size);
	  world_k_resolved_ycurrent_matrices[k_index].resize(
	       orbital_size, orbital_size);
     }
     // hf-expansion coefficients
     // Careful! c_1 = 1.0 only for diagonal matrix elements of GF.
     c_1 = Eigen::VectorXcd::Constant(orbital_size, 1.0).asDiagonal();
     Eigen::MatrixXcd sigma_0 = sigma->get_sigma_0();
     Eigen::MatrixXcd sigma_1 = sigma->get_sigma_1();
     Eigen::VectorXcd freqs = sigma->get_matsubara_frequencies();
     boost::shared_ptr<TailManager> temp_tail_manager(
	  new TailManager(sigma_0, sigma_1, freqs, world_rank_));
     tail_manager = temp_tail_manager;
}

tuple<int, double, double> DMFTModel::get_mu_from_density_bisec(double initial_mu,
                                                                double mu_increment) {
     boost::timer::auto_cpu_timer mu_calc;     
     if (world_rank_ == 0) {
	  cout << "****************************************" << endl;
	  cout << "REVERTING TO BISECTION METHOD" << endl;
	  cout << "****************************************" << endl;
	  cout << "SEARCH FOR UPPER AND LOWER BOUNDS: " << endl;
	  cout << "Initial_mu = " <<
	       initial_mu << endl;
	  cout << "Mu increment = " << mu_increment << endl;
	  cout << "Max iterations: " << max_iter_for_bisec << endl;
	  cout << "n_tot tolerance " << n_tolerance << endl;
	  cout << "target density " << target_density << endl;	  
     }
     int success = 0;
     double old_density, cur_density, cur_mu, cur_derivative, bracket_mu;
     double mu_min, mu_max;
     size_t iteration_idx(0);
     cur_mu = initial_mu;
     tie(cur_density, cur_derivative) = get_particle_density(cur_mu, false);
     old_density = cur_density;
     if (world_rank_ == 0) {
	  cout << "current density: " << cur_density << endl;
     }
     if (cur_density > target_density) {
	  mu_increment = std::min(-0.05, -abs(mu_increment));
     } else {
	  mu_increment = std::max(0.05, abs(mu_increment));
     }
     success = check_density_success(cur_density);
     MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
     // current value of mu is not satisfactory, we have to bracket.
     if (success == 0) {
	  while (((cur_density - target_density) * (old_density - target_density) > 0.0) &&
		 (iteration_idx < max_iter_for_bounds) &&
		 (success == 0)) {
	       cur_mu +=  mu_increment;
	       old_density = cur_density;
	       tie(cur_density, cur_derivative) = get_particle_density(cur_mu, false);
	       iteration_idx++;
	       if (world_rank_ == 0) {
		    cout << "new mu: " << cur_mu << endl;
		    cout << "new density " << cur_density << endl;
		    cout << "cur iteration for bounds: " << iteration_idx << endl;
	       }
	       success = check_density_success(cur_density);
	       MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
	  }
	  // Exit on bracketing failure
	  if (world_rank_ == 0) {
	       if ((iteration_idx >= max_iter_for_bounds) &&
		   (success == 0)) {
		    cout << "the bounds were not found...FAILURE" << endl;
		    throw runtime_error("Unable to find bounds for my in bisec !");
	       }
	  }
	  // Bracketing was a success - define bounds.
	  // Best candidate for mu is the value at the previous iteration
	  bracket_mu = cur_mu - mu_increment;
	  mu_min = min(bracket_mu, cur_mu);
	  mu_max = max(bracket_mu, cur_mu);
	  if (world_rank_ == 0) {
	       cout << "FOUND BOUNDS FOR MU " << mu_min
		    << "  " << mu_max << " after " << iteration_idx
		    << " density computations." << endl;
	  }
	  iteration_idx = 1;
	  // if density is not close enough to target, move on to the bisection
	  while((success == 0) &&
		(iteration_idx < max_iter_for_bisec)) {
	       cur_mu = 0.5 * (mu_min + mu_max);
	       tie(cur_density, cur_derivative) = get_particle_density(cur_mu, false);
	       if (world_rank_ == 0) {
		    cout << "Starting bisection " << mu_min << endl;
		    cout << "new mu: " << cur_mu << endl;
		    cout << "new density " << cur_density << endl;
		    cout << "cur iteration for bisec: " << iteration_idx << endl << endl;
	       }
	       if (cur_density > target_density) {
		    mu_max = cur_mu;
	       } else {
		    mu_min = cur_mu;
	       }
	       success = check_density_success(cur_density);
	       MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
	       iteration_idx++;
	  }
     }
     // Exit on bisection failure.
     if ((success == 0) && (iteration_idx >= max_iter_for_bisec)) {
	  cout << "chem. pot.: too many iterations - STOP" << endl;
	  throw runtime_error("Unable to find bounds for chemical potential in bisec !");
     } else {
	  if (world_rank_ == 0) {
	       cout << "Bisection successful " << endl;
	       cout << iteration_idx << " iterations in bisection" << endl;
	  }
     }
     // Success - gather the k-resolved occupation matrices.
     scatter_occ_matrices();
     if (compute_spin_current == true) {
	  scatter_xcurrent_matrices();
	  scatter_ycurrent_matrices();
     }
     return tuple<int, double, double>(success, cur_mu, cur_density);
}

tuple<int, double, double, double> DMFTModel::get_mu_from_density(double initial_mu) {
     boost::timer::auto_cpu_timer mu_calc;
     if (world_rank_ == 0) {
	  cout << "**********************************************" << endl;
	  cout << "** CALCULATION OF CHEM. POTENTIAL : NEWTON ***" << endl;
	  cout << "**********************************************" << endl;
	  cout << "n_tot tolerance " << n_tolerance << endl;
	  cout << "target density " << target_density << endl;
     }
     bool success = 0;
     double cur_density, cur_derivative, cur_mu, delta_mu;
     size_t iteration_idx(1);
     cur_mu = initial_mu;
     tie(cur_density, cur_derivative) = get_particle_density(cur_mu, true);
     if (world_rank_ == 0) {
	  cout << iteration_idx << "  " << initial_mu << "  "
	       << cur_density << endl;
     }
     delta_mu = (target_density - cur_density) / cur_derivative;
     // We could test for success and suppress the following
     // calculation in case success is already
     // reached. But keeping this as is allows the convergence
     // to become tighter as the DMFT loops are run.
     // Otherwise, the system may remain as far as n_tolerance
     // away from the target, indefinitely.
     cur_mu = initial_mu + delta_mu;
     for(iteration_idx = 2; iteration_idx < max_iter_for_newton; iteration_idx++) {
	  tie(cur_density, cur_derivative) = get_particle_density(cur_mu, true);
	  if (world_rank_ == 0) {	  
	       cout << iteration_idx << "  " << cur_mu << "  "
		    << cur_density << endl;
	  }
	  success = check_density_success(cur_density);
	  MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
	  if (success == 1) {
	       break;
	  } else {
	       delta_mu = (target_density - cur_density) / cur_derivative;
	       cur_mu += delta_mu;
	  }
     }
     if (success == 1) {
	  // Success - scatter/gather the k-resolved occupation matrices.
	  scatter_occ_matrices();
	  if (compute_spin_current == true) {
	       scatter_xcurrent_matrices();
	       scatter_ycurrent_matrices();
	  }
     }
     if (world_rank_ == 0) {     
	  cout << " mu_calc + energy - timing: ";
     }
     return tuple<int, double, double, double>(success, cur_mu, cur_density, delta_mu);
}

void DMFTModel::scatter_occ_matrices() {
     int world_size = lattice_bs_->get_world_size();
     int n_points_per_proc = lattice_bs_->get_n_points_per_proc();     
     for (size_t k_index = 0; k_index < n_points_per_proc; k_index++) {
	  if (world_rank_ == 0) {
	       for (int proc_index = 1; proc_index < world_size; proc_index++) {
		    MPI_Recv(
			 world_k_resolved_occupation_matrices
			 [proc_index * n_points_per_proc + k_index].data(),
			 k_resolved_occupation_matrices[k_index].size(),
			 MPI_DOUBLE_COMPLEX, proc_index, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	       }
	  } else {
	       MPI_Send(k_resolved_occupation_matrices[k_index].data(),
			k_resolved_occupation_matrices[k_index].size(),
			MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);	       
	  }
	  if (world_rank_ == 0) {
	       world_k_resolved_occupation_matrices[k_index] =
		    k_resolved_occupation_matrices[k_index];
	  }
     }
}

void DMFTModel::scatter_xcurrent_matrices() {
     int world_size = lattice_bs_->get_world_size();
     int n_points_per_proc = lattice_bs_->get_n_points_per_proc();     
     for (size_t k_index = 0; k_index < n_points_per_proc; k_index++) {
	  if (world_rank_ == 0) {
	       for (int proc_index = 1; proc_index < world_size; proc_index++) {
		    MPI_Recv(
			 world_k_resolved_xcurrent_matrices
			 [proc_index * n_points_per_proc + k_index].data(),
			 k_resolved_xcurrent_matrices[k_index].size(),
			 MPI_DOUBLE_COMPLEX, proc_index, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	       }
	  } else {
	       MPI_Send(k_resolved_xcurrent_matrices[k_index].data(),
			k_resolved_xcurrent_matrices[k_index].size(),
			MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);	       
	  }
	  if (world_rank_ == 0) {
	       world_k_resolved_xcurrent_matrices[k_index] =
		    k_resolved_xcurrent_matrices[k_index];
	  }
     }
}

void DMFTModel::scatter_ycurrent_matrices() {
     int world_size = lattice_bs_->get_world_size();
     int n_points_per_proc = lattice_bs_->get_n_points_per_proc();     
     for (size_t k_index = 0; k_index < n_points_per_proc; k_index++) {
	  if (world_rank_ == 0) {
	       for (int proc_index = 1; proc_index < world_size; proc_index++) {
		    MPI_Recv(
			 world_k_resolved_ycurrent_matrices
			 [proc_index * n_points_per_proc + k_index].data(),
			 k_resolved_ycurrent_matrices[k_index].size(),
			 MPI_DOUBLE_COMPLEX, proc_index, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	       }
	  } else {
	       MPI_Send(k_resolved_ycurrent_matrices[k_index].data(),
			k_resolved_ycurrent_matrices[k_index].size(),
			MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);	       
	  }
	  if (world_rank_ == 0) {
	       world_k_resolved_ycurrent_matrices[k_index] =
		    k_resolved_ycurrent_matrices[k_index];
	  }
     }
}

int DMFTModel::check_density_success(double cur_density) {
     int output = 0;
     if (world_rank_ == 0) {
	  output =
	       (abs(target_density - cur_density) < n_tolerance) ?
	       1 : 0;
     }
     return output;
}

/*
  Compute the exact contribution from first two orders in 1 / iomega_n
  Using the analytical expression of G(tau), and the coefficients
  c_1 = 1 and c_2, Cf expression in Gull, B.23 and B.34
  The following is B.23 for tau = 0-
*/
void DMFTModel::compute_tail_contribution(int k_index, size_t orbital_size,
					  double chemical_potential, double beta) {
     double order2_partial_sum = sigma_->get_order2_partial_sum();
     c_2 = Eigen::VectorXcd::Constant(
	  orbital_size, -chemical_potential).asDiagonal();
     c_2 += lattice_bs_->dispersion_[k_index]
	  + sigma_->get_sigma_0();
     // From this result, we subtract the contributions from the orders 1 and 2 in
     // 1 / i omega_n, for n between -Nmax and +Nmax,
     // since such contributions are included in the numerical 
     // explicit calculation of the partial summation of G(i omega_n)_-Nmax^Nmax
     // Note that the contribution of the naive summation
     // for odd orders in 1 / i omega_n is zero by symmetry.
     k_resolved_occupation_matrices.push_back(
	  (-c_1 / 2.0 - c_2 * beta / 4.0 - 2.0 * c_2 * order2_partial_sum / beta));
}

void DMFTModel::compute_analytical_tail(double chemical_potential, int k_index, double beta) {
     tail_manager->set_chemical_potential(chemical_potential);
     tail_manager->set_current_k(lattice_bs_->dispersion_[k_index]);
     //tail_manager->get_analytical_contribution(beta);
     k_resolved_occupation_matrices.push_back(
	  tail_manager->get_analytical_contribution(beta));
}

double DMFTModel::compute_derivative_tail(size_t orbital_size, double beta) {
     size_t k_min(0);
     size_t k_max(lattice_bs_->get_lattice_size());
     double weight_sum = lattice_bs_->get_weight_sum(k_min, k_max);     
     double dn_dmu(0.0);
     double order2_partial_sum(sigma_->get_order2_partial_sum());
     // terms of the analytical part of the expression
     // giving a non-zero contribution when differentiated
     //wrt chemical potential.
     dn_dmu = orbital_size * weight_sum * beta / 4.0 +
	  2.0 * orbital_size * weight_sum * order2_partial_sum / beta;
     return dn_dmu;
}

void DMFTModel::reset_current_matrices(size_t orbital_size)  {
     // Reset current matrix
     spin_current_matrix.clear();
     spin_current_matrix.push_back(Eigen::MatrixXcd::Zero(orbital_size, orbital_size));
     spin_current_matrix.push_back(Eigen::MatrixXcd::Zero(orbital_size, orbital_size));
     world_spin_current_matrix.clear();
     world_spin_current_matrix.push_back(Eigen::MatrixXcd::Zero(orbital_size, orbital_size));
     world_spin_current_matrix.push_back(Eigen::MatrixXcd::Zero(orbital_size, orbital_size));
}


void DMFTModel::reset_occupation_matrices(size_t orbital_size)  {
     // Reset occupation matrix
     occupation_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     world_occupation_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     k_resolved_occupation_matrices.clear();
     k_resolved_xcurrent_matrices.clear();
     k_resolved_ycurrent_matrices.clear();
}

Eigen::MatrixXcd DMFTModel::get_full_greens_function(double chemical_potential) {
     size_t k_min(0);
     size_t k_max = lattice_bs_->get_lattice_size();
     size_t orbital_size = lattice_bs_->get_orbital_size();
     double beta = sigma_->get_beta();
     size_t N_max = sigma_->get_n_matsubara_freqs();
     Eigen::MatrixXcd from_zero_plus_to_zero_minus = Eigen::VectorXcd::Constant(
	  orbital_size, 1.0).asDiagonal();
     Eigen::MatrixXcd greens_function(orbital_size, orbital_size);
     Eigen::MatrixXcd inverse_gf(orbital_size, orbital_size);	
     std::complex<double> mu = std::complex<double>(chemical_potential);
     Eigen::VectorXcd to_add(orbital_size);
     for (int k_index = k_min; k_index < k_max; k_index++) {
	  double l_weight = lattice_bs_->get_weight(k_index);
	  if (abs(l_weight) < 1e-6) {
	       if (world_rank_ == 0) {
		    cout << "skipping k point" << endl;
	       }
	       continue;
	  }
	  for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	       to_add = Eigen::VectorXcd::Constant
		    (orbital_size, mu + sigma_->get_matsubara_frequency(freq_index));		  
	       inverse_gf = - lattice_bs_->dispersion_[k_index] - sigma_->values_[freq_index];
	       inverse_gf.diagonal() += to_add;
	       greens_function = inverse_gf.inverse();
	  }
     }
}

/*
 * For documentation, refer to thesis by E. Gull, Appendix B, in particular B.1 - B.4
 * B.22-23 and B. 34. The following Implementation follows his notation. See also
 * Ferber's thesis, A.23 ff.
 * Keep in mind that we start with the definition
 * G(k, 0-) = <c^\dagger_k c_k>, while Gull's formula is valid for
 * positive tau; We need to use G_kl(0-) = G_kl(0+) - delta_kl orbital
 * in the GF (see Maham and the comments below).
 * We also use the fact that G(k, 0+) \simeq (1 / beta) * Sum_omega_n G(k, iomega_n)
 * in order to evaluate G(k, 0-), i.e. n_k, and we sum the orbital-diagonal terms
 * over k, and over orbitals.
 */
tuple<double, double> DMFTModel::get_particle_density(double chemical_potential,
						      bool compute_derivative) {
     double density(0.0);
     double world_density(0.0);
     size_t k_min(0);
     size_t k_max = lattice_bs_->get_lattice_size();
     size_t orbital_size = lattice_bs_->get_orbital_size();
     double dn_dmu(0.0);
     double world_dn_dmu(0.0);
     double beta = sigma_->get_beta();
     size_t N_max = sigma_->get_n_matsubara_freqs();
     double partial_kinetic_energy = 0.0;
     double partial_potential_energy = 0.0;
     double cum_partial_potential_energy = 0.0;
     kinetic_energy = 0.0;
     potential_energy = 0.0;
     reset_occupation_matrices(orbital_size);
     if (compute_spin_current == true)
	  reset_current_matrices(orbital_size);
     Eigen::MatrixXcd V_matrix = lattice_bs_->get_V_matrix();
     // Here we use a diagonal matrix, because the discontinuity at tau =0
     // only exists for Green's functions involving identical orbitals.
     // For the formula related to such discontinuity, check
     // Mahan 3.2.9, p140.
     Eigen::MatrixXcd from_zero_plus_to_zero_minus = Eigen::VectorXcd::Constant(
	  orbital_size, 1.0).asDiagonal();
     if (compute_derivative == true) dn_dmu = compute_derivative_tail(orbital_size, beta);     
     Eigen::MatrixXcd greens_function(orbital_size, orbital_size);
     Eigen::MatrixXcd pot_energy =
          Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     Eigen::MatrixXcd inverse_gf(orbital_size, orbital_size);
     std::complex<double> mu = std::complex<double>(chemical_potential);
     Eigen::VectorXcd to_add(orbital_size);
     for (int k_index = k_min; k_index < k_max; k_index++) {
	  double l_weight = lattice_bs_->get_weight(k_index);
	  if (abs(l_weight) < 1e-6) {
	       k_resolved_xcurrent_matrices.push_back(
		    Eigen::MatrixXcd::Zero(orbital_size, orbital_size));
	       k_resolved_ycurrent_matrices.push_back(
		    Eigen::MatrixXcd::Zero(orbital_size, orbital_size));
	       k_resolved_occupation_matrices.push_back(
		    Eigen::MatrixXcd::Zero(orbital_size, orbital_size));
	       if (world_rank_ == 0) {
		    cout << "skipping k point" << endl;
	       }
	       continue;
	  }
          partial_potential_energy = 0.0;
	  if (exact_tail == true) {
	       compute_analytical_tail(chemical_potential, k_index, beta);
	  } else {
	       compute_tail_contribution(k_index, orbital_size,
					 chemical_potential, beta);
	  }
          // At this point, the k resolved occupation matrix last eleent
          // is only the tail part -> we exploit this fact.
          partial_potential_energy += real(((k_resolved_occupation_matrices.back() +
                                             from_zero_plus_to_zero_minus) *
                                            sigma_->get_sigma_0()).diagonal().sum()) / 2.0;          
	  // Now we can calculate the exact partial sum over
	  // frequencies of the Green's function,
	  // by explicit evaluation, trace and summation.
	  for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	       to_add = Eigen::VectorXcd::Constant
		    (orbital_size, mu + sigma_->get_matsubara_frequency(freq_index));		  
	       inverse_gf = - lattice_bs_->dispersion_[k_index] - sigma_->values_[freq_index];
	       inverse_gf.diagonal() += to_add;
	       greens_function = inverse_gf.inverse();
               partial_potential_energy +=
                    real((greens_function * sigma_->values_[freq_index]).diagonal().sum()) / beta;
	       // the hermitian conjugate is added below
	       // to account for negative frequencies,
	       // since our target is the sum over Matsubara freqs.
	       k_resolved_occupation_matrices.back() +=
		    (greens_function +
		     greens_function.transpose().conjugate()) / beta;
	       if (compute_derivative) {
		    dn_dmu -= 2.0 * l_weight *
			 real((greens_function * greens_function).diagonal().sum()) / beta;
	       }
	  }
	  k_resolved_occupation_matrices.back() += from_zero_plus_to_zero_minus;
	  if (compute_spin_current == true) {
	       Eigen::VectorXd cur_k_point = lattice_bs_->get_k_point(k_index);
	       double x_phase_factor = 2.0 * cur_k_point(0) * M_PI;
	       double y_phase_factor = 2.0 * cur_k_point(1) * M_PI;;
	       k_resolved_xcurrent_matrices.push_back(
		    l_weight * exp(std::complex<double>(0.0, x_phase_factor)) *
		    k_resolved_occupation_matrices.back());
	       k_resolved_ycurrent_matrices.push_back(
		    l_weight * exp(std::complex<double>(0.0, y_phase_factor)) *
		    k_resolved_occupation_matrices.back());
	       spin_current_matrix[0] += k_resolved_xcurrent_matrices.back();
	       spin_current_matrix[1] += k_resolved_ycurrent_matrices.back();
	  }
	  occupation_matrix += l_weight * k_resolved_occupation_matrices.back();
	  partial_kinetic_energy += l_weight * real((lattice_bs_->dispersion_[k_index] *
						     k_resolved_occupation_matrices.back()).diagonal().sum());
          cum_partial_potential_energy += l_weight * partial_potential_energy;
     }
     density = real(occupation_matrix.diagonal().sum());
     MPI_Allreduce(&density, &world_density, 1,
		   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(&partial_kinetic_energy, &kinetic_energy, 1,
		   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);     
     MPI_Allreduce(&cum_partial_potential_energy, &potential_energy, 1,
		   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);     
     if (compute_derivative) {     
	  MPI_Allreduce(&dn_dmu, &world_dn_dmu, 1,
			MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     }
     MPI_Allreduce(occupation_matrix.data(),
		   world_occupation_matrix.data(),
		   occupation_matrix.size(),
		   MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
     for (int i = 0; i < spin_current_matrix.size(); ++i) {
	  MPI_Allreduce(spin_current_matrix[i].data(),
			world_spin_current_matrix[i].data(),
			spin_current_matrix[i].size(),
			MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
     }
     return tuple<double, double>(world_density, world_dn_dmu);
}

void DMFTModel::compute_order_parameter() {
     order_parameters.clear();
     int n_sites = sigma_->get_n_sites();
     int per_site_orbital_size = sigma_->get_per_site_orbital_size();
     // factor 2 for spin
     Eigen::MatrixXcd ordered_view =
	  Eigen::MatrixXcd::Zero(per_site_orbital_size / 2,
				 per_site_orbital_size / 2);
     Eigen::MatrixXcd local_moment_view =
	  Eigen::MatrixXcd::Zero(per_site_orbital_size,
				 per_site_orbital_size);
     std::complex<double> cim(0.0, 1.0);
     for (int i = 0; i < n_sites; i++) {
	  for (int j = 0; j < n_sites; j++) {
	       // Get partial view on the occupation matrix
	       Eigen::MatrixXcd partial_view = world_occupation_matrix.block(
		    i * per_site_orbital_size, j * per_site_orbital_size,
		    per_site_orbital_size, per_site_orbital_size);
	       if (per_site_orbital_size != 4) {
		    throw runtime_error("Only 2-spinful-orbital framework supported "
					"at the moment in function" + std::string(__FUNCTION__));
	       } else {
		    // reorder
		    // a^dagger_up b_up
		    ordered_view(0, 0) = partial_view(3, 0);
		    // a^dagger_up b_down
		    ordered_view(0, 1) = partial_view(1, 0);
		    // a^dagger_down b_up
		    ordered_view(1, 0) = partial_view(3, 2);
		    // a^dagger_down b_down
		    ordered_view(1, 1) = partial_view(1, 2);
		    // reorder
		    // a^dagger_up a_up
		    local_moment_view(0, 0) = partial_view(0, 0);
		    // a^dagger_down a_down
		    local_moment_view(1, 1) = partial_view(2, 2);
		    // a^dagger_up a_down
		    local_moment_view(0, 1) = partial_view(0, 2);
		    // a^dagger_down a_up
		    local_moment_view(1, 0) = partial_view(2, 0);
		    // b^dagger_up b_up		    
		    local_moment_view(2, 2) = partial_view(3, 3);
		    // b^dagger_down b_down
		    local_moment_view(3, 3) = partial_view(1, 1);
		    // b^dagger_up b_down		    
		    local_moment_view(2, 3) = partial_view(3, 1);
		    // b^dagger_down b_up
		    local_moment_view(3, 2) = partial_view(1, 3);
	       }
	       // Multiply by the tau matrices in order to get the
	       // 4 components of the order parameter
	       // Cf formula (11) from the review
	       // "Excitonic condensation in systems
	       // of strongly correlated electrons"
	       // by J. Kunes
	       Eigen::VectorXcd local_order_parameter =
		    Eigen::VectorXcd::Zero(phi_dimension);
	       // singlet component
	       local_order_parameter(0) =
		    0.5 * (ordered_view.cwiseProduct(
				(Eigen::MatrixXcd(2,2)
				 << 1.0, 0.0, 0.0, 1.0).finished())).sum();
	       // triplet x-component
	       local_order_parameter(1) =
		    0.5 * (ordered_view.cwiseProduct(
				(Eigen::MatrixXcd(2,2)
				 << 0.0, 1.0, 1.0, 0.0).finished())).sum();
	       // triplet y-component
	       local_order_parameter(2) =
		    0.5 * (ordered_view.cwiseProduct(
				(Eigen::MatrixXcd(2,2)
				 << 0.0, -cim, cim, 0.0).finished())).sum();
	       // triplet z-component
	       local_order_parameter(3) =
		    0.5 * (ordered_view.cwiseProduct(
				(Eigen::MatrixXcd(2,2)
				 << 1.0, 0.0, 0.0, -1.0).finished())).sum();
	       // S_x
	       local_order_parameter(4) =
		    0.5 * (local_moment_view.block(0, 0, per_site_orbital_size / 2, per_site_orbital_size / 2)
			   .cwiseProduct((Eigen::MatrixXcd(2,2)
					  << 0.0, 1.0, 1.0, 0.0).finished())).sum() +
		    0.5 * (local_moment_view.block(per_site_orbital_size / 2,per_site_orbital_size / 2,
						   per_site_orbital_size / 2, per_site_orbital_size / 2)
			   .cwiseProduct((Eigen::MatrixXcd(2,2)
					  << 0.0, 1.0, 1.0, 0.0).finished())).sum();
	       // S_y
	       local_order_parameter(5) =
		    0.5 * (local_moment_view.block(0, 0, per_site_orbital_size / 2, per_site_orbital_size / 2)
			   .cwiseProduct((Eigen::MatrixXcd(2,2)
					  << 0.0, -cim, cim, 0.0).finished())).sum() +
		    0.5 * (local_moment_view.block(per_site_orbital_size / 2,per_site_orbital_size / 2,
						   per_site_orbital_size / 2, per_site_orbital_size / 2)
			   .cwiseProduct((Eigen::MatrixXcd(2,2)
					  << 0.0, -cim, cim, 0.0).finished())).sum();
               // S_z
	       local_order_parameter(6) =
		    0.5 * (local_moment_view.block(0, 0, per_site_orbital_size / 2, per_site_orbital_size / 2)
			   .cwiseProduct((Eigen::MatrixXcd(2,2)
					  << 1.0, 0., 0., -1.0).finished())).sum() +
		    0.5 * (local_moment_view.block(per_site_orbital_size / 2,per_site_orbital_size / 2,
						   per_site_orbital_size / 2, per_site_orbital_size / 2)
			   .cwiseProduct((Eigen::MatrixXcd(2,2)
					  << 1.0, 0., 0., -1.0).finished())).sum();
	       // S^2
	       local_order_parameter(7) = std::sqrt(
		    std::pow(local_order_parameter(4), 2) +
		    std::pow(local_order_parameter(5), 2) +
		    std::pow(local_order_parameter(6), 2));
	       order_parameters.push_back(local_order_parameter);
	  }
     }
}

void DMFTModel::get_spin_current() {
     spin_current_components.clear();
     int n_sites = sigma_->get_n_sites();
     int per_site_orbital_size = sigma_->get_per_site_orbital_size();
     if (per_site_orbital_size != 4) {
	  throw runtime_error("Only 2-spinful-orbital framework supported "
			      "in function" + std::string(__FUNCTION__));
     }
     Eigen::MatrixXcd aa_component = Eigen::MatrixXcd::Zero(n_sites * 2, n_sites * 2);
     Eigen::MatrixXcd bb_component = Eigen::MatrixXcd::Zero(n_sites * 2, n_sites * 2);
     Eigen::MatrixXcd ab_component = Eigen::MatrixXcd::Zero(n_sites * 2, n_sites * 2);
     Eigen::MatrixXcd ba_component = Eigen::MatrixXcd::Zero(n_sites * 2, n_sites * 2);
     Eigen::MatrixXcd summed_component = Eigen::MatrixXcd::Zero(n_sites * 2, n_sites * 2);
     Eigen::MatrixXcd V_matrix = lattice_bs_->get_V_matrix();
     for (int line_idx = 0; line_idx < per_site_orbital_size; line_idx++) {
	  V_matrix(line_idx, line_idx) = 2.0 * V_matrix(line_idx, line_idx);
     }
     for (int direction_index = 0; direction_index < 2; direction_index++) {
	  for (int i = 0; i < n_sites; i++) {
	       for (int j = 0; j < n_sites; j++) {
                    spin_current_components.push_back(Eigen::VectorXcd::Zero(current_dimension));
		    // Get partial view on the occupation matrix
		    Eigen::MatrixXcd partial_view = world_spin_current_matrix[direction_index].block(
			 i * per_site_orbital_size, j * per_site_orbital_size,
			 per_site_orbital_size, per_site_orbital_size);
		    // reorder
		    aa_component.block(i * 2, j * 2, 2, 2)(0, 0) = partial_view(0, 0);
		    aa_component.block(i * 2, j * 2, 2, 2)(0, 1) = partial_view(0, 2);
		    aa_component.block(i * 2, j * 2, 2, 2)(1, 0) = partial_view(2, 0);
		    aa_component.block(i * 2, j * 2, 2, 2)(1, 1) = partial_view(2, 2);
		    bb_component.block(i * 2, j * 2, 2, 2)(0, 0) = partial_view(3, 3);
		    bb_component.block(i * 2, j * 2, 2, 2)(0, 1) = partial_view(3, 1);
		    bb_component.block(i * 2, j * 2, 2, 2)(1, 0) = partial_view(1, 3);
		    bb_component.block(i * 2, j * 2, 2, 2)(1, 1) = partial_view(1, 1);
		    ab_component.block(i * 2, j * 2, 2, 2)(0, 0) = partial_view(0, 3);
		    ab_component.block(i * 2, j * 2, 2, 2)(0, 1) = partial_view(0, 1);
		    ab_component.block(i * 2, j * 2, 2, 2)(1, 0) = partial_view(2, 3);
		    ab_component.block(i * 2, j * 2, 2, 2)(1, 1) = partial_view(2, 1);
		    ba_component.block(i * 2, j * 2, 2, 2)(0, 0) = partial_view(3, 0);
		    ba_component.block(i * 2, j * 2, 2, 2)(0, 1) = partial_view(3, 2);
		    ba_component.block(i * 2, j * 2, 2, 2)(1, 0) = partial_view(1, 0);
		    ba_component.block(i * 2, j * 2, 2, 2)(1, 1) = partial_view(1, 2);
		    summed_component.block(i * 2, j * 2, 2, 2) =
			 V_matrix(0, 0) * aa_component + V_matrix(1, 1) * bb_component +
			 0.0 * V_matrix(0, 3) * ab_component + 0.0 * V_matrix(3, 0) * ba_component;
		    for (int spin_component = 0; spin_component < current_dimension; spin_component++) {
			 spin_current_components.back()(spin_component) =
			      summed_component.block(i * 2, j * 2, 2, 2).
			      cwiseProduct((get_pauli_matrix(spin_component))).sum();			      
		    }
	       }
	  }
     }
}

void DMFTModel::display_occupation_matrix() {
     if (world_rank_ == 0) {
	  cout << endl;
	  cout << "OCCUPATION MATRIX: " << endl;
	  cout << "real: " << endl;
	  cout << world_occupation_matrix.real() << endl << endl;
	  cout << "imaginary: " << endl;
	  cout << world_occupation_matrix.imag() << endl << endl;
	  cout << "ORDER PARAMETER:" << endl;
	  cout << "SITE          ";
	  int n_sites = sigma_->get_n_sites();
	  for (int i = 0; i < n_sites; i++) {
	       for (int j = 0; j < n_sites; j++) {
		    cout << i << " " << j << "         ---        ";
	       }
	  }
	  cout << endl;
	  // Reduce output print precision for order parameter
	  auto old_precision = cout.precision(phi_output_precision);
	  for (int coord_index = 0; coord_index < phi_dimension; coord_index++) {
	       cout << "phi_" << coord_index << "   ";
	       for (std::vector<Eigen::VectorXcd>::const_iterator it = order_parameters.begin();
		    it != order_parameters.end(); ++it) {
		    cout << (*it)(coord_index) << "  ";
	       }
	       cout << endl;
	  }
	  cout.precision(old_precision);
     }
     cout << endl;
}

void DMFTModel::display_spin_current() {
     if (world_rank_ == 0) {
	  cout << "SPIN CURRENT:" << endl;
	  cout << "SITE          ";
	  int n_sites = sigma_->get_n_sites();
	  for (int i = 0; i < n_sites; i++) {
	       for (int j = 0; j < n_sites; j++) {
		    cout << i << " " << j << "         ---        ";
	       }
	  }
	  cout << endl;
	  auto old_precision = cout.precision(current_output_precision);
	  for (int direction_index = 0; direction_index < 2; direction_index++) {
	       cout << "direction " << direction_index << "   " << endl << endl;
	       for (int coord_index = 0; coord_index < current_dimension; coord_index++) {
		    cout << "S" << coord_index << "   ";
                    for (int i = 0; i < n_sites; i++) {
                         for (int j = 0; j < n_sites; j++) {
                              cout << spin_current_components[direction_index * n_sites * n_sites + i * n_sites + j][coord_index]
                                   << "   ";
                         }
                    }
                    std::cout << std::endl;
	       }
               std::cout << std::endl;
	  }
	  cout.precision(old_precision);
     }
     cout << endl;
}

double DMFTModel::get_kinetic_energy() {
     return kinetic_energy;
}

double DMFTModel::get_potential_energy() {
     return potential_energy;
}

void DMFTModel::dump_k_resolved_occupation_matrices() {
     if (world_rank_ == 0) {
	  boost::timer::auto_cpu_timer mu_calc;
	  std::ofstream out(k_resolved_occupation_dump_name);
	  out << fixed << setprecision(output_precision);	
	  size_t k_max = lattice_bs_->get_real_n_points();
	  size_t orbital_size = lattice_bs_->get_orbital_size();
	  for (size_t k_index = 0; k_index < k_max; k_index++) {
	       // We use k_index + 1 to keep consistency with the Fortran output
	       out << k_index + 1 << endl;
	       for (size_t orb1 = 0; orb1 < orbital_size; orb1++) {
		    for (size_t orb2 = 0; orb2 < orbital_size; orb2++) {	       
			 out << world_k_resolved_occupation_matrices[k_index](orb1, orb2) << endl;
		    }
	       }
	  }
	  out.close();
	  cout << "Dump of spin texture done -- timing: ";
     }
}

const size_t DMFTModel::max_iter_for_bisec = 30;
const size_t DMFTModel::max_iter_for_bounds = 30;
const size_t DMFTModel::max_iter_for_newton = 10;
const double DMFTModel::e_max = 50.0;
const std::string DMFTModel::k_resolved_occupation_dump_name = "c_nk.dmft";
const std::size_t DMFTModel::output_precision = 13;
const std::size_t DMFTModel::phi_output_precision = 4;
const std::size_t DMFTModel::current_output_precision = 7;
const std::size_t DMFTModel::phi_dimension = 8;
const std::size_t DMFTModel::current_dimension = 4;
