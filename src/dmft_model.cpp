#include "dmft_model.hpp"
#include <Eigen/LU>
#include<math.h>
#include <boost/timer/timer.hpp>
#include <algorithm>

using namespace std;

DMFTModel::DMFTModel(boost::shared_ptr<Bandstructure> const &lattice_bs,
		     boost::shared_ptr<Selfenergy> const &sigma,
		     const alps::params& parms, int world_rank)
     : lattice_bs_(lattice_bs), sigma_(sigma), world_rank_(world_rank) {
     size_t N_max = sigma_->get_n_matsubara_freqs();
     double omega_max = abs(sigma_->get_matsubara_frequency(N_max - 1));
     string ref_exact ("exact");
     string tail_style = parms["TAIL_STYLE"];
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
     for (int k_index = 0; k_index < n_points_per_proc * world_size; k_index++) {
	  world_k_resolved_occupation_matrices[k_index].resize(
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

Eigen::MatrixXcd DMFTModel::get_gf_derivative(int derivationindex, int k_index,
					      int freq_index, double chemical_potential,
					      bool negative_freq) {
     int orbital_size = lattice_bs_->get_orbital_size();
     Eigen::MatrixXcd output = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     if (derivationindex == 1) {
	  output = get_dx_greens_function(k_index, freq_index,
					  chemical_potential, negative_freq);
     } else if (derivationindex == 2) {
	  output = get_dy_greens_function(k_index, freq_index,
					  chemical_potential, negative_freq);
     } else if (derivationindex == 0) {
	  output = get_domega_greens_function(k_index, freq_index,
					      chemical_potential, negative_freq);
     } else {
	  throw std::runtime_error("Wrong index for GF derivation");
     }
     return output;
}

Eigen::MatrixXcd DMFTModel::get_domega_greens_function(
     int k_index, int freq_index, double chemical_potential, bool negative_freq) {
     int orbital_size = lattice_bs_->get_orbital_size();
     int N_max = sigma_->get_n_matsubara_freqs();
     Eigen::MatrixXcd output = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     if ((freq_index < N_max - 1) && (!negative_freq)) {
	  Eigen::MatrixXcd inverse_gf(orbital_size, orbital_size);
	  Eigen::MatrixXcd greens_function(orbital_size, orbital_size);
	  Eigen::MatrixXcd do_greens_function(orbital_size, orbital_size);
	  Eigen::MatrixXcd do_inverse_gf(orbital_size, orbital_size);
	  std::complex<double> mu = std::complex<double>(chemical_potential);
	  Eigen::VectorXcd to_add(orbital_size);
	  Eigen::VectorXcd do_to_add(orbital_size);
	  to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu + sigma_->get_matsubara_frequency(freq_index));
	  do_to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu + sigma_->get_matsubara_frequency(freq_index + 1));
	  inverse_gf = -lattice_bs_->dispersion_[k_index] - sigma_->values_[freq_index];
	  do_inverse_gf = -lattice_bs_->dispersion_[k_index] - sigma_->values_[freq_index + 1];
	  inverse_gf.diagonal() += to_add;
	  do_inverse_gf.diagonal() += do_to_add;
	  greens_function = inverse_gf.inverse();
	  do_greens_function = do_inverse_gf.inverse();
	  output = inverse_gf * (do_inverse_gf - greens_function) / (
	       std::abs(sigma_->get_matsubara_frequency(1) -
			sigma_->get_matsubara_frequency(0)));
     } else if ((freq_index < N_max - 1) && (negative_freq)) {
	  Eigen::MatrixXcd inverse_gf(orbital_size, orbital_size);
	  Eigen::MatrixXcd greens_function(orbital_size, orbital_size);
	  Eigen::MatrixXcd do_greens_function(orbital_size, orbital_size);
	  Eigen::MatrixXcd do_inverse_gf(orbital_size, orbital_size);
	  std::complex<double> mu = std::complex<double>(chemical_potential);
	  Eigen::VectorXcd to_add(orbital_size);
	  Eigen::VectorXcd do_to_add(orbital_size);
	  to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu - sigma_->get_matsubara_frequency(N_max - 1 - freq_index));
	  do_to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu - sigma_->get_matsubara_frequency(N_max - 2 - freq_index));
	  inverse_gf = -lattice_bs_->dispersion_[k_index] - sigma_->neg_values_[freq_index];
	  do_inverse_gf = -lattice_bs_->dispersion_[k_index] - sigma_->neg_values_[freq_index + 1];
	  inverse_gf.diagonal() += to_add;
	  do_inverse_gf.diagonal() += do_to_add;
	  greens_function = inverse_gf.inverse();
	  do_greens_function = do_inverse_gf.inverse();
	  output = inverse_gf * (do_inverse_gf - greens_function) / (
	       std::abs(sigma_->get_matsubara_frequency(1) -
			sigma_->get_matsubara_frequency(0)));
     } else if ((freq_index == (N_max - 1)) && (negative_freq)) {
	  Eigen::MatrixXcd inverse_gf(orbital_size, orbital_size);
	  Eigen::MatrixXcd greens_function(orbital_size, orbital_size);
	  Eigen::MatrixXcd do_greens_function(orbital_size, orbital_size);
	  Eigen::MatrixXcd do_inverse_gf(orbital_size, orbital_size);
	  std::complex<double> mu = std::complex<double>(chemical_potential);
	  Eigen::VectorXcd to_add(orbital_size);
	  Eigen::VectorXcd do_to_add(orbital_size);
	  to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu - sigma_->get_matsubara_frequency(0));
	  do_to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu + sigma_->get_matsubara_frequency(0));
	  inverse_gf = -lattice_bs_->dispersion_[k_index] - sigma_->neg_values_[freq_index];
	  do_inverse_gf = -lattice_bs_->dispersion_[k_index] - sigma_->values_[0];
	  inverse_gf.diagonal() += to_add;
	  do_inverse_gf.diagonal() += do_to_add;
	  greens_function = inverse_gf.inverse();
	  do_greens_function = do_inverse_gf.inverse();
	  output = inverse_gf * (do_inverse_gf - greens_function) / (
	       std::abs(sigma_->get_matsubara_frequency(1) -
			sigma_->get_matsubara_frequency(0)));	  
     }
     return output / (24.0 * std::pow(M_PI, 2));
}

Eigen::MatrixXcd DMFTModel::get_dx_greens_function(
     int k_index, int freq_index, double chemical_potential, bool negative_freq) {
     int N_max = sigma_->get_n_matsubara_freqs();
     int orbital_size = lattice_bs_->get_orbital_size();
     Eigen::MatrixXcd output = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     Eigen::MatrixXcd inverse_gf(orbital_size, orbital_size);
     Eigen::MatrixXcd greens_function(orbital_size, orbital_size);
     Eigen::MatrixXcd dx_greens_function(orbital_size, orbital_size);
     Eigen::MatrixXcd dx_inverse_gf(orbital_size, orbital_size);
     std::complex<double> mu = std::complex<double>(chemical_potential);
     Eigen::VectorXcd to_add(orbital_size);
     if (negative_freq) {
	  to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu - sigma_->get_matsubara_frequency(N_max - 1 - freq_index));
          inverse_gf = -lattice_bs_->dispersion_[k_index] -
	       sigma_->neg_values_[N_max - 1 - freq_index];
	  dx_inverse_gf = -lattice_bs_->dispersion_dx_[k_index] -
	       sigma_->neg_values_[N_max - 1 - freq_index];
     } else {
	  to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu + sigma_->get_matsubara_frequency(freq_index));
          inverse_gf = -lattice_bs_->dispersion_[k_index] -
	       sigma_->values_[freq_index];
	  dx_inverse_gf = -lattice_bs_->dispersion_dx_[k_index] -
	       sigma_->values_[freq_index];
     }
     inverse_gf.diagonal() += to_add;
     dx_inverse_gf.diagonal() += to_add;
     greens_function = inverse_gf.inverse();
     dx_greens_function = dx_inverse_gf.inverse();
     output = inverse_gf * (dx_inverse_gf - greens_function) / lattice_bs_->get_x_dim();
     return output;
}

Eigen::MatrixXcd DMFTModel::get_dy_greens_function(
     int k_index, int freq_index, double chemical_potential, bool negative_freq) {
     int N_max = sigma_->get_n_matsubara_freqs();
     int orbital_size = lattice_bs_->get_orbital_size();
     Eigen::MatrixXcd output = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     Eigen::MatrixXcd inverse_gf(orbital_size, orbital_size);
     Eigen::MatrixXcd greens_function(orbital_size, orbital_size);
     Eigen::MatrixXcd dy_greens_function(orbital_size, orbital_size);
     Eigen::MatrixXcd dy_inverse_gf(orbital_size, orbital_size);
     std::complex<double> mu = std::complex<double>(chemical_potential);
     Eigen::VectorXcd to_add(orbital_size);
     if (negative_freq) {
	  to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu - sigma_->get_matsubara_frequency(N_max - 1 - freq_index));
          inverse_gf = -lattice_bs_->dispersion_[k_index] -
	       sigma_->neg_values_[N_max - 1 - freq_index];
	  dy_inverse_gf = -lattice_bs_->dispersion_dy_[k_index] -
	       sigma_->neg_values_[N_max - 1 - freq_index];
     } else {
	  to_add = Eigen::VectorXcd::Constant
	       (orbital_size, mu + sigma_->get_matsubara_frequency(freq_index));
          inverse_gf = -lattice_bs_->dispersion_[k_index] -
	       sigma_->values_[freq_index];
	  dy_inverse_gf = -lattice_bs_->dispersion_dy_[k_index] -
	       sigma_->values_[freq_index];
     }
     inverse_gf.diagonal() += to_add;
     dy_inverse_gf.diagonal() += to_add;
     greens_function = inverse_gf.inverse();
     dy_greens_function = dy_inverse_gf.inverse();
     output = inverse_gf * (dy_inverse_gf - greens_function) / lattice_bs_->get_y_dim();
     return output;
}

std::complex<double> DMFTModel::get_chern_number(double chemical_potential) {
     //vector<Eigen::MatrixXcd> k_resolved_occupation_matrices;
     int index_1 = 0;
     int index_2 = 1;
     int index_3 = 2;
     int myints[] = {0, 1, 2};
     std::sort (myints, myints+3);
     std::vector<double> parities {1.0, -1.0, -1.0, 1.0, 1.0, -1.0};
     int k_min(0);
     int k_max = lattice_bs_->get_lattice_size();
     int N_max = sigma_->get_n_matsubara_freqs();
     int orbital_size = lattice_bs_->get_orbital_size();	  
     std::complex<double> output(0.0);
     int cur_idx = 0;
     do {
	  index_1 = myints[0];
	  index_2 = myints[1];
	  index_3 = myints[2];
	  Eigen::MatrixXcd current_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
	  Eigen::MatrixXcd temp_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
	  Eigen::MatrixXcd neg_temp_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
	  for (int k_index = k_min; k_index < k_max; k_index++) {
	       double l_weight = lattice_bs_->get_weight(k_index);
	       if (abs(l_weight) < 1e-6) {
		    if (world_rank_ == 0) {
			 cout << "skipping k point" << endl;
		    }
		    continue;
	       }
	       for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
		    temp_matrix =
			 get_gf_derivative(index_1, k_index, freq_index, chemical_potential, false) *
			 get_gf_derivative(index_2, k_index, freq_index, chemical_potential, false) *
			 get_gf_derivative(index_3, k_index, freq_index, chemical_potential, false);
		    current_matrix += temp_matrix * l_weight;
		    neg_temp_matrix =
			 get_gf_derivative(index_1, k_index, freq_index, chemical_potential, true) *
			 get_gf_derivative(index_2, k_index, freq_index, chemical_potential, true) *
			 get_gf_derivative(index_3, k_index, freq_index, chemical_potential, true);
		    current_matrix += neg_temp_matrix * l_weight;
	       }
	  }
	  world_chern_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
	  MPI_Allreduce(current_matrix.data(),
			world_chern_matrix.data(),
			current_matrix.size(),
			MPI::DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
	  if (!world_rank_) {
	       std::cout << "permutation: " << index_1 << " "
			 << index_2 << " " << index_3 << std::endl;
	       std::cout << "Chern " << world_chern_matrix.diagonal().sum()
			 << std::endl;
	  }
	  output += parities[cur_idx] * world_chern_matrix.diagonal().sum();
	  cur_idx += 1;
     } while (std::next_permutation(myints, myints+3));
     return output;
}

tuple<bool, double, double> DMFTModel::get_mu_from_density_bisec(double initial_mu,
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
     bool success = false;
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
	  mu_increment = -abs(mu_increment);
     } else {
	  mu_increment = abs(mu_increment);
     }
     success = check_density_success(cur_density);
     MPI_Bcast(&success, 1, MPI::BOOL, 0, MPI_COMM_WORLD);
     // current value of mu is not satisfactory, we have to bracket.
     if (success == false) {
	  while (((cur_density - target_density) * (old_density - target_density) > 0.0) &&
		 (iteration_idx < max_iter_for_bounds) &&
		 (success == false)) {
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
	       MPI_Bcast(&success, 1, MPI::BOOL, 0, MPI_COMM_WORLD);
	  }
	  // Exit on bracketing failure
	  if (world_rank_ == 0) {
	       if ((iteration_idx >= max_iter_for_bounds) &&
		   (success == false)) {
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
	  while((success == false) &&
		(iteration_idx < max_iter_for_bisec)) {
	       cur_mu = 0.5 * (mu_min + mu_max);
	       tie(cur_density, cur_derivative) = get_particle_density(cur_mu, false);
	       if (world_rank_ == 0) {
		    cout << "Starting bisection " << mu_min ;
		    cout << "new mu: " << cur_mu << endl;
		    cout << "new density " << cur_density << endl;
		    cout << "cur iteration for bisec: " << iteration_idx << endl;
	       }
	       if (cur_density > target_density) {
		    mu_max = cur_mu;
	       } else {
		    mu_min = cur_mu;
	       }
	       success = check_density_success(cur_density);
	       MPI_Bcast(&success, 1, MPI::BOOL, 0, MPI_COMM_WORLD);
	       iteration_idx++;
	  }
     }
     // Exit on bisection failure.
     if ((success == false) && (iteration_idx >= max_iter_for_bisec)) {
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
     return tuple<bool, double, double>(success, cur_mu, cur_density);
}

tuple<bool, double, double, double> DMFTModel::get_mu_from_density(double initial_mu) {
     boost::timer::auto_cpu_timer mu_calc;
     if (world_rank_ == 0) {
	  cout << "**********************************************" << endl;
	  cout << "** CALCULATION OF CHEM. POTENTIAL : NEWTON ***" << endl;
	  cout << "**********************************************" << endl;
	  cout << "n_tot tolerance " << n_tolerance << endl;
	  cout << "target density " << target_density << endl;
     }
     bool success = false;
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
	  MPI_Bcast(&success, 1, MPI::BOOL, 0, MPI_COMM_WORLD);
	  if (success == true) {
	       break;
	  } else {
	       delta_mu = (target_density - cur_density) / cur_derivative;
	       cur_mu += delta_mu;
	  }
     }
     if (success == true) {
	  // Success - scatter/gather the k-resolved occupation matrices.
	  scatter_occ_matrices();     
     }
     if (world_rank_ == 0) {     
	  cout << " mu_calc + energy - timing: ";
     }
     return tuple<bool, double, double, double>(success, cur_mu, cur_density, delta_mu);
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
			 MPI::DOUBLE_COMPLEX, proc_index, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	       }
	  } else {
	       MPI_Send(k_resolved_occupation_matrices[k_index].data(),
			k_resolved_occupation_matrices[k_index].size(),
			MPI::DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);	       
	  }
	  if (world_rank_ == 0) {
	       world_k_resolved_occupation_matrices[k_index] =
		    k_resolved_occupation_matrices[k_index];
	  }
     }
}

bool DMFTModel::check_density_success(double cur_density) {
     bool output = false;
     if (world_rank_ == 0) {
	  output =
	       (abs(target_density - cur_density) < n_tolerance) ?
	       true : false;
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

void DMFTModel::reset_occupation_matrices(size_t orbital_size)  {
     // Reset occupation matrix
     occupation_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     world_occupation_matrix = Eigen::MatrixXcd::Zero(orbital_size, orbital_size);
     k_resolved_occupation_matrices.clear();
}

std::vector<Eigen::MatrixXcd> DMFTModel::get_full_greens_functions(double chemical_potential) {
	std::vector<Eigen::MatrixXcd> output;
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
     kinetic_energy = 0.0;
     reset_occupation_matrices(orbital_size);
     // Here we use a diagonal matrix, because the discontinuity at tau =0
     // only exists for Green's functions involving identical orbitals.
     // For the formula related to such discontinuity, check
     // Mahan 3.2.9, p140.
     Eigen::MatrixXcd from_zero_plus_to_zero_minus = Eigen::VectorXcd::Constant(
	  orbital_size, 1.0).asDiagonal();
     if (compute_derivative == true) dn_dmu = compute_derivative_tail(orbital_size, beta);     
     Eigen::MatrixXcd greens_function(orbital_size, orbital_size);
     Eigen::MatrixXcd inverse_gf(orbital_size, orbital_size);	
     std::complex<double> mu = std::complex<double>(chemical_potential);
     Eigen::VectorXcd to_add(orbital_size);
     for (int k_index = k_min; k_index < k_max; k_index++) {
	  double l_weight = lattice_bs_->get_weight(k_index);
	  if (abs(l_weight) < 1e-6) {
	       k_resolved_occupation_matrices.push_back(
		    Eigen::VectorXcd::Zero(orbital_size, orbital_size));
	       if (world_rank_ == 0) {
		    cout << "skipping k point" << endl;
	       }
	       continue;
	  }
	  if (exact_tail == true) {
	       compute_analytical_tail(chemical_potential, k_index, beta);
	  } else {
	       compute_tail_contribution(k_index, orbital_size,
					 chemical_potential, beta);
	  }
	  // Now we can calculate the exact partial sum over
	  // frequencies of the Green's function,
	  // by explicit evaluation, trace and summation.
	  for (size_t freq_index = 0; freq_index < N_max; freq_index++) {
	       to_add = Eigen::VectorXcd::Constant
		    (orbital_size, mu + sigma_->get_matsubara_frequency(freq_index));		  
	       inverse_gf = - lattice_bs_->dispersion_[k_index] - sigma_->values_[freq_index];
	       inverse_gf.diagonal() += to_add;
	       greens_function = inverse_gf.inverse();
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
	  occupation_matrix += l_weight * k_resolved_occupation_matrices.back();
	  partial_kinetic_energy += l_weight * real((lattice_bs_->dispersion_[k_index] *
						     k_resolved_occupation_matrices.back()).diagonal().sum());
     }
     density = real(occupation_matrix.diagonal().sum());
     MPI_Allreduce(&density, &world_density, 1,
		   MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(&partial_kinetic_energy, &kinetic_energy, 1,
		   MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);     
     if (compute_derivative) {     
	  MPI_Allreduce(&dn_dmu, &world_dn_dmu, 1,
			MPI::DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     }
     MPI_Allreduce(occupation_matrix.data(),
		   world_occupation_matrix.data(),
		   occupation_matrix.size(),
		   MPI::DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
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
     std::complex<double> cim(0.0, 1.0);
     for (int i = 0; i < n_sites; i++) {
	  for (int j = 0; j < n_sites; j++) {
	       // Get partial view on the occupation matrix
	       Eigen::MatrixXcd partial_view = world_occupation_matrix.block(
		    i * per_site_orbital_size, j * per_site_orbital_size,
		    per_site_orbital_size, per_site_orbital_size);
	       if (per_site_orbital_size != 4) {
		    throw runtime_error("Only 2-spinful-orbital framework supported "
					"at the moment in function"
					+ std::string(__FUNCTION__));
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
	       }
	       // Multiply by the tau matrices in order to get the
	       // 4 components of the order parameter
	       // Cf formula (11) from the review
	       // "Excitonic condensation in systems
	       // of strongly correlated electrons"
	       // by J. Kunes
	       Eigen::VectorXcd local_order_parameter =
		    Eigen::VectorXcd::Zero(4);
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
	       local_order_parameter(3) =
		    partial_view(0, 0) - partial_view(1, 1)
		    -partial_view(2, 2) + partial_view(3, 3);
	       order_parameters.push_back(local_order_parameter);
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
	  // Rreduce output print precision for order parameter
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

double DMFTModel::get_kinetic_energy() {
     return kinetic_energy;
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

const size_t DMFTModel::max_iter_for_bisec = 10;
const size_t DMFTModel::max_iter_for_bounds = 10;
const size_t DMFTModel::max_iter_for_newton = 10;
const double DMFTModel::e_max = 50.0;
const std::string DMFTModel::k_resolved_occupation_dump_name = "c_nk.dmft";
const std::size_t DMFTModel::output_precision = 13;
const std::size_t DMFTModel::phi_output_precision = 4;
const std::size_t DMFTModel::phi_dimension = 4;
