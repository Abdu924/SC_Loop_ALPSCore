#include "tail_manager.hpp"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;

// define Fermi function, equipped with a cutoff
double fermi_function(double beta, double lambda, double fermi_cutoff) {
     double output(0.0);
     if (beta * lambda < -fermi_cutoff) {
	  output = 1.0;
     } else if (beta * lambda < fermi_cutoff) {
	  output = 1.0 / (1.0 + exp(beta * lambda));
     }
     return output;
}

TailManager::TailManager(const Eigen::Ref<Eigen::MatrixXcd> sigma_0,
			 const Eigen::Ref<Eigen::MatrixXcd> sigma_1,
			 const Eigen::Ref<Eigen::VectorXcd> matsubara_frequencies,
			 int world_rank)
     :world_rank_(world_rank) {
     flavor_size = sigma_0.cols();
     full_size = 2 * flavor_size;
     sigma_skeleton = Eigen::MatrixXcd::Zero(full_size, full_size);
     sigma_skeleton.block(0, 0, flavor_size, flavor_size) = sigma_0;
     Eigen::MatrixXcd tmp = sigma_1;
     Eigen::MatrixXcd sqrt_sigma_1_bis = tmp.sqrt();
     matsubara_frequencies_ = matsubara_frequencies;
     n_max = matsubara_frequencies_.size();
     sigma_skeleton.block(0, flavor_size, flavor_size, flavor_size) = sqrt_sigma_1_bis;
     sigma_skeleton.block(flavor_size, 0, flavor_size, flavor_size) =
	  sqrt_sigma_1_bis.transpose().conjugate();
     // HERE Check if we indeed need the transpose conjugate
     sigma_0_ = sigma_0;
     sigma_1_ = sigma_1;
}

void TailManager::set_chemical_potential(double chemical_potential) {
     Eigen::MatrixXcd to_add =
	  Eigen::VectorXcd::Constant(flavor_size, chemical_potential).asDiagonal();
     mirror_hamiltonian = sigma_skeleton;
     mirror_hamiltonian.block(0, 0, flavor_size, flavor_size) -= to_add;
     mu_ = chemical_potential;
}

void TailManager::set_current_k(const Eigen::Ref<Eigen::MatrixXcd> k_hamiltonian) {
     mirror_hamiltonian.block(0, 0, flavor_size, flavor_size) += k_hamiltonian;
}

Eigen::MatrixXcd TailManager::get_analytical_contribution(double beta) {
     double output(0.0), temp;
     double lambda, f_value;
     Eigen::ComplexEigenSolver<Eigen::MatrixXcd> eigen_solver;
     Eigen::MatrixXcd v_matrix;
     Eigen::VectorXcd f_vector(full_size);
     eigen_solver.compute(mirror_hamiltonian);
     for (int lambda_idx = 0; lambda_idx < f_vector.size(); ++lambda_idx) {
	  // Restriction, but not real assumption here: Hermitian Hamiltonian,
	  // ie real eigenvalues. We could check and emergency stop
	  // in case this is not verified.
	  //HERE Lift the restrition - OK?
	  // Reason is: Hamiltonian is Hermitian,
	  // but Sigma_0 not...
	  lambda = real(eigen_solver.eigenvalues()(lambda_idx));
	  //cout << "imag" << imag(eigen_solver.eigenvalues()(lambda_idx));
	  f_value = fermi_function(beta, lambda, fermi_cutoff);// - 0.5;
	  // for each eigenvalue, compute
	  // sum_{k} U_ki U^*_kj * f(lambda_k)
	  // and subtract the contribution from the numerically
	  // handled part of the frequencies.
	  f_value -= get_squared_partial_sum(lambda) / beta;
	  f_vector(lambda_idx) = f_value;
     }
     v_matrix = eigen_solver.eigenvectors() *
	  f_vector.asDiagonal();
     Eigen::MatrixXcd from_zero_plus_to_zero_minus = Eigen::VectorXcd::Constant(
	  flavor_size, 1.0).asDiagonal();
     return (v_matrix * eigen_solver.eigenvectors().inverse()).block(0, 0, flavor_size, flavor_size)
	  - from_zero_plus_to_zero_minus;
}

// Compute sum_-Im^Im [1 / (i omega_n - lambda)], i.e. the
// contribution of the numerically handled frequencies to
// the exact value of the tail.
double TailManager::get_squared_partial_sum(double lambda) {
     double output(0.0);
     for (int freq_idx = 0; freq_idx < n_max; ++freq_idx) {
	  output += 1.0 / (pow(lambda, 2) -
			   real(pow(matsubara_frequencies_(freq_idx), 2)));
     }
     return -2.0 * lambda * output;
}

const double TailManager::fermi_cutoff = 22.0;
