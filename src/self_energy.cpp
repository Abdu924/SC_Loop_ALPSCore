#include <boost/multi_array.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>
#include "self_energy.hpp"

/*
 * The sample sigma used for debugging comes from 
 * /storage/brno6/home/geffroy/DMFT_data/2site/sdw/af
 * on Metacentrum
 */

using namespace std;

// define function to be applied coefficient-wise
double square_inverse(complex<double> omega) {
     return real(pow(1.0 / omega, 2));
}

// The initial version, able to read sigma from txt or hdf5,
// -- compliant with the output format of the mix procedure.
Selfenergy::Selfenergy(const alps::params &parms, int world_rank,
		       alps::hdf5::archive h5_archive, string h5_group_name,
		       bool verbose)
     :world_rank_(world_rank), is_alps3(false), is_analytic_tail(true) {
     basic_init(parms, verbose);
     read_input_sigma(parms, h5_archive, h5_group_name);
     if (verbose) {
	  display_asymptotics();
     }
     compute_order2_partial_sum();
}

// The overloaded version to be used for mix
// -- compliant with qmc generated format. Beware! convention for
// annihilation/creation operator in rows/colums in different in QMC and
// dmft :(
Selfenergy::Selfenergy(const alps::params &parms, int world_rank,
		       boost::shared_ptr<Chemicalpotential> chempot,
		       int ref_site_index, alps::hdf5::archive h5_archive, int input_type,
		       bool verbose)
     :world_rank_(world_rank), chempot_(chempot), input_type(input_type),
      is_alps3(false) {
     basic_init(parms, verbose);
     is_analytic_tail = static_cast<bool>(parms["mixing.analytic_sigma_tail"]);
     std::string symmetry_file;
     sanity_check(parms);
     if (parms.exists("SITE_SYMMETRY")) {
	  std::string fname = parms["SITE_SYMMETRY"];
	  symmetry_file = fname;
	  cout << "Reading QMC for one site only and Using symmetry as defined in " <<
	       symmetry_file << endl;
     }
     // Get the result of the qmc calculation,
     // executed for one site only
     // The whole construction is done for ref_site,
     // and the symmetry is applied last.
     // Deduce the result for the other sites by applying
     // the relevant symmetry.
     read_symmetry_definition(symmetry_file);
     read_qmc_sigma(ref_site_index, h5_archive);
     // Smooth the noisy tail
     feed_tail_params(ref_site_index, parms, h5_archive);
     compute_tail_coeffs(ref_site_index);
     log_sigma_tails(ref_site_index);
     compute_qmc_tail(ref_site_index);
     append_qmc_tail(ref_site_index, parms);
     symmetrize_sites(ref_site_index);
     // precompute some matsubara frequency sums for later use
     // in the Fourier transforms.
     compute_order2_partial_sum();
}

// Constructor used with the Alps3 QMC impurity solver CTHyb
// (matrix implementation)
Selfenergy::Selfenergy(const alps::params &parms, int world_rank,
		       boost::shared_ptr<Chemicalpotential> chempot,
		       int ref_site_index, alps::hdf5::archive h5_archive,
		       boost::shared_ptr<Greensfunction> greens_function)
     :world_rank_(world_rank), chempot_(chempot), input_type(1), is_alps3(true) {
     basic_init(parms);
     // defaults to true
     is_analytic_tail = static_cast<bool>(parms["mixing.analytic_sigma_tail"]);
     std::string symmetry_file;
     sanity_check(parms);
     if (parms.exists("SITE_SYMMETRY")) {
	  std::string fname = parms["SITE_SYMMETRY"];
	  symmetry_file = fname;
	  cout << "Reading QMC for one site only and using symmetry as defined in " <<
	       symmetry_file << endl;
     }
     // Get the result of the qmc calculation,
     // executed for one site only
     // The whole construction is done for ref_site,
     // and the symmetry is applied last.
     // Deduce the result for the other sites by applying
     // the relevant symmetry.
     read_symmetry_definition(symmetry_file);
     read_qmc_sigma(ref_site_index, greens_function);
     //feed_tail_params(ref_site_index, parms, h5_archive);
     compute_tail_coeffs(greens_function, chempot, ref_site_index);
     log_sigma_tails(ref_site_index);
     compute_qmc_tail(ref_site_index);
     // No need to append tails when data comes from Legendre
     // append_qmc_tail(ref_site_index, parms);
     symmetrize_sites(ref_site_index);
     // precompute some matsubara frequency sums for later use
     // in the Fourier transforms.
     compute_order2_partial_sum();
}

void Selfenergy::sanity_check(const alps::params &parms) {
     if ((!parms.exists("SITE_SYMMETRY")) && (n_sites > 1)) {
	  throw runtime_error("n_sites > 1 but SITE_SYMMETRY is not defined !");
     }
}

void Selfenergy::read_symmetry_definition(std::string symmetry_file) {
     //Read symmetry definition file
     // HERE This does not really work, insofar as we cannot
     // pick site 1 as a reference site and apply the symmetry:
     // site 1 itself would be impacted...
     std::ifstream infile(symmetry_file.c_str());
     if(!infile.good()) {
	  cout << "Could not find file " + symmetry_file +
	       ". Abandoning" << endl;
	  throw runtime_error("symmetry file " + symmetry_file + " absent!");	  
     }
     alps::hdf5::archive sym_archive(symmetry_file, alps::hdf5::archive::READ);
     for (int site_index = 0; site_index < n_sites; site_index++) {
	  std::vector<double> temp_data;
	  temp_data.resize(per_site_orbital_size * per_site_orbital_size);
	  std::stringstream symmetry_path;
	  symmetry_path << "site_" + 
	       boost::lexical_cast<std::string>(site_index) + "/" +
	       "symmetry_matrix";
	  sym_archive >> alps::make_pvp(symmetry_path.str(),
					temp_data);
	  int cur_index = 0;
	  for (int orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
	       for (int orb2 = 0; orb2 < per_site_orbital_size; orb2++, cur_index++) {
		    site_symmetry_matrix.block(site_index * per_site_orbital_size,
					       site_index * per_site_orbital_size,
					       per_site_orbital_size,
					       per_site_orbital_size)(orb1, orb2) =
			 temp_data[cur_index];
	       }
	  }	       
     }
}

void Selfenergy::symmetrize_matrix_elements(int ref_site_index) {
     Eigen::MatrixXcd temp_matrix =
	  Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     if (enforce_real) {
	  for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	       temp_matrix = values_[freq_index].block(ref_site_index * per_site_orbital_size,
						       ref_site_index * per_site_orbital_size,
						       per_site_orbital_size,
						       per_site_orbital_size);
	       temp_matrix.transposeInPlace();
	       values_[freq_index].block(ref_site_index * per_site_orbital_size,
					 ref_site_index * per_site_orbital_size,
					 per_site_orbital_size,
					 per_site_orbital_size) =
		    0.5 * (values_[freq_index].block(ref_site_index * per_site_orbital_size,
						     ref_site_index * per_site_orbital_size,
						     per_site_orbital_size,
						     per_site_orbital_size) +
			   temp_matrix);
	       temp_matrix = qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
							ref_site_index * per_site_orbital_size,
							per_site_orbital_size,
							per_site_orbital_size);
	       temp_matrix.transposeInPlace();
	       qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
					  ref_site_index * per_site_orbital_size,
					  per_site_orbital_size,
					  per_site_orbital_size) =
		    0.5 * (qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
						      ref_site_index * per_site_orbital_size,
						      per_site_orbital_size,
						      per_site_orbital_size) +
			   temp_matrix);
	  }
	  temp_matrix = Sigma_0_.block(ref_site_index * per_site_orbital_size,
				       ref_site_index * per_site_orbital_size,
				       per_site_orbital_size,
				       per_site_orbital_size);
	  temp_matrix.transposeInPlace();
	  Sigma_0_.block(ref_site_index * per_site_orbital_size,
			 ref_site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) =
	       0.5 * (Sigma_0_.block(ref_site_index * per_site_orbital_size,
				     ref_site_index * per_site_orbital_size,
				     per_site_orbital_size,
				     per_site_orbital_size) +
		      temp_matrix);
	  temp_matrix = a_dagger_b.block(ref_site_index * per_site_orbital_size,
					 ref_site_index * per_site_orbital_size,
					 per_site_orbital_size,
					 per_site_orbital_size);
	  temp_matrix.transposeInPlace();
	  a_dagger_b.block(ref_site_index * per_site_orbital_size,
			   ref_site_index * per_site_orbital_size,
			   per_site_orbital_size,
			   per_site_orbital_size) =
	       0.5 * (a_dagger_b.block(ref_site_index * per_site_orbital_size,
				       ref_site_index * per_site_orbital_size,
				       per_site_orbital_size,
				       per_site_orbital_size) +
		      temp_matrix);
     } else {
	  // use G_ij(i omega_n) = G^*_ji(-i omega_n)
	  for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	       temp_matrix = neg_values_[n_matsubara_freqs - 1 - freq_index].
		    block(ref_site_index * per_site_orbital_size,
			  ref_site_index * per_site_orbital_size,
			  per_site_orbital_size,
			  per_site_orbital_size);
	       temp_matrix.adjointInPlace();
	       values_[freq_index].block(ref_site_index * per_site_orbital_size,
					 ref_site_index * per_site_orbital_size,
					 per_site_orbital_size,
					 per_site_orbital_size) =
		    0.5 * (values_[freq_index].block(ref_site_index * per_site_orbital_size,
						     ref_site_index * per_site_orbital_size,
						     per_site_orbital_size,
						     per_site_orbital_size) +
			   temp_matrix);
	       // tails are Hermitian
	       temp_matrix = qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
							ref_site_index * per_site_orbital_size,
							per_site_orbital_size,
							per_site_orbital_size);
	       temp_matrix.transposeInPlace();
	       qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
					  ref_site_index * per_site_orbital_size,
					  per_site_orbital_size,
					  per_site_orbital_size) =
		    0.5 * (qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
						      ref_site_index * per_site_orbital_size,
						      per_site_orbital_size,
						      per_site_orbital_size) +
			   temp_matrix);
	  }
     }
}

void Selfenergy::symmetrize_sites(int ref_site_index) {
     for (int site_index = 0; site_index < n_sites; site_index++) {
	  if (site_index == ref_site_index) 	  {
	       continue;
	  } else {
	       a_dagger_b.block(site_index * per_site_orbital_size,
				site_index * per_site_orbital_size,
				per_site_orbital_size,
				per_site_orbital_size) =
		    a_dagger_b.block(ref_site_index * per_site_orbital_size,
				     ref_site_index * per_site_orbital_size,
				     per_site_orbital_size,
				     per_site_orbital_size);
	       density_density_correl.block(site_index * per_site_orbital_size,
					    site_index * per_site_orbital_size,
					    per_site_orbital_size,
					    per_site_orbital_size) =
		    density_density_correl.block(ref_site_index * per_site_orbital_size,
						 ref_site_index * per_site_orbital_size,
						 per_site_orbital_size,
						 per_site_orbital_size);
	       Sigma_0_.block(site_index * per_site_orbital_size,
			      site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size) =
		    Sigma_0_.block(ref_site_index * per_site_orbital_size,
				   ref_site_index * per_site_orbital_size,
				   per_site_orbital_size,
				   per_site_orbital_size);
	       Sigma_1_.block(site_index * per_site_orbital_size,
			      site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size) =
		    Sigma_1_.block(ref_site_index * per_site_orbital_size,
				   ref_site_index * per_site_orbital_size,
				   per_site_orbital_size,
				   per_site_orbital_size);
	       for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
		    values_[freq_index].block(site_index * per_site_orbital_size,
					      site_index * per_site_orbital_size,
					      per_site_orbital_size,
					      per_site_orbital_size) =
			 values_[freq_index].block(ref_site_index * per_site_orbital_size,
						   ref_site_index * per_site_orbital_size,
						   per_site_orbital_size,
						   per_site_orbital_size);
		    qmc_tail[freq_index].block(site_index * per_site_orbital_size,
					       site_index * per_site_orbital_size,
					       per_site_orbital_size,
					       per_site_orbital_size) =
			 qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
						    ref_site_index * per_site_orbital_size,
						    per_site_orbital_size,
						    per_site_orbital_size);
	       }
	  }
     }
     a_dagger_b = a_dagger_b.cwiseProduct(site_symmetry_matrix);
     Sigma_0_ = Sigma_0_.cwiseProduct(site_symmetry_matrix);
     Sigma_1_ = Sigma_1_.cwiseProduct(site_symmetry_matrix);
     for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	  values_[freq_index] = values_[freq_index].cwiseProduct(site_symmetry_matrix);
	  qmc_tail[freq_index] = qmc_tail[freq_index].cwiseProduct(site_symmetry_matrix);
     }
}

void Selfenergy::run_dyson_equation(int ref_site_index,
				    boost::shared_ptr<Greensfunction> greens_function) {
     for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	  values_[freq_index].block(ref_site_index * per_site_orbital_size,
				    ref_site_index * per_site_orbital_size,
				    per_site_orbital_size,
				    per_site_orbital_size) =
	       greens_function->get_dyson_result(freq_index, false);
	  neg_values_[freq_index].block(ref_site_index * per_site_orbital_size,
					ref_site_index * per_site_orbital_size,
					per_site_orbital_size,
					per_site_orbital_size) =
	       greens_function->get_dyson_result(freq_index, true);
     }
}

// Initialize structures common to qmc- and dmft-style objects
void Selfenergy::basic_init(const alps::params &parms, bool verbose) {
     // Read orbital and block structure
     // This is necessary, because it defines the
     // structure of the text file where the self-energy is stored.
     // Sensitive parameter - if too large, the noise at high frequency is translated
     // into shifts in the Legendre representation, and badly wrong estimates
     // of the tails.
     matsubara_tail_estimate_region = std::round(2.0 * static_cast<double>(parms["C_MIN"]));
     n_blocks = static_cast<size_t>(parms["N_BLOCKS"]);
     n_sites = parms.exists("N_SITES") ?
	  static_cast<size_t>(parms["N_SITES"]) : 1;
     per_site_orbital_size = parms.exists("N_ORBITALS") ?
	  static_cast<size_t>(parms["N_ORBITALS"]) : 2;
     tot_orbital_size = per_site_orbital_size * n_sites;
     if (parms.exists("REAL_DELTA")) {
	  enforce_real = static_cast<bool>(parms["REAL_DELTA"]);
     } else {
	  cout << "REAL_DELTA not found in params file ---> reverting to true" << endl;
	  // arguably counter intuitive behavior,
	  // enforced for backwards compatibility.
	  enforce_real = true;
     }
     // Careful - this is the actual number of Matsubara frequencies.
     // not the actual value of N for the max value of omega such that
     // max_freq = 2N + 1 pi / beta, because n = 0 is also a frequency,
     // so that n_matsubara_freqs = N + 1
     if (is_alps3) {
	  //n_matsubara_freqs = static_cast<size_t>(parms["measurement.G1.N_MATSUBARA"]);
	  n_matsubara_freqs = static_cast<size_t>(parms["N_MATSUBARA"]);
     } else {
	  n_matsubara_freqs = static_cast<size_t>(parms["N_MATSUBARA"]);
     }
     init_sigma_container();
     beta = static_cast<double>(parms["BETA"]);
     matsubara_frequencies_ = Eigen::VectorXcd::Zero(n_matsubara_freqs);
     for (size_t freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	  matsubara_frequencies_(freq_index) =
	       complex<double>(0.0, (2.0 * freq_index + 1) * M_PI / beta);
     }
     interaction_matrix =
	  Eigen::MatrixXcd::Constant(tot_orbital_size, tot_orbital_size, 0.0);
     site_symmetry_matrix = Eigen::MatrixXcd::Constant(tot_orbital_size, tot_orbital_size, 1.0);
     density_density_correl = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     a_dagger_b = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     Sigma_0_ = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     Sigma_1_ = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     if (n_blocks < per_site_orbital_size) {
	  is_diagonal = false;
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
	  is_diagonal = true;
	  blocks.resize(n_blocks, std::vector<size_t>());
	  for(std::size_t i=0; i < blocks.size(); ++i) {
	       std::vector<size_t> temp{i};
	       blocks[i] = temp;
	  }
     }
}

void Selfenergy::read_qmc_sigma(int ref_site_index,
				boost::shared_ptr<Greensfunction> greens_function) {
     run_dyson_equation(ref_site_index, greens_function);
     // No need to symmetrize for alps3 data coming from Legendre
}

void Selfenergy::read_qmc_sigma(int ref_site_index, alps::hdf5::archive h5_archive) {
     string qmc_root_path = (input_type == 0) ? "/od_hyb_S_omega" : "/od_hyb_S_l_omega";
     if (!h5_archive.is_group(qmc_root_path)) {
	  throw runtime_error("Trying to mix but no QMC data found !");
     }
     get_qmc_single_site_hdf5_data(ref_site_index, h5_archive, qmc_root_path);
     symmetrize_qmc_sigma(ref_site_index);
}

void Selfenergy::symmetrize_tail_params(int ref_site_index) {
     Eigen::MatrixXcd temp_matrix =
	  Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     temp_matrix = a_dagger_b.block(ref_site_index * per_site_orbital_size,
				    ref_site_index * per_site_orbital_size,
				    per_site_orbital_size,
				    per_site_orbital_size);
     temp_matrix.adjointInPlace();
     a_dagger_b.block(ref_site_index * per_site_orbital_size,
		      ref_site_index * per_site_orbital_size,
		      per_site_orbital_size,
		      per_site_orbital_size) =
	  0.5 * (a_dagger_b.block(ref_site_index * per_site_orbital_size,
				  ref_site_index * per_site_orbital_size,
				  per_site_orbital_size,
				  per_site_orbital_size) +
		 temp_matrix);
}

void Selfenergy::symmetrize_qmc_sigma(int ref_site_index) {
     Eigen::MatrixXcd temp_matrix =
	  Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     if (enforce_real) {
	  for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	       temp_matrix = values_[freq_index].block(ref_site_index * per_site_orbital_size,
						       ref_site_index * per_site_orbital_size,
						       per_site_orbital_size,
						       per_site_orbital_size);
	       temp_matrix.transposeInPlace();
	       values_[freq_index].block(ref_site_index * per_site_orbital_size,
					 ref_site_index * per_site_orbital_size,
					 per_site_orbital_size,
					 per_site_orbital_size) =
		    0.5 * (values_[freq_index].block(
				ref_site_index * per_site_orbital_size,
				ref_site_index * per_site_orbital_size,
				per_site_orbital_size,
				per_site_orbital_size) + temp_matrix);
	  }
     } else {
	  // use G_ij(i omega_n) = G^*_ji(-i omega_n)
	  for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	       temp_matrix = neg_values_[n_matsubara_freqs - 1 - freq_index].
		    block(ref_site_index * per_site_orbital_size,
			  ref_site_index * per_site_orbital_size,
			  per_site_orbital_size,
			  per_site_orbital_size);
	       temp_matrix.adjointInPlace();
	       values_[freq_index].block(ref_site_index * per_site_orbital_size,
					 ref_site_index * per_site_orbital_size,
					 per_site_orbital_size,
					 per_site_orbital_size) =
		    0.5 * (values_[freq_index].block(
				ref_site_index * per_site_orbital_size,
				ref_site_index * per_site_orbital_size,
				per_site_orbital_size,
				per_site_orbital_size) + temp_matrix);
	       
	  }
     }
}

void Selfenergy::init_sigma_container() {
     values_.resize(n_matsubara_freqs);
     neg_values_.resize(n_matsubara_freqs);
     is_nil_sigma = false;
     //initialize self-energy
     for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	  values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
	  neg_values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     }
}

std::vector<double> Selfenergy::get_u_elements(const alps::params &parms) {
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
     return u_elements;
}

void Selfenergy::feed_tail_params(int ref_site_index,
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
     std::vector<double> u_elements = get_u_elements(parms);
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
     if (is_alps3) {
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
	  // g(tau) data
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
	  // now Alps2 version
     } else {
	  int n_tau = boost::lexical_cast<int>(parms["N_TAU"]) + 1;
	  std::string gtau_path("/od_hyb_G_tau/");
	  std::vector<std::complex<double> > temp_data;
	  temp_data.resize(n_tau);
	  double cur_dd_correl;
	  for(int block_index = 0; block_index < n_blocks; ++block_index) {
	       int cur_index = 0;
	       for(int line_idx = 0; line_idx < blocks[block_index].size(); ++line_idx) {
		    for(int col_idx = 0; col_idx < blocks[block_index].size();
			++col_idx) {
			 std::stringstream orbital_path;
			 std::stringstream density_path;
			 // ATTENTION here: convention of QMC is F_ij = -T<c_i c^dag_j>,
			 // but DMFT is looking for  c^dag_i c_j
			 cur_index = line_idx + col_idx * blocks[block_index].size();
			 orbital_path << gtau_path << boost::lexical_cast<std::string>(block_index) +
			      "/" + boost::lexical_cast<std::string>(cur_index) + "/mean/value";
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

void Selfenergy::fit_tails(int ref_site_index) {
     cout << "Sigma_1_ is fitted numerically"
	  << endl << endl;
     size_t N_max = matsubara_tail_estimate_region;
     for (size_t freq_index = N_max - tail_fit_length;
	  freq_index < N_max; freq_index++) {
	  // Sigma_0_.block(ref_site_index * per_site_orbital_size,
	  // 		 ref_site_index * per_site_orbital_size,
	  // 		 per_site_orbital_size,
	  // 		 per_site_orbital_size) += 0.5 *
	  //      (values_[freq_index].block(ref_site_index * per_site_orbital_size,
	  // 				  ref_site_index * per_site_orbital_size,
	  // 				  per_site_orbital_size,
	  // 				  per_site_orbital_size) +
	  // 	values_[freq_index].block(ref_site_index * per_site_orbital_size,
	  // 				  ref_site_index * per_site_orbital_size,
	  // 				  per_site_orbital_size,
	  // 				  per_site_orbital_size).transpose().conjugate()) / tail_fit_length;
	  Sigma_1_.block(ref_site_index * per_site_orbital_size,
			 ref_site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size) +=
	       //0.5 * get_matsubara_frequency(freq_index) *
	       0.5 * get_matsubara_frequency(freq_index) *
	       (values_[freq_index].block(ref_site_index * per_site_orbital_size,
					  ref_site_index * per_site_orbital_size,
					  per_site_orbital_size,
					  per_site_orbital_size) -
		values_[freq_index].block(ref_site_index * per_site_orbital_size,
					  ref_site_index * per_site_orbital_size,
					  per_site_orbital_size,
					  per_site_orbital_size).transpose().conjugate()) / tail_fit_length;
     }
}

void Selfenergy::compute_tail_coeffs( boost::shared_ptr<Greensfunction> greens_function,
				      boost::shared_ptr<Chemicalpotential> chempot,
				      int ref_site_index) {
     cout << "SELF ENERGY tails from Legendre cumulatives" << endl << endl;
     Sigma_0_.block(ref_site_index * per_site_orbital_size,
		    ref_site_index * per_site_orbital_size,
		    per_site_orbital_size,
		    per_site_orbital_size) = greens_function->get_measured_c2();
     for (int orbital = 0; orbital < per_site_orbital_size; orbital++) {
     	  Sigma_0_.block(ref_site_index * per_site_orbital_size,
     			ref_site_index * per_site_orbital_size,
     			per_site_orbital_size,
     			per_site_orbital_size)(orbital, orbital) +=
     	       (*chempot)[orbital];
     }
     Sigma_1_.block(ref_site_index * per_site_orbital_size,
		    ref_site_index * per_site_orbital_size,
		    per_site_orbital_size,
		    per_site_orbital_size) =
	  greens_function->get_measured_c3() - (
	       greens_function->get_measured_c2() *
	       greens_function->get_measured_c2());
}

void Selfenergy::compute_tail_coeffs(int ref_site_index) {
     // 0.5 factor stems from the fact that the formal expression
     // of the Hamiltonian is without order in the thesis, while it has
     // order (a before b) in the papers ==> equivalent to specifying U/2
     // in the papers
     // For details of derivation of formulae, see Gull thesis,
     // Appendix B.4, or hopefully even better, my own thesis, appendices.
     // Sell also Ferber's thesis for order 3 coeff.
     cout << "analytic trail treatment for tail of SELF ENERGY"
	  << endl << endl;
     Sigma_0_.block(ref_site_index * per_site_orbital_size,
		    ref_site_index * per_site_orbital_size,
		    per_site_orbital_size,
		    per_site_orbital_size) =
	  -0.5 * (interaction_matrix.block(ref_site_index * per_site_orbital_size,
					   ref_site_index * per_site_orbital_size,
					   per_site_orbital_size,
					   per_site_orbital_size) +
		  interaction_matrix.block(ref_site_index * per_site_orbital_size,
					   ref_site_index * per_site_orbital_size,
					   per_site_orbital_size,
					   per_site_orbital_size).transpose())
	  .cwiseProduct(a_dagger_b.block(
			     ref_site_index * per_site_orbital_size,
			     ref_site_index * per_site_orbital_size,
			     per_site_orbital_size,
			     per_site_orbital_size));
     // Now deal with diagonal terms
     Sigma_0_.block(ref_site_index * per_site_orbital_size,
		    ref_site_index * per_site_orbital_size,
		    per_site_orbital_size,
		    per_site_orbital_size).diagonal() =
	  Eigen::VectorXcd::Zero(per_site_orbital_size);
     for(int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx) {
	  for(int col_idx = 0; col_idx < per_site_orbital_size; ++col_idx) {
	       // HERE Note that there is some possible discrepancy to remove between
	       // dmft and qmc: on the dmft side, the crystal field is
	       // input via the diagonal elements of the hopping matrix,
	       // while on the qmc side it is blent with mu in MUvector.
	       // as a consequence, the energy of each orbital in the present code
	       // is indeed the \epsilon_k needed in the formula from my thesis.
	       Sigma_0_.block(ref_site_index * per_site_orbital_size,
			      ref_site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size)(line_idx, line_idx) +=
		    0.5 * (interaction_matrix.block(ref_site_index * per_site_orbital_size,
						    ref_site_index * per_site_orbital_size,
						    per_site_orbital_size,
						    per_site_orbital_size)(line_idx, col_idx) +
			   interaction_matrix.block(ref_site_index * per_site_orbital_size,
						    ref_site_index * per_site_orbital_size,
						    per_site_orbital_size,
						    per_site_orbital_size)(col_idx, line_idx))*
		    density_density_correl.block(ref_site_index * per_site_orbital_size,
						 ref_site_index * per_site_orbital_size,
						 per_site_orbital_size,
						 per_site_orbital_size)(col_idx, col_idx);
	       for(int ter_idx = 0; ter_idx < per_site_orbital_size; ++ter_idx) {
		    Sigma_1_.block(ref_site_index * per_site_orbital_size,
				   ref_site_index * per_site_orbital_size,
				   per_site_orbital_size,
				   per_site_orbital_size)(line_idx, line_idx) +=
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
     if (!is_analytic_tail) {
	  fit_tails(ref_site_index);
     }
}


void Selfenergy::log_sigma_tails(int ref_site_index) {
     // log for sigma asymptotics
     cout << "QMC Sigma asymptotics : site " << ref_site_index << endl
	  << "Sigma_inf "<< endl << endl;
     cout << Sigma_0_.block(ref_site_index * per_site_orbital_size,
			    ref_site_index * per_site_orbital_size,
			    per_site_orbital_size,
			    per_site_orbital_size) << endl << endl;
     cout << "Sigma_1 "<< endl << endl;
     cout << Sigma_1_.block(ref_site_index * per_site_orbital_size,
			    ref_site_index * per_site_orbital_size,
			    per_site_orbital_size,
			    per_site_orbital_size) << endl << endl;
}

void Selfenergy::compute_qmc_tail(int ref_site_index) {
     //initialize qmc tail
     qmc_tail.clear();
     for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	  qmc_tail.push_back(Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size));
	  qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
				     ref_site_index * per_site_orbital_size,
				     per_site_orbital_size,
				     per_site_orbital_size) =
	       Sigma_0_.block(ref_site_index * per_site_orbital_size,
			      ref_site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size) +
	       Sigma_1_.block(ref_site_index * per_site_orbital_size,
			      ref_site_index * per_site_orbital_size,
			      per_site_orbital_size,
			      per_site_orbital_size)
	       / matsubara_frequencies_[freq_index];
     }
}

void Selfenergy::append_qmc_tail(int ref_site_index,
				 const alps::params &parms) {
     double sum = 0.0;
     int fine_steps = 2000;
     int raw_steps = 200;
     double c_min = parms["C_MIN"], c_max = parms["C_MAX"];
     Eigen::VectorXd base_func(fine_steps);
     Eigen::VectorXd conn_func(raw_steps);
     for (int i = 0; i < fine_steps; i++) {
	  double x = double(i - fine_steps / 2) / (fine_steps / 2);
	  base_func(i) = exp(-1.0 / (1.0 - x * x));
     }
     base_func = base_func / base_func.sum();
     for (int i = 0; i < raw_steps; i++) {
	  int j_min = int(double(fine_steps / 2) * double(i) / (raw_steps / 2));
	  conn_func(i) = base_func.tail(fine_steps - j_min).sum();
     }
     conn_func(0) = 1.0;
     conn_func(raw_steps - 1) = 0.0;
     Eigen::VectorXd append_weights(n_matsubara_freqs);
     for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	  if (std::abs(matsubara_frequencies_[freq_index]) < c_min) {
	       append_weights(freq_index) = 1.0;
	  } else if (std::abs(matsubara_frequencies_[freq_index]) > c_max) {
	       append_weights(freq_index) = 0.0;
	  } else {
	       int target_index = floor(raw_steps / 2 *
					(2 * std::abs(matsubara_frequencies_[freq_index])
					 - c_min - c_max) / (c_max - c_min));
	       target_index += raw_steps / 2;
	       append_weights(freq_index) = conn_func(target_index);
	  }
	  values_[freq_index].block(ref_site_index * per_site_orbital_size,
				    ref_site_index * per_site_orbital_size,
				    per_site_orbital_size,
				    per_site_orbital_size) =
	       append_weights(freq_index) *
	       values_[freq_index].block(ref_site_index * per_site_orbital_size,
					 ref_site_index * per_site_orbital_size,
					 per_site_orbital_size,
					 per_site_orbital_size)
	       + (1.0 - append_weights(freq_index)) *
	       qmc_tail[freq_index].block(ref_site_index * per_site_orbital_size,
					  ref_site_index * per_site_orbital_size,
					  per_site_orbital_size,
					  per_site_orbital_size);
     }
}

// Get the raw data from qmc from hdf5 format. Raw data, i.e.
// non smoothed self energy.
void Selfenergy::get_qmc_single_site_hdf5_data(size_t site_index,
					       alps::hdf5::archive h5_archive,
					       string rootpath) {
	std::vector<std::complex<double>> temp_data;
	temp_data.resize(2 * n_matsubara_freqs);
	for (size_t block_index = 0; block_index < n_blocks; block_index++) {
		size_t cur_index = 0;
		for (size_t orb1 = 0; orb1 < blocks[block_index].size(); orb1++) {
			for (size_t orb2 = 0; orb2 < blocks[block_index].size(); orb2++) {
				cur_index = blocks[block_index][orb2] *  blocks[block_index].size() +
					blocks[block_index][orb1];
				std::stringstream orbital_path;
				orbital_path << rootpath << "/" +
					boost::lexical_cast<std::string>(block_index) +
					"/" + boost::lexical_cast<std::string>(cur_index) + "/mean/value";
				h5_archive >> alps::make_pvp(orbital_path.str(), temp_data);
				for (size_t freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
					values_[freq_index].block(site_index * per_site_orbital_size,
								  site_index * per_site_orbital_size,
								  per_site_orbital_size,
								  per_site_orbital_size)
					     (blocks[block_index][orb1],
					      blocks[block_index][orb2]) =
					     temp_data[freq_index];
					neg_values_[freq_index].block(site_index * per_site_orbital_size,
								      site_index * per_site_orbital_size,
								      per_site_orbital_size,
								      per_site_orbital_size)
					     (blocks[block_index][orb1],
					      blocks[block_index][orb2]) =
					     temp_data[n_matsubara_freqs + freq_index];
				}
			}
		}
	}
}

void Selfenergy::get_single_site_hdf5_data(size_t site_index,
					   alps::hdf5::archive h5_archive,
					   string rootpath) {
     std::vector<std::complex<double>> temp_data;
     temp_data.resize(n_matsubara_freqs);
     for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
	  for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
	       std::stringstream orbital_pair_path;
	       orbital_pair_path << rootpath << "/" +
		    boost::lexical_cast<std::string>(orb1) + "/" +
		    boost::lexical_cast<std::string>(orb2) + "/values/value";
	       h5_archive >> alps::make_pvp(orbital_pair_path.str(),
	       			    temp_data);
	       for (size_t freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
		    values_[freq_index].block(site_index * per_site_orbital_size,
					      site_index * per_site_orbital_size,
					      per_site_orbital_size,
					      per_site_orbital_size)(orb1, orb2) =
			 temp_data[freq_index];
	       }
	  }
     }
}

Eigen::MatrixXcd Selfenergy::get_single_site_hdf5_asymptotics(size_t site_index,
						  alps::hdf5::archive h5_archive,
						  string rootpath, int asymptotic_order) {
     // The asymptotics arrays are initialized in basic init
     // we can work with them directly here.
     std::vector<std::complex<double>> temp_data;
     temp_data.resize(1);
     Eigen::MatrixXcd output =
	  Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
	  for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
	       std::stringstream orbital_pair_path;
	       orbital_pair_path << rootpath << "/" +
		    boost::lexical_cast<std::string>(orb1) + "/" +
		    boost::lexical_cast<std::string>(orb2) +
		    "/tail_" + boost::lexical_cast<std::string>(asymptotic_order) +
		    "/value";
	       h5_archive >> alps::make_pvp(orbital_pair_path.str(),
	       			    temp_data);
	       output(orb1, orb2) =
	       temp_data[0];
	  }
     }
     return output;
}

void Selfenergy::read_input_sigma(const alps::params &parms,
				  alps::hdf5::archive h5_archive,
				  string h5_group_name) {
     // initialize container
     bool is_hdf5_input = static_cast<bool>(parms["SIGMA_IN_HDF5"]);
     if (is_hdf5_input) {
	  for (size_t site_index = 0; site_index < n_sites; site_index++) {
	       std::stringstream rootpath;
	       rootpath << h5_group_name + "/site_"
			<< boost::lexical_cast<std::string>(site_index);
	       if (!h5_archive.is_group(rootpath.str())) {
		    cout << "Could not find group " + rootpath.str() +
			 ". Assuming zero SELF ENERGY" << endl;
		    is_nil_sigma = true;
	       } else {
		    get_single_site_hdf5_data(site_index, h5_archive, rootpath.str());
		    // Read in asymptotics
		    Sigma_0_.block(site_index * per_site_orbital_size,
				   site_index * per_site_orbital_size,
				   per_site_orbital_size,
				   per_site_orbital_size) =
			 get_single_site_hdf5_asymptotics(site_index, h5_archive, rootpath.str(), 0);
		    Sigma_1_.block(site_index * per_site_orbital_size,
				   site_index * per_site_orbital_size,
				   per_site_orbital_size,
				   per_site_orbital_size) =
			 get_single_site_hdf5_asymptotics(site_index, h5_archive, rootpath.str(), 1);
	       }
	  }
     } else {
	  //read from text file - Cf hybfun.cpp for inspiration for other formats.
	  // define input file
	  // NOT SUPPORTED ANYMORE
	  std::cerr << "TXT format for self-energy not supoprted because of "
	       "new tail transfer policy in "
	       + std::string(__FUNCTION__);
	  throw std::runtime_error("pb reading sigma in txt format");
	  std::string fname = parms["SIGMA"];
	  std::ifstream infile(fname.c_str());
	  if(!infile.good()) {
	       cout << "Could not find file " << parms["SIGMA"] <<
		    ". Assuming zero SELF ENERGY" << endl;
	       is_nil_sigma = true;
	  } else {
	       for (size_t site_index = 0; site_index < n_sites; site_index++) {
		    for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
			 for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			      for (size_t freq_index = 0;
				   freq_index < n_matsubara_freqs; freq_index++) {
				   std::string line;
				   std::getline(infile, line);
				   std::istringstream iss(line);
				   double dummy, re_part, im_part;
				   iss >> dummy >> re_part >> im_part;
				   values_[freq_index].block(site_index * per_site_orbital_size,
							     site_index * per_site_orbital_size,
							     per_site_orbital_size,
							     per_site_orbital_size)(orb1, orb2) =
					complex<double>(re_part, im_part);
			      }
			 }
		    }
	       }
	  }
     }
}

Eigen::MatrixXcd  Selfenergy::get_sigma_0() {
     return Sigma_0_;
}

Eigen::MatrixXcd  Selfenergy::get_sigma_1() {
     return Sigma_1_;
}

void Selfenergy::display_asymptotics() {
     if (world_rank_ == 0) {
	  auto old_precision = cout.precision(5);
	  for (size_t site_index = 0; site_index < n_sites; site_index++) {
	       cout << "SELF-ENERGY ASYMPTOTICS for site " <<
		    site_index << ": " << endl << endl;;
	       cout << "ETA(infinity): " << endl;
	       cout << Sigma_0_.block(site_index * per_site_orbital_size,
				      site_index * per_site_orbital_size,
				      per_site_orbital_size,
				      per_site_orbital_size) << endl << endl;
	       cout << "ETA(1): " << endl;		
	       cout << Sigma_1_.block(site_index * per_site_orbital_size,
				      site_index * per_site_orbital_size,
				      per_site_orbital_size,
				      per_site_orbital_size) << endl << endl;
	  }
	  cout.precision(old_precision);
     }
}

void Selfenergy::compute_order2_partial_sum() {
     order2_partial_sum_ =
	  matsubara_frequencies_.unaryExpr(ptr_fun(square_inverse)).sum();
}

void Selfenergy::apply_linear_combination(boost::shared_ptr<Selfenergy> const &old_sigma,
					  double alpha)
{
     for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	  values_[freq_index] = alpha * values_[freq_index] +
	       (1.0 - alpha) * old_sigma->values_[freq_index];
     }
     Sigma_0_ = alpha * Sigma_0_ + (1.0 - alpha) * old_sigma->Sigma_0_;
     Sigma_1_ = alpha * Sigma_1_ + (1.0 - alpha) * old_sigma->Sigma_1_;
}

double Selfenergy::get_order2_partial_sum() {
     return order2_partial_sum_;
}

double Selfenergy::get_beta() {
     return beta;
}

size_t Selfenergy::get_per_site_orbital_size() {
     return static_cast<int>(per_site_orbital_size);
}

int Selfenergy::get_n_sites() {
     return static_cast<int>(n_sites);
}

size_t Selfenergy::get_n_matsubara_freqs() { return n_matsubara_freqs; }

std::complex<double> Selfenergy::get_matsubara_frequency(size_t n) {
     return matsubara_frequencies_(n);
}

Eigen::VectorXcd Selfenergy::get_matsubara_frequencies() {
     return matsubara_frequencies_;
}

void Selfenergy::hdf5_dump(alps::hdf5::archive h5_archive, string h5_group_name) {
     for (int site_index = 0; site_index < n_sites; site_index++) {
	  std::stringstream site_path;
	  site_path << h5_group_name + "/site_" +
	       boost::lexical_cast<std::string>(site_index) + "/";
	  for(int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx) {
	       for(int col_idx = 0; col_idx < per_site_orbital_size; ++col_idx) {
		    std::stringstream orbital_path;
		    orbital_path << site_path.str() <<
			 boost::lexical_cast<std::string>(line_idx) + "/"
			 + boost::lexical_cast<std::string>(col_idx) + "/values/value";
		    std::vector<std::complex<double>> temp_data;
		    temp_data.resize(n_matsubara_freqs);
		    for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
			 temp_data[freq_index] =
			      values_[freq_index].block(site_index * per_site_orbital_size,
							site_index * per_site_orbital_size,
							per_site_orbital_size,
							per_site_orbital_size)(line_idx, col_idx);
		    }
		    h5_archive << alps::make_pvp(orbital_path.str(),
		    				 temp_data);
	       }
	  }
     }
}

void Selfenergy::hdf5_dump_tail(alps::hdf5::archive h5_archive, string h5_group_name,
				int ref_site_index, int tail_order) {
     for (int site_index = 0; site_index < n_sites; site_index++) {
	  std::stringstream site_path;
	  site_path << h5_group_name + "/site_" +
	       boost::lexical_cast<std::string>(site_index) + "/";
	  for (int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx) {
	       for (int col_idx = 0; col_idx < per_site_orbital_size; ++col_idx) {
		    std::stringstream orbital_path;
		    orbital_path << site_path.str() <<
			 boost::lexical_cast<std::string>(line_idx) + "/"
			 + boost::lexical_cast<std::string>(col_idx) +
			 "/tail_" + boost::lexical_cast<std::string>(tail_order) +
			 "/value";
		    std::vector<std::complex<double>> temp_data;
		    if (tail_order == 0) {
			 temp_data.push_back(Sigma_0_.block(
						  site_index * per_site_orbital_size,
						  site_index * per_site_orbital_size,
						  per_site_orbital_size,
						  per_site_orbital_size)(line_idx, col_idx));			 
		    } else if (tail_order == 1) {
			 temp_data.push_back(Sigma_1_.block(
						  site_index * per_site_orbital_size,
						  site_index * per_site_orbital_size,
						  per_site_orbital_size,
						  per_site_orbital_size)(line_idx, col_idx));
		    } else {
			 throw std::runtime_error("Tail order is not correct in " +
						  std::string(__FUNCTION__));
		    }
		    h5_archive << alps::make_pvp(orbital_path.str(),
		    				 temp_data);
	       }
	  }
     }
}

const size_t Selfenergy::tail_fit_length = 10;
const std::string Selfenergy::density_density_result_name = "DENSITY_DENSITY_CORRELATION_FUNCTIONS";
