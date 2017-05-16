#include <boost/multi_array.hpp>
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
     :GfBase(parms, world_rank), is_alps3(false), is_analytic_tail(true) {
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
		       boost::shared_ptr<Bandstructure> const &lattice_bs,
		       int ref_site_index, alps::hdf5::archive h5_archive, int input_type,
		       bool verbose)
     :GfBase(parms, world_rank, chempot, lattice_bs), input_type(input_type), is_alps3(false) {
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
		       boost::shared_ptr<Bandstructure> const &lattice_bs,
		       int ref_site_index, alps::hdf5::archive h5_archive,
		       boost::shared_ptr<Greensfunction> greens_function)
     :GfBase(parms, world_rank, chempot,lattice_bs ), input_type(1), is_alps3(true) {
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
     compute_tail_coeffs(greens_function, ref_site_index);
     log_sigma_tails(ref_site_index);
     compute_qmc_tail(ref_site_index);
     // No need to append tails when data comes from Legendre
     append_qmc_tail(ref_site_index, parms);
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
	  for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
	  for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
	       temp_matrix = neg_values_[n_matsubara - 1 - freq_index].
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
	       for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
	  values_[freq_index] = values_[freq_index].cwiseProduct(site_symmetry_matrix);
	  qmc_tail[freq_index] = qmc_tail[freq_index].cwiseProduct(site_symmetry_matrix);
     }
}

void Selfenergy::run_dyson_equation(int ref_site_index,
				    boost::shared_ptr<Greensfunction> greens_function) {
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
     is_diagonal = (n_blocks < per_site_orbital_size) ? false : true;
     matsubara_tail_estimate_region = std::round(2.0 * static_cast<double>(parms["C_MIN"]));
     if (parms.exists("REAL_DELTA")) {
	  enforce_real = static_cast<bool>(parms["REAL_DELTA"]);
     } else {
	  cout << "REAL_DELTA not found in params file ---> reverting to true" << endl;
	  // arguably counter intuitive behavior,
	  // enforced for backwards compatibility.
	  enforce_real = true;
     }
     init_sigma_container();
     Sigma_0_ = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     Sigma_1_ = Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
}

void Selfenergy::read_qmc_sigma(int ref_site_index,
				boost::shared_ptr<Greensfunction> greens_function) {
     run_dyson_equation(ref_site_index, greens_function);
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
	  for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
	  for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
	       temp_matrix = neg_values_[n_matsubara - 1 - freq_index].
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
     values_.resize(n_matsubara);
     neg_values_.resize(n_matsubara);
     is_nil_sigma = false;
     //initialize self-energy
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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

void Selfenergy::compute_tail_coeffs(boost::shared_ptr<Greensfunction> greens_function,
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
     	       (*chempot_)[orbital];
     }
     Sigma_1_.block(ref_site_index * per_site_orbital_size,
		    ref_site_index * per_site_orbital_size,
		    per_site_orbital_size,
		    per_site_orbital_size) =
	  Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     Sigma_1_.block(ref_site_index * per_site_orbital_size,
		    ref_site_index * per_site_orbital_size,
		    per_site_orbital_size,
		    per_site_orbital_size).diagonal() =
	  (greens_function->get_measured_c3() - (
	       greens_function->get_measured_c2() *
	       greens_function->get_measured_c2())).diagonal();
}

void Selfenergy::compute_tail_coeffs(int ref_site_index) {
     cout << "analytic trail treatment for tail of SELF ENERGY"
	  << endl << endl;
     Sigma_0_.block(ref_site_index * per_site_orbital_size,
		    ref_site_index * per_site_orbital_size,
		    per_site_orbital_size,
		    per_site_orbital_size) = target_c2;
     for (int orbital = 0; orbital < per_site_orbital_size; orbital++) {
     	  Sigma_0_.block(ref_site_index * per_site_orbital_size,
			 ref_site_index * per_site_orbital_size,
			 per_site_orbital_size,
			 per_site_orbital_size)(orbital, orbital) +=
     	       (*chempot_)[orbital];
     }     
     for(int line_idx = 0; line_idx < per_site_orbital_size; ++line_idx) {
	  for(int col_idx = 0; col_idx < per_site_orbital_size; ++col_idx) {
	       // HERE Note that there is some possible discrepancy to remove between
	       // dmft and qmc: on the dmft side, the crystal field is
	       // input via the diagonal elements of the hopping matrix,
	       // while on the qmc side it is blent with mu in MUvector.
	       // as a consequence, the energy of each orbital in the present code
	       // is indeed the \epsilon_k needed in the formula from my thesis.
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
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
     Eigen::VectorXd append_weights(n_matsubara);
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
	temp_data.resize(2 * n_matsubara);
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
				if ((cur_index == 0)) {
				}
				for (size_t freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
					     temp_data[n_matsubara + freq_index];
				}
			}
		}
	}
}

void Selfenergy::get_single_site_hdf5_data(size_t site_index,
					   alps::hdf5::archive h5_archive,
					   string rootpath) {
     std::vector<std::complex<double>> temp_data;
     temp_data.resize(n_matsubara);
     for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
	  for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
	       std::stringstream orbital_pair_path;
	       orbital_pair_path << rootpath << "/" +
		    boost::lexical_cast<std::string>(orb1) + "/" +
		    boost::lexical_cast<std::string>(orb2) + "/values/value";
	       h5_archive >> alps::make_pvp(orbital_pair_path.str(),
	       			    temp_data);
	       for (size_t freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
				   freq_index < n_matsubara; freq_index++) {
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
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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

size_t Selfenergy::get_n_matsubara_freqs() { return n_matsubara; }

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
		    temp_data.resize(n_matsubara);
		    for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
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
