//#include <boost/math/special_functions/bessel.hpp>
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
#include "greens_function.hpp"
#include "hybridization_function.hpp"

using namespace std;
typedef boost::multi_array<complex<double> , 3> cplx_array_type;
typedef boost::multi_array<double , 4> real_array_type;

// transformation matrix from Legendre to Matsubara basis
// std::complex<double> t_coeff(int n, int l) {
//      std::complex<double> i_c(0., 1.);
//      std::complex<double> out = std::sqrt(static_cast<double>(2 * l + 1)) * // /
// 				 //	 std::sqrt(static_cast<double>(std::abs(2 * n + 1)))) *
// 	  std::exp(i_c * (n + 0.5) * M_PI) * std::pow(i_c, l) *
// 	  boost::math::sph_bessel(l, 0.5 * std::abs(2 * n + 1) * M_PI);
//      if (n < 0) {
// 	  out *= std::pow(-1.0, l);
//      }
//      return out;
// }

Greensfunction::Greensfunction(const alps::params &parms, int world_rank,
			       int sampling_type, alps::hdf5::archive &h5_archive)
     :sampling_type(sampling_type), world_rank_(world_rank) {
     basic_init(parms);
     read_bare_gf();
     read_t_coeffs(h5_archive);
     generate_data(h5_archive);
}

std::complex<double> Greensfunction::get_t_coeff(int n, int l) {
     std::complex<double> out = full_t_set(std::abs(n), l);
     if (n < 0) {
	  out *= std::pow(-1.0, l);
     }
     return out;
}

void Greensfunction::generate_data(alps::hdf5::archive &h5_archive) {
     read_single_site_legendre(h5_archive);
     get_matsubara_from_legendre();
}

void Greensfunction::read_t_coeffs(alps::hdf5::archive &h5_archive) {
     boost::multi_array<std::complex<double> , 2> init_full_t_set(boost::extents[1000][200]);
     h5_archive["t_coeffs"] >>  init_full_t_set;
     full_t_set = Eigen::MatrixXcd::Zero(1000, 200);
     for (int n_index = 0; n_index < 1000; n_index++) {
	  for (int l_index = 0; l_index < 200; l_index++) {
	       full_t_set(n_index, l_index) = init_full_t_set[n_index][l_index];
	  }
     }
}

// void Greensfunction::generate_t_coeffs(alps::hdf5::archive &h5_archive) {
//      boost::multi_array<complex<double> , 2> full_t_set(boost::extents[1000][200]);
//      for (int n_index = 0; n_index < 1000; n_index++) {
// 	  for (int l_index = 0;l_index < 200; l_index++) {
// 	       full_t_set[n_index][l_index] = t_coeff(n_index, l_index);
// 	  }
//      }
//      h5_archive["/t_coeffs"] = full_t_set;
// }

void Greensfunction::basic_init(const alps::params &parms) {
     ref_site_index = 0;
     n_blocks = static_cast<size_t>(parms["N_BLOCKS"]);
     n_sites = parms.exists("N_SITES") ?
	  static_cast<size_t>(parms["N_SITES"]) : 1;
     per_site_orbital_size = parms.exists("N_ORBITALS") ?
	  static_cast<size_t>(parms["N_ORBITALS"]) : 2;
     tot_orbital_size = per_site_orbital_size * n_sites;
     n_matsubara = static_cast<int>(parms["measurement.G1.N_MATSUBARA"]);
     n_matsubara_for_alps2 = static_cast<int>(parms["N_MATSUBARA"]);
     n_legendre = static_cast<int>(parms["cthyb.N_LEGENDRE"]);
     l_max = static_cast<int>(parms["mixing.L_MAX"]);
     beta = static_cast<double>(parms["BETA"]);
     init_gf_container();
}

void Greensfunction::read_bare_gf() {
     std::ifstream infile(HybFunction::bare_gf_no_shift_dump_name.c_str());
     //std::ifstream infile(HybFunction::matsubara_bare_gf_dump_name.c_str());     
     if(!infile.good()) {
	  std::cerr << "Could not find file " << HybFunction::bare_gf_no_shift_dump_name << endl;
	  throw std::runtime_error("pb reading bare GF in txt format");
     } else {
	  for (size_t site_index = 0; site_index < n_sites; site_index++) {
	       for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
		    for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
			 for (size_t freq_index = 0;
			      freq_index < n_matsubara_for_alps2; freq_index++) {
			      std::string line;
			      std::getline(infile, line);
			      std::istringstream iss(line);
			      double dummy, re_part, im_part;
			      iss >> dummy >> re_part >> im_part;
			      bare_gf_values_[freq_index].block(site_index * per_site_orbital_size,
								site_index * per_site_orbital_size,
								per_site_orbital_size,
								per_site_orbital_size)(orb1, orb2) =
				   complex<double>(re_part, im_part);
			 }
		    }
	       }
	       for (size_t freq_index = 0;
		    freq_index < n_matsubara_for_alps2; freq_index++) {
		    bare_gf_neg_values_[n_matsubara_for_alps2 - 1 - freq_index].
			 block(site_index * per_site_orbital_size,
			       site_index * per_site_orbital_size,
			       per_site_orbital_size,
			       per_site_orbital_size) =
			 bare_gf_values_[freq_index].block(site_index * per_site_orbital_size,
							   site_index * per_site_orbital_size,
							   per_site_orbital_size,
							   per_site_orbital_size).transpose().conjugate();
	       }
	  }
	  std::cout << "bare gf" << std::endl;
	  std::cout << bare_gf_values_[50] << std::endl;
     }
}

Eigen::MatrixXcd Greensfunction::get_measured_c2() {
     return measured_c2;
}

Eigen::MatrixXcd Greensfunction::get_measured_c3() {
     return measured_c3;
}

Eigen::MatrixXcd Greensfunction::get_dyson_result(int freq_index, bool is_negative_freq) {
     assert(freq_index < n_matsubara_for_alps2);
     if (!is_negative_freq) {
	  return (-full_gf_values_[freq_index].block(ref_site_index * per_site_orbital_size,
						     ref_site_index * per_site_orbital_size,
						     per_site_orbital_size,
						     per_site_orbital_size).inverse() +
		  bare_gf_values_[freq_index].block(ref_site_index * per_site_orbital_size,
						    ref_site_index * per_site_orbital_size,
						    per_site_orbital_size,
						    per_site_orbital_size).inverse()); 
     } else {
	  return (-full_gf_neg_values_[freq_index].block(ref_site_index * per_site_orbital_size,
							 ref_site_index * per_site_orbital_size,
							 per_site_orbital_size,
							 per_site_orbital_size).inverse() +
		  bare_gf_neg_values_[freq_index].block(ref_site_index * per_site_orbital_size,
							ref_site_index * per_site_orbital_size,
							per_site_orbital_size,
							per_site_orbital_size).inverse());
     }
}

// void Greensfunction::read_single_site_legendre(alps::hdf5::archive &h5_archive, int site_index) {
//      typedef boost::multi_array<double, 4> array_type;
//      std::cerr << "Legendre input not supported " << endl;
//      throw std::runtime_error("Legendre input not supported for 1p GF");
//      array_type legendre_data(
// 	  boost::extents[per_site_orbital_size][per_site_orbital_size][n_legendre][2]);
//      h5_archive["G1_LEGENDRE"] >> legendre_data;
//      LegendreTransformer legendre_transformer(n_matsubara, n_legendre);
//      const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &Tnl(legendre_transformer.Tnl());
//      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> neg_Tnl(Tnl);
//      Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
// 	  tmp_mat(n_legendre, 1), tmp_mat2(n_matsubara, 1), neg_tmp_mat2(n_matsubara, 1);
//      for (int freq_index = 0; freq_index < n_matsubara; ++freq_index) {
// 	  for (int il = 0; il < n_legendre; ++il) {
// 	       neg_Tnl(n_matsubara - 1 - freq_index, il) =
// 		    std::pow(-1.0, il) * Tnl(freq_index, il);
// 	  }
//      }
//      for (int flavor = 0; flavor < per_site_orbital_size; ++flavor) {
// 	  for (int flavor2 = 0; flavor2 < per_site_orbital_size; ++flavor2) {
// 	       for (int il = 0; il < n_legendre; ++il) {
// 		    tmp_mat(il, 0) = std::complex<double>(legendre_data[flavor][flavor2][il][0],
// 							  legendre_data[flavor][flavor2][il][1]);
// 	       }
// 	       tmp_mat2 = Tnl * tmp_mat;
// 	       neg_tmp_mat2 = neg_Tnl * tmp_mat;
// 	       for (int freq_index = 0; freq_index < n_matsubara; ++freq_index) {
// 		    full_gf_values_[freq_index].block(site_index * per_site_orbital_size,
// 						      site_index * per_site_orbital_size,
// 						      per_site_orbital_size,
// 						      per_site_orbital_size)(flavor, flavor2) =
// 			 tmp_mat2(freq_index, 0);		    
// 		    full_gf_neg_values_[freq_index].block(site_index * per_site_orbital_size,
// 							  site_index * per_site_orbital_size,
// 							  per_site_orbital_size,
// 							  per_site_orbital_size)(flavor, flavor2) =
// 			 neg_tmp_mat2(freq_index, 0);
// 	       }
// 	  }
//      }
// }

void Greensfunction::get_matsubara_from_legendre(int site_index) {
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
	  full_gf_values_[freq_index].block(site_index * per_site_orbital_size,
					    site_index * per_site_orbital_size,
					    per_site_orbital_size,
					    per_site_orbital_size) =
	       Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
	  full_gf_neg_values_[n_matsubara - 1 - freq_index].block(site_index * per_site_orbital_size,
								  site_index * per_site_orbital_size,
								  per_site_orbital_size,
								  per_site_orbital_size) =
	       Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
	  for (int l = 0; l < l_max; l++) {
	  	  full_gf_values_[freq_index].block(site_index * per_site_orbital_size,
						    site_index * per_site_orbital_size,
						    per_site_orbital_size,
						    per_site_orbital_size) +=
		       gl_values_[l] * get_t_coeff(freq_index, l);
		  full_gf_neg_values_[n_matsubara - 1 - freq_index].block(site_index * per_site_orbital_size,
						    site_index * per_site_orbital_size,
						    per_site_orbital_size,
						    per_site_orbital_size) +=
		       gl_values_[l] * get_t_coeff(-freq_index, l);
	  }
     }
}

void Greensfunction::read_single_site_full_gf_matsubara(alps::hdf5::archive &h5_archive, int site_index) {
     cplx_array_type raw_full_gf(
      	  boost::extents[n_matsubara][per_site_orbital_size][per_site_orbital_size]);
     h5_archive["/gf/data"] >> raw_full_gf;
     //typedef cplx_array_type::index_range range;
     // cplx_array_type::array_view<3>::type myview =
     //      raw_full_gf[ boost::indices[range(freq_index)][range()][range()] ];
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
	  for (int flavor = 0; flavor < per_site_orbital_size; ++flavor) {
	       for (int flavor2 = 0; flavor2 < per_site_orbital_size; ++flavor2) {
		    full_gf_values_[freq_index].block(site_index * per_site_orbital_size,
						      site_index * per_site_orbital_size,
						      per_site_orbital_size,
						      per_site_orbital_size)(flavor, flavor2) =
			 raw_full_gf[freq_index][flavor][flavor2];
	       }
	  }
	  full_gf_neg_values_[n_matsubara - 1 - freq_index].block(site_index * per_site_orbital_size,
								  site_index * per_site_orbital_size,
								  per_site_orbital_size,
								  per_site_orbital_size) =
	       full_gf_values_[freq_index].block(site_index * per_site_orbital_size,
						 site_index * per_site_orbital_size,
						 per_site_orbital_size,
						 per_site_orbital_size).transpose().conjugate();
     }
}

void Greensfunction::read_single_site_legendre(alps::hdf5::archive &h5_archive, int site_index) {
     cplx_array_type raw_legendre_data(
	  boost::extents[per_site_orbital_size][per_site_orbital_size][n_legendre]);
     if (l_max < 5) {
	     std::string err_string = "L_MAX is less than 5, I think this is not reasonable, stopping now! (in " +
		     std::string(__FUNCTION__) + ")";
	     std::cerr << err_string << std::endl;
	     throw runtime_error(err_string);
     }
     // Read MC output, depending on engine.
     if (sampling_type == 1) {
	  h5_archive["G1_LEGENDRE"] >> raw_legendre_data;
     } else {
	  real_array_type real_raw_legendre_data(
	       boost::extents[per_site_orbital_size][per_site_orbital_size][n_legendre][2]);
	  h5_archive["G1_LEGENDRE"] >> real_raw_legendre_data;
	  for (int row_index = 0; row_index < per_site_orbital_size; row_index++) {
	       for (int col_index = 0; col_index < per_site_orbital_size; col_index++) {
		    for (int l_index = 0; l_index < l_max; l_index++) {
			 raw_legendre_data[row_index][col_index][l_index] =
			      std::complex<double>(real_raw_legendre_data[row_index][col_index][l_index][0],
						   real_raw_legendre_data[row_index][col_index][l_index][1]);
		    }
	       }
	  }
     }
     // initialize raw_gl_matrices
     raw_gl_matrices.clear();
     for (int l_index = 0; l_index < l_max; l_index++) {
	  raw_gl_matrices.push_back(Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size));
	  for (int row_index = 0; row_index < per_site_orbital_size; row_index++) {
	       for (int col_index = 0; col_index < per_site_orbital_size; col_index++) {
		    raw_gl_matrices[l_index](row_index, col_index) =
			 raw_legendre_data[row_index][col_index][l_index];
	       }
	  }
     }
     if (sampling_type == 1) {
	  for (int l_index = 0; l_index < l_max; l_index++) {
	       // different convention for DMFT and QMC in Alps2...
	       raw_gl_matrices[l_index].transposeInPlace();
	  }
     }
     symmetrize_matrix_elements();
     // measure c_1
     for (int l_index = 0; l_index < l_max; l_index++) {
	  for (int row_index = 0; row_index < per_site_orbital_size; row_index++) {
	       for (int col_index = 0; col_index < per_site_orbital_size; col_index++) {
		    gl_values_[l_index](row_index, col_index) = raw_gl_matrices[l_index](row_index, col_index);
	       }
	  }
     }
     measured_c1 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     for (int l_index = 0; l_index < l_max; l_index += 2) {
	  // this is unnecessary, since these will be fixed later, but better for
	  // bug avoiding...
	  measured_c1 += tl_values[l_index] * gl_values_[l_index] / beta;
     }
     // measure c_2
     measured_c2 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     for (int l_index = 1; l_index < l_max; l_index += 2) {
	  measured_c2 -= tl_values[l_index] * (double)l_index * (l_index + 1.0) *
	       gl_values_[l_index] / (std::pow(beta, 2));
     }
     // Fix c_1
     // Cf paper by Boehnke et al. PRB 84, 075145 (2011)
     // Eq 10.
     for (int l_index = 0; l_index < l_max; l_index += 2) {
	  gl_values_[l_index] = raw_gl_matrices[l_index] + (
	       Eigen::MatrixXcd::Identity(per_site_orbital_size, per_site_orbital_size) - measured_c1) *
	       beta * tl_values[l_index] / tl_modulus;
     }
     // now reset measured_c1 to its target value, and compute measured_c3,
     // with corrected gl values.
     measured_c1 = Eigen::MatrixXcd::Identity(per_site_orbital_size, per_site_orbital_size);
     // measure c_3
     measured_c3 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     for (int l_index = 0; l_index < l_max; l_index += 2) {
	  double l_factor = (double)l_index;
	  measured_c3 += 0.5 * tl_values[l_index] * gl_values_[l_index] *
	       (l_factor + 2.0) * (l_factor + 1.0) * l_factor * (l_factor - 1.0) / (std::pow(beta, 3));
     }
     //display_fixed_legendre();
}

void Greensfunction::symmetrize_matrix_elements() {
	Eigen::MatrixXcd temp_matrix =
		Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
	for (int l_index = 0; l_index < l_max; l_index++) {
		temp_matrix = raw_gl_matrices[l_index];
		temp_matrix.adjointInPlace();
		raw_gl_matrices[l_index] = 0.5 * (raw_gl_matrices[l_index] + temp_matrix);
	}
}

void Greensfunction::display_fixed_legendre() {
     for (int l_index = 0; l_index < l_max; l_index++) {
	     std::cout << "GL values: " << std::endl;
	     std::cout << gl_values_[l_index] << std:: endl << std::endl;
     }
}
     
void Greensfunction::init_gf_container() {
     measured_c1 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     measured_c2 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     measured_c3 = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     tl_values.resize(l_max);
     tl_modulus = 0.0;
     for (int l_index = 0; l_index < l_max; l_index++) {
	  tl_values[l_index] = - 2.0 * std::sqrt(2.0 * l_index + 1.0);
     }
     for (int l_index = 0; l_index < l_max; l_index += 2) {
	  tl_modulus += std::abs(std::pow(tl_values[l_index], 2));
     }
     gl_values_.resize(l_max);
     bare_gf_values_.resize(n_matsubara_for_alps2);
     bare_gf_neg_values_.resize(n_matsubara_for_alps2);
     full_gf_values_.resize(n_matsubara);
     full_gf_neg_values_.resize(n_matsubara);
     for (int l_index = 0; l_index < l_max; l_index++) {
	  gl_values_[l_index] = Eigen::MatrixXcd::Zero(per_site_orbital_size, per_site_orbital_size);
     }
     // Initialize self-energy
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
	  full_gf_values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
	  full_gf_neg_values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     }
     for (int freq_index = 0; freq_index < n_matsubara_for_alps2; freq_index++) {
	  bare_gf_values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
	  bare_gf_neg_values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     }	  
}
