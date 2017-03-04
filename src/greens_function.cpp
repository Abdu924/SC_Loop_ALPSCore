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
#include <math.h>
#include <gsl/gsl_sf_bessel.h>

using namespace std;

std::complex<double> legendre_coeff(int n, int l) {
     // transformation matrix from Legendre to Matsubara basis
     std::complex<double> i_c(0., 1.);
     //std::complex<double>  testzzz = boost::math::cyl_bessel_j(l + 0.5, (n + 0.5) * M_PI);
     double testzzz = jn(l, static_cast<double>(0.5 * (2 * n + 1) * M_PI));
     return (sqrt(2 * l + 1) / sqrt(2 * n + 1)) *
	  exp(i_c * (n + 0.5) * M_PI) * pow(i_c, l);
     // return (std::sqrt(2 * l + 1) / std::sqrt(2 * n + 1)) *
     // 	  std::exp(i_c * (n + 0.5) * M_PI) * std::pow(i_c, l) *
     // 	  boost::math::cyl_bessel_j(l + 0.5, (n + 0.5) * M_PI);
}


Greensfunction::Greensfunction(const alps::params &parms, int world_rank,
			       alps::hdf5::archive &h5_archive)
     :world_rank_(world_rank) {   
     basic_init(parms);
     read_bare_gf();
     read_single_site_full_gf(h5_archive);
}

void Greensfunction::basic_init(const alps::params &parms) {
     ref_site_index = 0;
     n_blocks = static_cast<size_t>(parms["N_BLOCKS"]);
     n_sites = parms.exists("N_SITES") ?
	  static_cast<size_t>(parms["N_SITES"]) : 1;
     per_site_orbital_size = parms.exists("N_ORBITALS") ?
	  static_cast<size_t>(parms["N_ORBITALS"]) : 2;
     tot_orbital_size = per_site_orbital_size * n_sites;
     n_matsubara = static_cast<int>(parms["N_MATSUBARA"]);
     n_legendre = static_cast<int>(parms["N_LEGENDRE"]);
     beta = static_cast<double>(parms["BETA"]);
     init_gf_container();
}

void Greensfunction::read_bare_gf() {
     std::string matsubara_bare_gf_dump_name = "c_gw";
     std::ifstream infile(matsubara_bare_gf_dump_name.c_str());
     if(!infile.good()) {
	  std::cerr << "Could not find file " << matsubara_bare_gf_dump_name << endl;
	  throw std::runtime_error("pb reading bare GF in txt format");
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
			      bare_gf_values_[freq_index].block(site_index * per_site_orbital_size,
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

void Greensfunction::read_single_site_full_gf(alps::hdf5::archive &h5_archive, int site_index) {
     typedef boost::multi_array<double, 4> array_type;
     typedef boost::multi_array<complex<double> , 3> cplx_array_type;
     typedef cplx_array_type::index_range range;

     array_type legendre_data(boost::extents[4][4][40][2]);
     h5_archive["G1_LEGENDRE"] >> legendre_data;
     //typedef array_type::index index; 
     
     /*
      * Initialize LegendreTransformer
      */
     LegendreTransformer legendre_transformer(n_matsubara, n_legendre);
     const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &Tnl(legendre_transformer.Tnl());
     Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>
	  tmp_mat(n_legendre, 1), tmp_mat2(n_matsubara, 1);

     for (int flavor = 0; flavor < per_site_orbital_size; ++flavor) {
	  for (int flavor2 = 0; flavor2 < per_site_orbital_size; ++flavor2) {
	       for (int il = 0; il < n_legendre; ++il) {
		    tmp_mat(il, 0) = std::complex<double>(legendre_data[flavor][flavor2][il][0],
							  legendre_data[flavor][flavor2][il][1]);
	       }
	       tmp_mat2 = Tnl * tmp_mat;
	       for (int freq_index = 0; freq_index < n_matsubara; ++freq_index) {
		    full_gf_values_[freq_index].block(site_index * per_site_orbital_size,
						      site_index * per_site_orbital_size,
						      per_site_orbital_size,
						      per_site_orbital_size)(flavor, flavor2) =
			 tmp_mat2(freq_index, 0);
	       }
	  }
     }

     cplx_array_type raw_full_gf(
	  boost::extents[n_matsubara][per_site_orbital_size][per_site_orbital_size]);
     h5_archive["/gf/data"] >> raw_full_gf;

     cplx_array_type::array_view<3>::type myview =
	  raw_full_gf[ boost::indices[range(0,2)][range(1,3)][range(0,4,2)] ];

     // std::vector<std::complex<double> > temp_data;
     // temp_data.resize(2 * n_matsubara_freqs);
     // size_t cur_index = 0;
     // for (size_t orb1 = 0; orb1 < per_site_orbital_size; orb1++) {
     // 	  for (size_t orb2 = 0; orb2 < per_site_orbital_size; orb2++) {
     // 	       cur_index = blocks[block_index][orb2] *  blocks[block_index].size() +
     // 		    blocks[block_index][orb1];
     // 	       std::stringstream orbital_path;
     // 	       orbital_path << rootpath << "/" +
     // 		    boost::lexical_cast<std::string>(block_index) +
     // 		    "/" + boost::lexical_cast<std::string>(cur_index) + "/mean/value";
     // 	       h5_archive >> alps::make_pvp(orbital_path.str(),
     // 					    temp_data);
     // 	       for (size_t freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
     // 		    values_[freq_index].block(site_index * per_site_orbital_size,
     // 					      site_index * per_site_orbital_size,
     // 					      per_site_orbital_size,
     // 					      per_site_orbital_size)
     // 			 (blocks[block_index][orb1],
     // 			  blocks[block_index][orb2]) =
     // 			 temp_data[freq_index];
     // 		    neg_values_[freq_index].block(site_index * per_site_orbital_size,
     // 						  site_index * per_site_orbital_size,
     // 						  per_site_orbital_size,
     // 						  per_site_orbital_size)
     // 			 (blocks[block_index][orb1],
     // 			  blocks[block_index][orb2]) =
     // 			 temp_data[n_matsubara_freqs + freq_index];
     // 	       }
     // 	  }
     // }
}

void Greensfunction::init_gf_container() {
     bare_gf_values_.resize(n_matsubara);
     full_gf_values_.resize(n_matsubara);
     //initialize self-energy
     for (int freq_index = 0; freq_index < n_matsubara; freq_index++) {
	  bare_gf_values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
	  full_gf_values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     }
}
