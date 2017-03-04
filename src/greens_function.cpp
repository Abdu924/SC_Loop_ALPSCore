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


using namespace std;

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
     n_matsubara_freqs = static_cast<size_t>(parms["N_MATSUBARA"]);
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
			      freq_index < n_matsubara_freqs; freq_index++) {
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

void Greensfunction::read_single_site_full_gf(alps::hdf5::archive &h5_archive) {

     typedef boost::multi_array<complex<double>, 4> array_type;
     array_type test(boost::extents[4][4][40][2]);
     h5_archive["G1_LEGENDRE"] >> test;
     cout << "test element :" << test[0][0][0][0] << endl;
     //typedef array_type::index index; 
     
     typedef boost::multi_array<complex<double> , 3> cplx_array_type;
     typedef cplx_array_type::index_range range;

     cplx_array_type raw_full_gf(
	  boost::extents[n_matsubara_freqs][per_site_orbital_size][per_site_orbital_size]);
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
     bare_gf_values_.resize(n_matsubara_freqs);
     //initialize self-energy
     for (int freq_index = 0; freq_index < n_matsubara_freqs; freq_index++) {
	  bare_gf_values_[freq_index] =
	       Eigen::MatrixXcd::Zero(tot_orbital_size, tot_orbital_size);
     }
}
