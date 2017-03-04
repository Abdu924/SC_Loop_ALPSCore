#ifndef GREENS_FUNCTION__
#define GREENS_FUNCTION__
#include <boost/multi_array.hpp>
#include <Eigen/Dense>
#include<vector>
#include<iostream>
#include <complex>
#include <cmath>
#include <alps/params.hpp>
#include "chemical_potential.hpp"

using namespace std;

class Greensfunction {
	/*
	 * class providing interface for the Green's function,
	 * but not much more than an accessor to Alps3 data
	 */
public:
	Greensfunction(const alps::params &parms, int world_rank,
		       alps::hdf5::archive &h5_archive);
     
	virtual ~Greensfunction() {}
     
protected:
     void read_single_site_full_gf(alps::hdf5::archive &h5_archive);
     void read_bare_gf();
     void basic_init(const alps::params &parms);
     void init_gf_container();
     
     Eigen::VectorXcd matsubara_frequencies_;
     size_t n_blocks;
     size_t n_sites;
     size_t per_site_orbital_size;
     size_t tot_orbital_size;
     double beta;
     size_t n_matsubara_freqs;
     int ref_site_index;
     
     std::vector<Eigen::MatrixXcd> bare_gf_values_;
     std::vector<Eigen::MatrixXcd> full_gf_values_;
private:
     int world_rank_;
};

#endif //GREENS_FUNCTION__
