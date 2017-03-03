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
		       alps::hdf5::archive &h5_archive, string h5_group_name,
		       bool verbose=false);
     
	virtual ~Greensfunction() {}
     
protected:
	boost::multi_array<std::complex<double>, 3> raw_gf_data;//(boost::extents[n_flavors][n_flavors][n_legendre]);
private:
	int world_rank_;
};

#endif //GREENS_FUNCTION__
