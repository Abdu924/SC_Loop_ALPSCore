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
#include "legendre.hpp"

using namespace std;

class Greensfunction {
     /*
      * class providing interface for the Green's function,
      * but not much more than an accessor to Alps3 data
      */
public:
     Greensfunction(const alps::params &parms, int world_rank,
		    int sampling_type,  alps::hdf5::archive &h5_archive);
     Eigen::MatrixXcd get_dyson_result(int freq_index, bool is_negative);
     Eigen::MatrixXcd get_measured_c2();
     Eigen::MatrixXcd get_measured_c3();
     
     virtual ~Greensfunction() {}
     
protected:
     void read_single_site_legendre(alps::hdf5::archive &h5_archive,
				    int site_index=0);
     void read_single_site_full_gf_matsubara(alps::hdf5::archive &h5_archive,
					     int site_index=0);
     void read_bare_gf();
     void basic_init(const alps::params &parms);
     void init_gf_container();
     void generate_data(alps::hdf5::archive &h5_archive);
     void get_matsubara_from_legendre(int site_index=0);
     void display_fixed_legendre();

     Eigen::VectorXcd matsubara_frequencies_;
     size_t n_blocks;
     size_t n_sites;
     size_t per_site_orbital_size;
     size_t tot_orbital_size;
     double beta;
     int n_matsubara;
     int n_matsubara_for_alps2;
     int n_legendre;
     int l_max;
     int ref_site_index;
     int sampling_type;

     Eigen::MatrixXcd measured_c1, measured_c2, measured_c3;
     std::vector<Eigen::MatrixXcd> gl_values_;
     std::vector<Eigen::MatrixXcd> bare_gf_values_;
     std::vector<Eigen::MatrixXcd> bare_gf_neg_values_;
     std::vector<Eigen::MatrixXcd> full_gf_values_;
     std::vector<Eigen::MatrixXcd> full_gf_neg_values_;
     std::vector<std::complex<double> > tl_values;
     std::complex<double> tl_modulus;
private:
     int world_rank_;
};

#endif //GREENS_FUNCTION__
