#pragma once
#include <boost/multi_array.hpp>
#include <Eigen/Dense>
#include<vector>
#include<iostream>
#include <complex>
#include <cmath>
#include <alps/params.hpp>
#include "chemical_potential.hpp"
#include "legendre.hpp"
#include "gf_base.hpp"

using namespace std;

class Greensfunction: public GfBase {
     /*
      * class providing interface for the Green's function,
      * but not much more than an accessor to Alps3 data
      */
public:
     Greensfunction(const alps::params &parms, int world_rank,
		    int sampling_type,  alps::hdf5::archive &h5_archive);
     Eigen::MatrixXcd get_dyson_result(int freq_index, bool is_negative);
     //void generate_t_coeffs(alps::hdf5::archive &h5_archive);
     
     virtual ~Greensfunction() {}
     
protected:
     void read_single_site_raw_legendre(alps::hdf5::archive &h5_archive,
					int site_index=0);
     void fix_moments();
     void read_single_site_full_gf_matsubara(alps::hdf5::archive &h5_archive,
					     int site_index=0);
     void read_bare_gf();
     void basic_init(const alps::params &parms);
     void init_gf_container();
     void generate_data(alps::hdf5::archive &h5_archive);
     void get_matsubara_from_legendre(int site_index=0);
     void display_fixed_legendre();
     void read_t_coeffs(alps::hdf5::archive &h5_archive);
     void symmetrize_matrix_elements();
     std::complex<double> get_t_coeff(int n, int l);
 
     int n_matsubara_for_alps2;
     int n_legendre;
     int l_max;
     int ref_site_index;
     int sampling_type;
     bool fix_sigma_0;
     bool fix_sigma_1;
	
     std::vector<Eigen::MatrixXcd > raw_gl_matrices;
     Eigen::MatrixXcd full_t_set;
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
