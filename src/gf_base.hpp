#pragma once
#include <Eigen/Dense>
#include<vector>
#include<iostream>
#include <complex>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <alps/params.hpp>
#include <boost/multi_array.hpp>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>
#include <alps/hdf5/pointer.hpp>
#include "chemical_potential.hpp"

using namespace std;

class GfBase {
public:
     GfBase(const alps::params &parms, int world_rank);
     GfBase(const alps::params &parms, int world_rank,
	    boost::shared_ptr<Chemicalpotential> chempot,
	    boost::shared_ptr<Bandstructure> const &lattice_bs);
     
     virtual ~GfBase() {}

     void get_interaction_matrix(int ref_site_index, const alps::params &parms);
     void get_density_density_correl(int ref_site_index, const alps::params &parms,
						 alps::hdf5::archive &h5_archive);
     void get_a_dagger_b(int ref_site_index, const alps::params &parms,
			 alps::hdf5::archive &h5_archive);
     void read_params(const alps::params &parms);
     void get_target_c2(int ref_site_index);
     void get_target_c3(int ref_site_index);
     void get_new_target_c3(int ref_site_index);
     
     Eigen::VectorXcd matsubara_frequencies_;
     size_t n_blocks;
     size_t n_sites;
     size_t per_site_orbital_size;
     size_t tot_orbital_size;
     double beta;
     std::vector<std::vector<size_t> > blocks;
     Eigen::MatrixXcd get_measured_c2();
     Eigen::MatrixXcd get_measured_c3();

protected:
     int world_rank_;
     Eigen::MatrixXcd interaction_matrix;
     Eigen::MatrixXcd site_symmetry_matrix;
     Eigen::MatrixXcd a_dagger_b;
     Eigen::MatrixXcd density_density_correl;
     Eigen::MatrixXcd measured_c1, measured_c2, measured_c3;
     Eigen::MatrixXcd target_c1, target_c2, target_c3;
     std::vector<Eigen::MatrixXcd> qmc_tail;
     boost::shared_ptr<Chemicalpotential> chempot_;
     boost::shared_ptr<Bandstructure> lattice_bs_;
     int n_matsubara;

     void feed_tail_params(int ref_site_index,
			   const alps::params &parms,
			   alps::hdf5::archive &h5_archive);     
};
