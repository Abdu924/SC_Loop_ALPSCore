#pragma once
#include "gf_base.hpp"
#include "greens_function.hpp"

using namespace std;

class Selfenergy: public GfBase {
     /*
      * class providing interface for the self-energy
      */
public:
     Selfenergy(const alps::params &parms, int world_rank,
		alps::hdf5::archive h5_archive,
		string h5_group_name, bool verbose=false);
     Selfenergy(const alps::params &parms, int world_rank,
		boost::shared_ptr<Chemicalpotential> chempot,
		int ref_site_index,
		alps::hdf5::archive h5_archive, int input_type,
		bool verbose=false);
     Selfenergy(const alps::params &parms, int world_rank,
		boost::shared_ptr<Chemicalpotential> chempot,
		int ref_site_index, alps::hdf5::archive h5_archive, 
		boost::shared_ptr<Greensfunction> greens_function);
     void display_asymptotics();
     double get_beta();
     size_t get_n_matsubara_freqs();
     std::complex<double> get_matsubara_frequency(size_t n);
     size_t get_per_site_orbital_size();
     Eigen::MatrixXcd get_sigma_0();
     Eigen::MatrixXcd get_sigma_1();
     double get_order2_partial_sum();
     int get_n_sites();
     bool get_enforce_real() {return enforce_real;};
     Eigen::VectorXcd get_matsubara_frequencies();
     /* for each frequency, 
      * we have a matrix of dimension 
      * tot_orbital_size x tot_orbital_size.
      */
     std::vector<Eigen::MatrixXcd> values_;
     std::vector<Eigen::MatrixXcd> neg_values_;
     bool get_is_nil_sigma() {return is_nil_sigma;};
     void hdf5_dump(alps::hdf5::archive h5_archive, string h5_group_name);
     void hdf5_dump_tail(alps::hdf5::archive h5_archive,
			 string h5_group_name, int ref_site_index, int tail_order);
     void apply_linear_combination(boost::shared_ptr<Selfenergy> const &old_sigma,
				   double alpha);
     
     virtual ~Selfenergy() {}
     
protected:
     void compute_order2_partial_sum();
     void read_input_sigma(const alps::params &parms,
			   alps::hdf5::archive h5_archive,
			   string h5_group_name);
     void basic_init(const alps::params &parms, bool verbose=false);
     void read_qmc_sigma(int ref_site_index, alps::hdf5::archive h5_archive);
     void read_qmc_sigma(int ref_site_index, boost::shared_ptr<Greensfunction> greens_function);
     void init_sigma_container();
     void get_single_site_hdf5_data(size_t site_index,
				    alps::hdf5::archive h5_archive,
				    string rootpath);
     Eigen::MatrixXcd get_single_site_hdf5_asymptotics(size_t site_index,
						       alps::hdf5::archive h5_archive,
						       string rootpath,
						       int asymptotic_order);
     void get_qmc_single_site_hdf5_data(size_t site_index,
					alps::hdf5::archive h5_archive, string rootpath);

     Eigen::MatrixXcd Sigma_0_;
     Eigen::MatrixXcd Sigma_1_;	
     static const size_t tail_fit_length;
     bool is_nil_sigma;
     bool is_diagonal;
     size_t n_matsubara_freqs;
     double order2_partial_sum_;
	  
private:
     bool enforce_real;
     int input_type;
     bool is_alps3;
     bool is_analytic_tail;
     int matsubara_tail_estimate_region;
     
     void run_dyson_equation(int ref_site_index,
			     boost::shared_ptr<Greensfunction> greens_function);
     void symmetrize_tail_params(int ref_site_index);
     void symmetrize_qmc_sigma(int ref_site_index);
     void symmetrize_matrix_elements(int ref_site_index);
     void symmetrize_sites(int ref_site_index);
     void read_symmetry_definition(std::string symmetry_file);
     std::vector<double> get_u_elements(const alps::params &parms);
     void log_sigma_tails(int ref_site_index);
     void feed_tail_params(int ref_site_index,
			   const alps::params &parms,
			   alps::hdf5::archive &h5_archive);
     void compute_tail_coeffs(boost::shared_ptr<Greensfunction> greens_function,
			      int ref_site_index);
     void compute_tail_coeffs(int ref_site_index);
     void fit_tails(int ref_site_index);
     void compute_qmc_tail(int ref_site_index);
     void append_qmc_tail(int ref_site_index, const alps::params &parms);
     void sanity_check(const alps::params &parms);
     
     static const std::string density_density_result_name;
};
