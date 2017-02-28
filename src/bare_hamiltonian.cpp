#include<iostream>
#include <cmath>
#include <mpi.h>
#include "bare_hamiltonian.hpp"

using namespace std;

BareHamiltonian::BareHamiltonian(const alps::params& parms,
				 int world_rank, bool verbose)
     :world_rank_(world_rank) {
	double kx, ky, kz, phase_factor, unique_weight;
	n_flavors = static_cast<size_t>(parms.value_or_default("N_ORBITALS", 2)) *
		static_cast<size_t>(parms.value_or_default("N_SITES", 1));     
	if (world_rank_ == 0) {
		// Read input file
		read_hoppings(parms, verbose);
		unique_weight = 1.0 / (x_dim * y_dim * z_dim);	  
		// Calculate FT of hoppings -> dispersion
		Eigen::MatrixXcd m(n_flavors, n_flavors);
		for (int k1 = 0; k1 < x_dim; ++k1) {
			for (int k2 = 0; k2 < y_dim; ++k2) {
				for (int k3 = 0; k3 < z_dim; ++k3) {
					kx = double(k1) / double(x_dim);
					ky = double(k2) / double(y_dim);
					kz = double(k3) / double(z_dim);
					std::vector<double> tmp {kx, ky, kz};
					k_lattice_.push_back(tmp);
					weights_.push_back(unique_weight);
					m = Eigen::MatrixXcd::Zero(n_flavors, n_flavors);
					for (int i = 0; i < nb_r_points; ++i) {
						phase_factor = 2.0 * M_PI * (
							r_lattice_[i][0] * kx
							+ r_lattice_[i][1] * ky
							+ r_lattice_[i][2] * kz);
						m += hoppings_[i] *
						     exp(std::complex<double>(0.0, phase_factor));
					}
					dispersion_.push_back(m);
				}
			}
		}
	}
}

void BareHamiltonian::read_hoppings(const alps::params& parms, bool verbose) {
     string line;
     int rx, ry, rz;
     int dimension;
     double hr, hi;
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     cout << " what am I doing here?" << endl;
     cout << "Using " << world_size << " processors" << endl;
     std::ifstream hopping_file(parms["HOPPINGFILE"].c_str());
     // check input file exists
     if(!hopping_file.good()) {
	  cerr << "ERROR: Bandstructure: problem to read HOPPINGFILE = "
	       << parms["HOPPINGFILE"] << endl;
	  throw runtime_error("HOPPINGFILE is not good!");
     }
     // Read nb of R points and nb of orbitals from file
     hopping_file >> nb_r_points >> dimension;
     // discard additional line contents, i.e. skip to next line
     getline(hopping_file, line);
     // Check inputs are consistent
     if(dimension != n_flavors) {
	  cerr<<"ERROR: Bandstructure: FLAVORS from parameter file "
	       "differs from the number of bands available in HOPPINGFILE = "
	      << parms["HOPPINGFILE"] << endl;
	  throw runtime_error("Parameter conflict in HOPPINGFILE !");
     }
     if (verbose) {
	  cout << "BARE HAMILTONIAN: "
	       << "using hoppings loaded from " << parms["HOPPINGFILE"] << std::endl
	       << dimension << " orbitals, "
	       << nb_r_points << " R points" << endl;
     }
     // Read hoppings
     for (int i = 0; i < nb_r_points; i++) {
	     hopping_file >> rx >> ry >> rz;
	     getline(hopping_file, line);	     
	     std::vector<int> tmp {rx, ry, rz};
	     r_lattice_.push_back(tmp);
	     Eigen::MatrixXcd m(dimension, dimension);
	     for(size_t j = 0; j < n_flavors; ++j) {
		     for(size_t k = 0; k < n_flavors; ++k) {
			     if(!(hopping_file >> hr >> hi)) {
				     throw std::runtime_error("DISPFILE is corrupt!");
			     }
			     m(j, k) = hr + hi * std::complex<double>(0,1);
		     }
	     }
	     hoppings_.push_back(m);
     }
}

void BareHamiltonian::dump_hamilt() {
	if (world_rank_ == 0) {
		ofstream out(hamiltonian_dump_name);
		out.precision(output_precision);
		out << fixed << setprecision(output_precision);
		out << dispersion_.size() << "   " << n_flavors << endl;
		// Print out weight and k point coordinates
		for (size_t k_index = 0; k_index < dispersion_.size(); k_index++) {
			out << weights_[k_index];
			for (auto const& c : k_lattice_[k_index]) { out << "   " << c; }
			out << endl;
			for (size_t i = 0; i < dispersion_[k_index].rows(); i++) {
				for (size_t j = 0; j < dispersion_[k_index].cols(); j++) {
					out << real(dispersion_[k_index](i, j)) << "   " <<
						imag(dispersion_[k_index](i, j)) << endl;
				}
			}
		}
		out.close();
	}
}

int BareHamiltonian::get_nb_k_points() { return k_lattice_.size(); }

int BareHamiltonian::get_n_flavors() { return n_flavors; } 

const size_t BareHamiltonian::output_precision = 16;
const size_t BareHamiltonian::x_dim = 55;
const size_t BareHamiltonian::y_dim = 55;
const size_t BareHamiltonian::z_dim = 1;
const string BareHamiltonian::hamiltonian_dump_name = "hamilt.dump";
