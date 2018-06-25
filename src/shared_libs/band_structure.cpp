#include<iostream>
#include<fstream>
#include <cmath>
#include <mpi.h>
#include "band_structure.hpp"

using namespace std;

Bandstructure::Bandstructure(const alps::params& parms, int world_rank, bool verbose)
     :world_rank_(world_rank) {     
     std::vector<Eigen::MatrixXcd> world_dispersion_;
     Eigen::VectorXd world_weights_;
     std::vector<Eigen::VectorXd> world_k_lattice_;
     int N_Qmesh = static_cast<int>(parms["bseq.N_QBSEQ"]);
     double len_qmesh = static_cast<double>(parms["bseq.MAX_QBSEQ"]);
     double min_xq_mesh = static_cast<double>(parms["bseq.MIN_XBSEQ"]);
     double min_yq_mesh = static_cast<double>(parms["bseq.MIN_YBSEQ"]);     
     n_space_sites = static_cast<int>(parms["model.space_sites"]);
     per_site_orbital_size = static_cast<int>(parms["N_ORBITALS"]);
     generate_bseq_lattice(N_Qmesh, min_xq_mesh, min_yq_mesh, len_qmesh);
     if (world_rank == 0) {
	  int n_k_points;
	  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	  cout << "Using " << world_size << " processors" << endl;
	  if (parms.exists("HOPPINGFILE")) {
	       double unique_weight;
	       n_k_points = x_dim * y_dim * z_dim;
	       unique_weight = 1.0 / double(n_k_points);
	       read_hoppings(parms, verbose);
	       init_world_containers(n_k_points);
	       world_weights_ =
		    Eigen::VectorXd::Constant(n_points_per_proc * world_size,
					      unique_weight);
	       world_dispersion_ = generate_band_from_hoppings(verbose, world_weights_, unique_weight);
	  } else if (parms.exists("DISPFILE")) {
	       double weight_sum;
	       int index;
	       n_k_points = read_nb_k_points(parms, verbose);
	       init_world_containers(n_k_points);
	       world_weights_ = Eigen::VectorXd::Zero(n_points_per_proc * world_size);
	       world_dispersion_ = read_dispersion(parms, world_weights_, verbose);
	       //manage_mpi_padding();
	       index = world_dispersion_.size();
	       for (int i = index; i < n_points_per_proc * world_size; i++)
	       {
		    world_dispersion_[i] =
			 Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
		    world_weights_(i) = 0.0;
	       }
	       weight_sum = world_weights_.sum();
	       world_weights_ /= weight_sum;
	       epsilon_bar /= weight_sum;
	       epsilon_squared_bar /= weight_sum;
	  }
	  world_k_lattice_.clear();
	  for (int k_index = 0; k_index < k_lattice_.size(); k_index++) {
	       world_k_lattice_.push_back((Eigen::VectorXd(3) << k_lattice_[k_index][0],
					  k_lattice_[k_index][1],
					   k_lattice_[k_index][2]).finished());
	  }
	  for(int k_index = k_lattice_.size(); k_index < n_points_per_proc * world_size;
	      k_index++) {
	       world_k_lattice_.push_back(Eigen::VectorXd::Zero(3));
	  }
     }
     // Broadcast the quantities defined on master only by code above.
     MPI_Bcast(&orbital_size_, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&nb_r_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&world_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&real_n_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&n_points_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
     weights_.resize(n_points_per_proc);
     dispersion_.resize(n_points_per_proc);
     proc_k_lattice_.resize(n_points_per_proc);
     for (int k_index = 0; k_index < n_points_per_proc; k_index++) {
	  dispersion_[k_index] = Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
	  proc_k_lattice_[k_index] = Eigen::VectorXd::Zero(3);
     }
     if (world_rank_ != 0) {
	  hoppings_.resize(nb_r_points);
	  r_lattice_.resize(nb_r_points);
	  for (int i = 0; i < nb_r_points; i++) {
	       hoppings_[i] = Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
	       r_lattice_[i] = Eigen::VectorXi::Zero(3);
	  }
	  epsilon_bar = Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
	  epsilon_squared_bar = Eigen::MatrixXcd::Zero(orbital_size_,
     orbital_size_);
     }
     // scatter the weights to each process
     MPI_Scatter(world_weights_.data(), n_points_per_proc, MPI_DOUBLE,
		 weights_.data(), n_points_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

     // scatter the dispersion to each process
     for (int k_index = 0; k_index < n_points_per_proc; k_index++) {
	  if (world_rank == 0) {
	       for (int proc_index = 1; proc_index < world_size; proc_index++) {		    
		    MPI_Send(world_dispersion_[proc_index * n_points_per_proc + k_index].data(),
			     world_dispersion_[proc_index * n_points_per_proc + k_index].size(),
			     MPI_DOUBLE_COMPLEX, proc_index, 0, MPI_COMM_WORLD);
	       }
	  } else {
	       MPI_Recv(dispersion_[k_index].data(),
			dispersion_[k_index].size(),
			MPI_DOUBLE_COMPLEX,
			0, 0, MPI_COMM_WORLD,
			MPI_STATUS_IGNORE);
	  }
	  if (world_rank == 0) {
	       dispersion_[k_index] = world_dispersion_[k_index];
	  }
     }
     // scatter the lattice points to each process
     for (int k_index = 0; k_index < n_points_per_proc; k_index++) {
     	  if (world_rank == 0) {
     	       for (int proc_index = 1; proc_index < world_size; proc_index++) {		    
     		    MPI_Send(world_k_lattice_[proc_index * n_points_per_proc + k_index].data(),
     			     world_k_lattice_[proc_index * n_points_per_proc + k_index].size(),
     			     MPI_DOUBLE, proc_index, 0, MPI_COMM_WORLD);
     	       }
     	  } else {
     	       MPI_Recv(proc_k_lattice_[k_index].data(),
     			proc_k_lattice_[k_index].size(),
     			MPI_DOUBLE,
     			0, 0, MPI_COMM_WORLD,
     			MPI_STATUS_IGNORE);
     	  }
     	  if (world_rank == 0) {
     	       proc_k_lattice_[k_index] = world_k_lattice_[k_index];
     	  }
     }
     // Broadcast lattice and hoppings to all processes
     // for dispersion computation
     for (int i = 0; i < nb_r_points; i++) {
	  MPI_Bcast(hoppings_[i].data(), hoppings_[i].size(),
		    MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
     }
     for (int i = 0; i < nb_r_points; i++) {
	  MPI_Bcast(r_lattice_[i].data(), r_lattice_[i].size(),
		    MPI_INT, 0, MPI_COMM_WORLD);
     }
     // Broadcast the averaged quantities to all processes
     MPI_Bcast(epsilon_bar.data(), epsilon_bar.size(),
	       MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
     MPI_Bcast(epsilon_squared_bar.data(), epsilon_squared_bar.size(),
	       MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
}

void Bandstructure::generate_bseq_lattice(int n_q_mesh, double min_xq_mesh, double min_yq_mesh, double len_q_mesh) {
     secondary_q_lattice_.clear();
     for (int k1 = 0; k1 < n_q_mesh; ++k1) {
	  for (int k2 = 0; k2 <= k1; ++k2) {
	       double kx(min_xq_mesh + len_q_mesh * double(k1) / (n_q_mesh - 1));
	       double ky(min_yq_mesh + len_q_mesh * double(k2) / (n_q_mesh - 1));
	       double kz(0.0);
	       secondary_q_lattice_.push_back(
		    (Eigen::VectorXd(3) << kx, ky, kz).finished());
	  }
     }
}

std::vector<Eigen::MatrixXcd> Bandstructure::generate_band_from_hoppings(
     bool verbose, Eigen::Ref<Eigen::VectorXd> weights, double unique_weight) {
     std::vector<Eigen::MatrixXcd> output;
     int n_k_points;
     double kx, ky, kz, phase_factor;
     // Read input file
     output.clear();
     k_lattice_.clear();
     // Calculate FT of hoppings -> dispersion
     Eigen::MatrixXcd m(orbital_size_, orbital_size_);
     for (int k1 = 0; k1 < x_dim; ++k1) {
	  for (int k2 = 0; k2 < y_dim; ++k2) {
	       for (int k3 = 0; k3 < z_dim; ++k3) {
		    kx = double(k1) / double(x_dim);
		    ky = double(k2) / double(y_dim);
		    kz = double(k3) / double(z_dim);
		    std::vector<double> tmp {kx, ky, kz};
		    k_lattice_.push_back(tmp);
		    m = get_k_basis_matrix((Eigen::VectorXd(3) << kx, ky, kz).finished());
		    //Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
		    // Check Hermiticity
		    if(((m - m.adjoint()).array().abs() > hermitian_tolerance).any()) {
			 display_hoppings();
			 cerr << "ERROR: Bandstructure: non hermitian Hamiltonian " << endl;
			 throw runtime_error("Non Hermitian Hamiltonian input !");
		    }
		    output.push_back(m);
		    epsilon_bar += m * unique_weight;
		    epsilon_squared_bar += m * m * unique_weight;
	       }
	  }
     }
     // Pad the data structures for easy MPI scattering
     int index = output.size();
     for (int i = index; i < n_points_per_proc * world_size; i++)
     {
	  output.push_back(Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_));
	  weights(i) = 0.0;
     }
     return output;
}

/*
get the k-basis hamiltonian matrix at specific k point
in_k_x, in_k_y, in_k_z, defined in r.l.u.
*/
Eigen::MatrixXcd Bandstructure::get_k_basis_matrix(
	Eigen::Ref<Eigen::VectorXd> k_point) {
     // Calculate FT of hoppings -> dispersion
     Eigen::MatrixXcd m(orbital_size_, orbital_size_);
     double phase_factor;
     m = Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
     for (int i = 0; i < nb_r_points; ++i) {
	  phase_factor = 2.0 * M_PI * (
	       r_lattice_[i](0) * k_point(0) +
	       r_lattice_[i](1) * k_point(1) +
	       r_lattice_[i](2) * k_point(2));
	  m += hoppings_[i] *
	       exp(std::complex<double>(0.0, phase_factor));
     }
     return m;
}

Eigen::MatrixXcd Bandstructure::get_local_hoppings() {
     // Calculate FT of hoppings -> dispersion
     Eigen::MatrixXcd m = hoppings_[0];
     return m;
}

Eigen::MatrixXcd Bandstructure::get_V_matrix(int direction_index) {
     // TODO the management of the hoppings needs to be
     // completely reviewed. THere is no need for an explicit listing
     // of all values like this. The hopping, and the symmetry should be listed
     // as they are in the equations in the papers, and that should be enough.
     // It will allow us to get rig of the below stuff...
     // direction_index: 1 is +x
     // 2 is -x
     // 3 is +y
     // 4 is -y
     Eigen::MatrixXcd m = hoppings_[direction_index];
     if (n_space_sites == 1) {
          for (int line_idx = 0; line_idx < orbital_size_; line_idx++) {
               m(line_idx, line_idx) = 0.5 * m(line_idx, line_idx);
          }
     } else if (n_space_sites == 2) {
          for (int line_idx = 0; line_idx < orbital_size_; line_idx++) {
               m(line_idx, line_idx) = 0.5 * m(line_idx, per_site_orbital_size + line_idx);
          }
     } else {
          cerr << "ERROR: Bandstructure: number of sites is not supported" << endl;
          throw runtime_error("number of sites is not supported !");
     }
     return m;
}

std::vector<Eigen::MatrixXcd> Bandstructure::read_dispersion(const alps::params& parms,
							     Eigen::Ref<Eigen::VectorXd> weights,
							     bool verbose) {
     std::vector<Eigen::MatrixXcd> output;
     double hr, hi;
     double weight, kx, ky, kz, eps, d;
     int n_points, dimension, index;
     std::string fname = parms["DISPFILE"];
     std::ifstream disp_file(fname.c_str());
     // re-read nb of k points and nb of orbitals from file
     disp_file >> n_points >> dimension;
     index = 0;
     k_lattice_.clear();
     while(disp_file >> weight >> kx >> ky >> kz) {
	  std::vector<double> tmp {kx, ky, kz};
	  k_lattice_.push_back(tmp);
	  weights(index) = weight;
	  Eigen::MatrixXcd m(orbital_size_, orbital_size_);
	  for(int i = 0; i < orbital_size_; ++i) {
	       for(int j = 0; j < orbital_size_; ++j) {
		    if(!(disp_file >> hr >> hi)) {
			 throw std::runtime_error("DISPFILE is corrupt!");
		    }
		    m(i, j) = std::complex<double>(hr, hi);
	       }
	  }
	  // Check Hermiticity
	  if(((m - m.adjoint()).array().abs() > hermitian_tolerance).any()) {
	       display_hoppings();
	       cerr << "ERROR: Bandstructure: non hermitian Hamiltonian " << endl;
	       throw runtime_error("Non Hermitian Hamiltonian input !");
	  }
	  output.push_back(m);
	  epsilon_bar += m * weight;
	  epsilon_squared_bar += m * m * weight;
	  index +=1;
     }
     return output;
}

int Bandstructure::read_nb_k_points(const alps::params& parms, bool verbose) {
     // Read input file
     int n_points, dimension;
     std::string fname = parms["DISPFILE"];
     std::ifstream disp_file(fname.c_str());
     // check input file exists
     if(!disp_file.good()) {
	  cerr << "ERROR: Bandstructure: problem to read DISPFILE = "
	       << parms["DISPFILE"] << endl;
	  throw runtime_error("DISPFILE is not good!");
     }
     // Read nb of k points and nb of orbitals from file
     disp_file >> n_points >> dimension;
     // Check inputs are consistent
     check_flavor_dim_consistency(parms, dimension);
     if (verbose) {
	  cout << "BANDSTRUCTURE: "
	       << "using dispersion loaded from " << parms["DISPFILE"] << std::endl
	       << dimension << " orbitals, "
	       << n_points << " k points" << endl;
     }
     return n_points;
}

void Bandstructure::read_hoppings(const alps::params& parms, bool verbose) {
     string line, sparsity;
     string ref_sparse ("SPR");
     bool is_sparse;
     int rx, ry, rz;
     int dimension;
     double hr, hi;
     hoppings_.clear();
     std::string fname = parms["HOPPINGFILE"];
     std::ifstream hopping_file(fname.c_str());
     // check input file exists
     if(!hopping_file.good()) {
	  cerr << "ERROR: Bandstructure: problem to read HOPPINGFILE = "
	       << parms["HOPPINGFILE"] << endl;
	  throw runtime_error("HOPPINGFILE is not good!");
     }
     // Read nb of R points and nb of orbitals from file
     hopping_file >> nb_r_points >> dimension >> sparsity;
     is_sparse = !(ref_sparse.compare(sparsity));
     // discard additional line contents, i.e. skip to next line
     getline(hopping_file, line);
     // Check inputs are consistent
     check_flavor_dim_consistency(parms, dimension);
     if (verbose) {
	  cout << "BARE HAMILTONIAN: "
	       << "using hoppings loaded from " << parms["HOPPINGFILE"] << std::endl
	       << orbital_size_ << " flavors, "
	       << nb_r_points << " R points" << endl
	       << "Sparse input format: " << is_sparse << endl;
     }
     // Read hoppings
     if (is_sparse) {
	  string cur_line, dummy;
	  string point_delimiter("###");
	  int line_idx, col_idx;
	  int r_pt_index = -1;
	  hoppings_.resize(nb_r_points);
	  for (int r_pt_index = 0; r_pt_index < hoppings_.size(); r_pt_index++)
	  {
	       hoppings_[r_pt_index] = Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
	  }
	  while (getline(hopping_file, cur_line)) {
	       if (hopping_file.eof()) {
		    // This was the least bit of info.
		    // Exit the loop.
		    break;
	       }	       
	       if (cur_line.compare(0, 3, point_delimiter) == 0) {
		    // We have a new point coming up!
		    // Count it
		    r_pt_index++;
		    // If we have already all the points,
		    // this means we have hit the end of file marker.
		    if (r_pt_index >= nb_r_points) {
			 break;
		    }
		    // Otherwise, handle the new point
		    getline(hopping_file, cur_line);
		    stringstream cur_input;
		    cur_input << cur_line;
		    cur_input >> rx >> ry >> rz;
		     r_lattice_.push_back((Eigen::VectorXi(3)
					   << rx, ry, rz).finished());
	       } else {
		    // Feed the new hopping matrix
		    // until we hit EOF or a new point.
		    stringstream cur_input;
		    cur_input << cur_line;
		    cur_input >> line_idx >> col_idx >> hr >> hi;
		    hoppings_[r_pt_index](line_idx - 1, col_idx - 1) = complex<double>(hr, hi);
	       }
	  }
     } else {
	  throw std::runtime_error("non sparse not supported anymore in " +
				   std::string(__FUNCTION__));
	  for (int i = 0; i < nb_r_points; i++) {
	       hopping_file >> rx >> ry >> rz;
	       getline(hopping_file, line);	     
	       r_lattice_.push_back((Eigen::VectorXi(3)
				     << rx, ry, rz).finished());
	       Eigen::MatrixXcd m(orbital_size_, orbital_size_);
	       for(int j = 0; j < orbital_size_; ++j) {
		    for(int k = 0; k < orbital_size_; ++k) {
			 if(!(hopping_file >> hr >> hi)) {
			      throw std::runtime_error("DISPFILE is corrupt!");
			 }
			 m(j, k) = complex<double>(hr, hi);
		    }
	       }
	       hoppings_.push_back(m);
	  }
     }
}

void Bandstructure::init_world_containers(int n_points) {
     real_n_points = n_points;
     n_points_per_proc = n_points / world_size;
     if (n_points % world_size > 0) {
	  n_points_per_proc += 1;
     }
     epsilon_bar = Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
     epsilon_squared_bar = Eigen::MatrixXcd::Zero(orbital_size_, orbital_size_);
}

void Bandstructure::check_flavor_dim_consistency(const alps::params& parms,
						 int dimension) {
     int n_flavors = per_site_orbital_size * n_space_sites;
     if(dimension != n_flavors) {
	  cerr << "ERROR: Bandstructure: FLAVORS from parameter file "
	       "differs from the number of bands available in DISPFILE = "
	       << parms["DISPFILE"] << endl;
	  cerr << "n_flavors: " << n_flavors << endl;
	  cerr << "dimension " << dimension << endl; 
	  throw runtime_error("Parameter conflict in DISPFILE !");
     } else {
	  orbital_size_ = static_cast<int>(dimension);
     }
}

void Bandstructure::compute_bare_dos(double chemical_potential) {
     if (world_rank_ == 0) {
	  cout << "***********************************************" << endl;
	  cout << "** SPECTRAL  FUNCTION CALCULATION      ********" << endl;
	  cout << "***********************************************" << endl << endl;
     }
     int k_min(0);
     int k_max = get_lattice_size();
     std::complex<double> omega;
     double delta_omega = freq_cutoff / static_cast<double>(nb_freq_points);
     Eigen::MatrixXcd temp;
     Eigen::VectorXcd omega_plus_mu;
     bare_dos.clear();
     world_bare_dos.clear();
     for (int freq_index = 0;
	  freq_index < 2 * nb_freq_points + 1; freq_index++) {
	  bare_dos.push_back(Eigen::VectorXd::Zero(orbital_size_));
	  world_bare_dos.push_back(Eigen::VectorXd::Zero(orbital_size_));
	  omega = (freq_index - nb_freq_points) * delta_omega + infinitesimal;
	  omega_plus_mu = Eigen::VectorXcd::Constant(orbital_size_, omega + chemical_potential);
	  for (int k_index = k_min; k_index < k_max; k_index++) {
	       if (abs(weights_(k_index)) < 1e-6) {
		    continue;
	       }	       
	       temp = -dispersion_[k_index];
	       temp.diagonal() += omega_plus_mu;
	       bare_dos.back() -= weights_(k_index) *
		    temp.inverse().diagonal().imag() / M_PI;
	  }
	  MPI_Allreduce(bare_dos[freq_index].data(),
			world_bare_dos[freq_index].data(),
			world_bare_dos[freq_index].size(),
			MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
     }
     if (world_rank_ == 0) {
	  cout << "***********   DONE      ********" << endl << endl;
     }
}

double Bandstructure::get_weight_sum(int k_min, int k_max) {
     int segment_size(k_max - k_min);
     return weights_.segment(k_min, segment_size).sum();
}

double Bandstructure::get_weight(int k_index) { return weights_(k_index); }

int Bandstructure::get_lattice_size() { return dispersion_.size(); }

int Bandstructure::get_orbital_size() { return orbital_size_; }

Eigen::MatrixXcd Bandstructure::get_epsilon_bar() { return epsilon_bar; }

Eigen::MatrixXcd Bandstructure::get_epsilon_squared_bar() {
	return epsilon_squared_bar;
}

Eigen::VectorXd Bandstructure::get_k_plus_q_point(int k_index, int q_index) {
     return proc_k_lattice_[k_index] + secondary_q_lattice_[q_index];
}

void Bandstructure::dump_bare_dos() {
     if (world_rank_ == 0) {
	  double delta_omega = freq_cutoff / static_cast<double>(nb_freq_points);
	  std::ofstream out(bare_dos_dump_name);
	  out << fixed << setprecision(output_precision);	  
	  for (int orb_index = 0; orb_index < orbital_size_; orb_index++) {
	       out << "#  " << orb_index + 1 << endl;
	       for (int freq_index = 0; freq_index < bare_dos.size(); freq_index++) {
		    out << (static_cast<double>(freq_index) -  static_cast<double>(nb_freq_points))
			 * delta_omega << "  " << world_bare_dos[freq_index](orb_index) << endl;
	       }
	       out << endl;
	  }
	  out.close();
     }
}

void Bandstructure::display_hoppings() {
     if (world_rank_ == 0) {
	  for (int r_index = 0; r_index < hoppings_.size(); r_index++) {
	       cout << "R point number " << r_index << endl << endl;
	       cout << hoppings_[r_index] << endl << endl;
	  }
     }
}

// Consider MPI_file style I/O for this and other quantities.
void Bandstructure::dump_hamilt(const alps::params& parms) {
     if (world_rank_ == 0) {
	  std::ofstream out;
	  if (parms.exists("DISPFILE")) {
	       std::string fname = parms["DISPFILE"];
	       out.open(fname.c_str(), std::ofstream::out);
	       cout << "dumping to DISPFILE defined as  "
		    << parms["DISPFILE"] << endl;
	  } else {
	       out.open(hamiltonian_dump_name, std::ofstream::out);
	       cout << "DISPFILE not defined dumping to  "
		    << hamiltonian_dump_name << endl;
	  }
	  out.precision(output_precision);
	  out << fixed << setprecision(output_precision);
	  out << dispersion_.size() << "   " << orbital_size_ << endl;
	  // Print out weight and k point coordinates
	  for (int k_index = 0; k_index < dispersion_.size(); k_index++) {
	       out << weights_[k_index];
	       for (auto const& c : k_lattice_[k_index]) {
		    out << "   " << c;
	       }
	       out << endl;
	       for (int i = 0; i < dispersion_[k_index].rows(); i++) {
		    for (int j = 0; j < dispersion_[k_index].cols(); j++) {
			 out << real(dispersion_[k_index](i, j)) << "   " <<
			      imag(dispersion_[k_index](i, j)) << endl;
		    }
	       }
	  }
	  out.close();
     }
}

int Bandstructure::get_world_size() { return world_size; }
int Bandstructure::get_n_points_per_proc() { return n_points_per_proc; }
int Bandstructure::get_real_n_points() {return real_n_points; }

const complex<double> Bandstructure::infinitesimal = complex<double>(0.0, 0.005);
const int Bandstructure::output_precision = 13;
const int Bandstructure::x_dim = 55;
const int Bandstructure::y_dim = 55;
const int Bandstructure::z_dim = 1;
const double Bandstructure::freq_cutoff = 10.0;
const double Bandstructure::hermitian_tolerance = 1.0e-5;
const string Bandstructure::bare_dos_dump_name = "c_aw.dmft";
const string Bandstructure::hamiltonian_dump_name = "hamilt.dump";
const int Bandstructure::nb_freq_points = 500;
