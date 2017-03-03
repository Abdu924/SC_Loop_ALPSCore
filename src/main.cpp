//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/filesystem.hpp>
#include "band_structure.hpp"
#include "self_energy.hpp"
#include "dmft_model.hpp"
#include "bare_hamiltonian.hpp"
#include "hybridization_function.hpp"
#include <fstream>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <boost/timer/timer.hpp>
#include <boost/program_options.hpp>
#include <mpi.h>
#include "chemical_potential.hpp"
#include <alps/mc/mcbase.hpp>

using namespace std;
namespace po = boost::program_options;

void init(int world_rank, int computation_type,
	  string &output_file_name, string &backup_file_name) {
     if (computation_type == 0) {
	  output_file_name = "c_dmft.out";
     } else if (computation_type == 1) {
	  output_file_name = "c_mix.out";
     } else if (computation_type == 2) {
	  output_file_name = "c_dump_hamilt.out";
     } else if (computation_type == 3) {
	  output_file_name = "c_debug.out";
     }
     else if (computation_type == 4) {
	  output_file_name = "c_bubble.out";
     }
     backup_file_name = output_file_name + ".old";
     if (world_rank == 0) {
	     if (boost::filesystem::exists(output_file_name)) {
		     if (boost::filesystem::is_regular_file(output_file_name)) {
			     boost::filesystem::copy_file(output_file_name, backup_file_name,
							  boost::filesystem::copy_option::overwrite_if_exists);
			     //cout << p << " size is " << boost::filesystem::file_size(p) << endl;
		     }
	     }
	     auto dummy1 = freopen(output_file_name.c_str(), "w", stdout );
	     auto dummy2 = freopen("error.txt", "w", stderr);
	     cout << "Using Boost "
		  << BOOST_VERSION / 100000     << "."  // major version
		  << BOOST_VERSION / 100 % 1000 << "."  // minor version
		  << BOOST_VERSION % 100                // patch level
		  << endl;
     }
}

void define_parameters(alps::params &parameters) {
     // If the parameters are restored, they exist
     if (parameters.is_restored()) {
	  return;
     }
     // Adds the parameters of the base class
     alps::mcbase::define_parameters(parameters);
     // Adds the convenience parameters (for save/load)
     // alps::define_convenience_parameters(parameters);
     // Define ct-hyb related parameters
     parameters
	  .description("hybridization expansion simulation")
	  .define<bool>("cthyb.ACCURATE_COVARIANCE", false, "TODO: UNDERSTAND WHAT THIS DOES")
	  .define<std::string>("cthyb.BASEPATH","", "path in hdf5 file to which results are stored")
	  .define<double>("BETA", "inverse temperature")
	  .define<int >("N_BLOCKS", "number of interacting blocks")
	  .define<std::string>("BLOCKS", "file name for file with blocks description")
	  .define<bool >("SIGMA_IN_HDF5", "is sigma located in a hdf5 file?")
	  .define<std::string>("SIGMA", "hdf5 object name where current self-energy is found")
	  .define<std::string>("SITE_SYMMETRY", "file name where the site symmetry is described")
	  .define<std::string>("TAIL_STYLE", "exact or normal tail management in the sc loop")
	  .define<bool>("cthyb.COMPUTE_VERTEX", false, "whether to compute the vertex functions or not.")
	  .define<std::string>("cthyb.DELTA","path for hybridization function file")
	  .define<bool>("cthyb.DELTA_IN_HDF5",false,"true if hybridization function file is in hdf5 format")
	  .define<std::string>("DISPFILE", "path for bare dispersion file")
	  .define<std::string>("HOPPINGFILE", "path for hopping description file")
	  .define<bool >("cthyb.REAL_DELTA", "if true, we force the hybridization function to be real")
	  .define<double>("N_ELECTRONS", "electronic density")
	  .define<int >("N_ORBITALS", "total number of spin-orbitals")
	  .define<double>("C_MAX", "parameter for tail adjustment of gf")
	  .define<double>("C_MIN", "parameter for tail adjustment of gf")
	  .define<double>("ALPHA", "parameter for combination of new and old self-energies")	  
	  .define<bool>("cthyb.DMFT_FRAMEWORK",false,"true if we need to tie into a dmft framework")
	  .define<bool>("cthyb.GLOBALFLIP", false, "TODO: UNDERSTAND WHAT THIS DOES.")
	  .define<double>("cthyb.J",0,"interaction value for density-density Hund's coupling term J.")
	  .define<bool>("cthyb.K_IN_HDF5",false,"set to true if retarded interaction K is stored in hdf5.")
	  .define<bool>("cthyb.MEASURE_freq",true, "measure in frequency domain")
	  .define<bool>("cthyb.MEASURE_g2w",false, "measure two-particle Green's function in frequency space")
	  .define<bool>("cthyb.MEASURE_h2w",false, "measure two-particle H Green's function in frequency space")
	  .define<bool>("cthyb.MEASURE_legendre",false, "measure legendre Green's function coefficients")
	  .define<bool>("cthyb.MEASURE_nn",false, "measure static density-density correlation functions")
	  .define<bool>("cthyb.MEASURE_nnt",false, "measure density-density correlation functions <n(0) n(t)>")
	  .define<bool>("cthyb.MEASURE_nnw",false, "measure density-density correlation functions in frequency domain")
	  .define<bool>("cthyb.MEASURE_sector_statistics",false, "measure sector statistics")
	  .define<bool>("cthyb.MEASURE_time",false, "measure in the time domain")
	  .define<double>("MU", "chemical potential / orbital energy values")
	  .define<std::string>("MU_VECTOR", "file name for file with chemical potential / orbital energy values")
	  .define<bool>("MU_IN_HDF5", false,"true if the file MU_VECTOR points to a hdf5 file")
	  .define<int >("cthyb.N_HISTOGRAM_ORDERS",200, "orders for the histograms of probability per order")
	  .define<int >("cthyb.N_LEGENDRE",0,"number of legendre coefficients")
	  .define<int >("N_MATSUBARA",40,"number of matsubara coefficients")
	  .define<int >("cthyb.N_MEAS","number of updates per measurement")
	  .define<int >("FLAVORS","number of spin-orbitals (sometimes called flavors)")
	  .define<int >("N_TAU","number of imaginary time discretization points")
	  .define<int >("cthyb.N_W",0,"number of bosonic Matsubara frequencies for the two-particle measurement (0 ... N_W)")
	  .define<int >("cthyb.N_nn",0,"number of points for the measurement of the density density correlator")
	  .define<int >("cthyb.N_w2",0,"number of fermionic frequencies for the two-particle measurement (-N_w2 ... N_w2-1)")
	  .define<std::string>("cthyb.RET_INT_K","file with the retarted interaction information. See doc for format.")
	  .define<bool>("cthyb.SPINFLIP",false,"TODO: UNDERSTAND THIS PARAMETER")
	  .define<unsigned long>("cthyb.SWEEPS","total number of Monte Carlo sweeps to be done")
	  .define<bool>("cthyb.TEXT_OUTPUT","if this is enabled, we write text files in addition to hdf5 files")
	  .define<unsigned long>("cthyb.THERMALIZATION","thermalization steps")
	  .define<double>("U","interaction value. Only specify if you are not reading an U matrix")
	  .define<double>("Uprime",0,"interaction value Uprime. Only specify if you are not reading an U matrix")
	  .define<std::string>("U_MATRIX","file name for file that contains the interaction matrix")
	  .define<bool>("UMATRIX_IN_HDF5",false,"true if we store the U_matrix as /Umatrix in a hdf5 file")
	  .define<bool>("VERBOSE",false,"how verbose the code is. true = more output")
	  .define<std::size_t>("MAX_TIME", "maximum solver runtime")
	  .define<std::string>("solver.OUTFILE_H5GF", "alps_solver_in.h5gf", "H5GF Green's function input file containing G0(omega,k) at /G0 and G0(ij, tau) at /G0_tau_rs")
	  .define<std::string>("solver.INFILE_H5GF", "alps_solver_out.h5gf","H5GF Green's function output file containing G(omega,k) at /G_omega and G(ij, tau) at /G_tau")
	  .define<int >("L", 200, "Number of Brillouin Zone points")
	  .define<double>("t", 1.0, "Hopping element: nearest neighbor")
	  .define<double>("tprime", 0.0, "Hopping element: next-nearest neighbor")
	  ;
}

bool testMatch(const boost::regex &ex, const std::string st)  {
     if (boost::regex_match(st, ex)) {
	  return true;
     } else {
	  return false;
     }
}

tuple<bool, string> get_match_from_file(const boost::regex regexp, string &search_file_name) {
     string output = "";
     bool found_elem = false;
     vector<string> matches;
     boost::char_separator<char> sep("\n");

     if (boost::filesystem::exists(search_file_name)) {
	  if (boost::filesystem::is_regular_file(search_file_name)) {
	       std::ifstream latest_log_file(search_file_name);
	       std::string text((std::istreambuf_iterator<char>(latest_log_file)),
				std::istreambuf_iterator<char>());
	       boost::tokenizer<boost::char_separator<char>> tokens(text, sep);
	       for(auto val : tokens) {
		    if (testMatch(regexp, val))  {
			 found_elem = true;
			 matches.push_back(val);
		    }
	       }
	  }
     }
     if (found_elem) {
	  output = matches.back();
     }
     return tuple<bool, string>(found_elem, output);
}

tuple<bool, double, double> get_old_mu(string &backup_file_name) {
     double max_mu_increment = 1.0;
     double min_mu_increment = 0.001;
     static const boost::regex mu_expression(".*:MU.*");
     bool found_mu = false;
     double old_mu(1.1);
     double old_derivative(1.0);
     string last_match;
     tie(found_mu, last_match) = get_match_from_file(mu_expression, backup_file_name);
     if (found_mu) {
	  string dummy;
	  std::istringstream iss(last_match);
	  iss >> dummy >> old_mu >> old_derivative;
     }
     old_derivative *= 4.0;
     old_derivative = min(max_mu_increment, max(min_mu_increment, abs(old_derivative)));
     return tuple<bool, double, double>(found_mu, old_mu, old_derivative);
}

tuple<string, int, bool> handle_command_line(po::variables_map vm, po::options_description desc) {
     int computation_type;
     bool from_alps3(false);
     if (vm.count("help")) {
	  cout << desc << "\n";
	  exit(1);
     }
     // Read hdf5 input define base_name for computation
     if (vm.count("input-file"))
     {
	  cout << "Input file is: "
	       << vm["input-file"].as<string>() << "\n";
     }
     if (vm.count("compute_delta"))
     {
	  cout << "Computing hybridization function\n";
	  computation_type = 0;
     } else if (vm.count("mix")) {
	  cout << "Running mix action\n";
	  computation_type = 1;
     } else if (vm.count("dump_hamilt")) {
	  cout << "Dumping Hamiltonian\n";
	  computation_type = 2;
     } else if (vm.count("debug")) {
	  cout << "Debug mode\n";
	  computation_type = 3;
     } else if (vm.count("compute_bubble")) {
	  cout << "Computing bubble\n";
	  computation_type = 4;
     } else if (vm.count("from_alps3")) {
	     from_alps3 = true;
	     cout << "Taking ***MATRIX RESULT*** as input format\n";
     } else {
	  cout << desc << "\n";
	  exit(1);
     }
     string input_file(vm["input-file"].as<string>());
     return tuple<string, int, bool>(input_file, computation_type, from_alps3);
}

double extract_chemical_potential(boost::shared_ptr<Bandstructure> bare_band,
				  boost::shared_ptr<Chemicalpotential> chempot) {
     double tolerance = 1.0e-6;
     std::size_t n_orbitals = chempot->n_orbitals();
     Eigen::VectorXd energies = bare_band->get_epsilon_bar().real().block(
	  0, 0, n_orbitals, n_orbitals).diagonal();
     double candidate_mu(0.0);
     double ref_candidate_mu(energies(0) + (*chempot)[0]);
     for (int i = 1; i < n_orbitals; i++) {
	  candidate_mu = energies(i) + (*chempot)[i];
	  if ((i > 0) && (abs(candidate_mu - ref_candidate_mu) > tolerance)) {
		    cout << "MU is inconsistent. Quitting. " << endl;
		    throw runtime_error("MU is inconsistent !");
	  }
     }
     return ref_candidate_mu;
}

int main(int argc, char** argv) {
	int world_size, world_rank;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	string output_file_name;
	string old_output_file_name;
	double density, derivative;
	double old_chemical_potential, new_chemical_potential,
		dn_dmu, new_dn_dmu;
	bool found_old_mu, newton_success, bisec_success;
	// kind of computation
	// 0 --- compute hybridization function
	// 1 --- compute tail of new_sigma, mix new_sigma with old_sigma,
	// and copy sigma to old_sigma.
	int computation_type;
	bool from_alps3(false);
	string tmp_file_name;
	//alps::params parms;
	cout << "try for parms" << endl;
	//alps::params parms(argc, argv);
	alps::params parms(argc, (const char**)argv, "/parameters");
	define_parameters(parms);
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("compute_delta", "Evaluate hybridization function")
		("compute_bubble", "Evaluate bubble (local and lattice)")
		("mix", "post process QMC result: mix old and new self-energy values, including tail smoothing")
		("dump_hamilt", "dump the Hamiltonian in reciprocal space. output file name taken from params")
		("debug", "Enter debug mode")
		("from_alps3", "Evaluate hybridization function or bubble, from results calculated with Alps3")
		("input-file", po::value<string >(), "input file");
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
	// Initialize output behavior, check input number
	cout << fixed << setprecision(7);
	tie(tmp_file_name, computation_type, from_alps3) = handle_command_line(vm, desc);
	init(world_rank, computation_type, output_file_name, old_output_file_name);
	const string input_file(tmp_file_name);
	// Note: the input (parameter) file also contains the
	// result of previous runs for sigma.
	alps::hdf5::archive h5_archive(input_file, alps::hdf5::archive::READ);
	boost::shared_ptr<Chemicalpotential> chempot(
		new Chemicalpotential(parms, world_rank));
	// Compute hybridization function
	if ((computation_type == 0) || (computation_type == 4)) {
		bool compute_bubble(computation_type == 4 ? true : false);
		boost::shared_ptr<Bandstructure> bare_band(
			new Bandstructure(parms, world_rank, true));
		string h5_group_name("/current_sigma");
		if (!from_alps3) {
			boost::shared_ptr<Selfenergy> self_energy(
				new Selfenergy(parms, world_rank, h5_archive, h5_group_name, true));
			{
				boost::timer::auto_cpu_timer all_loop;
				boost::shared_ptr<DMFTModel> dmft_model(
					new DMFTModel(bare_band, self_energy,
						      parms, world_rank));
				// Restrict reading to process 0, then broadcast.
				if (world_rank == 0) {
					// HERE mu should be retrieved directly from crystal
					// field indo + bare hamiltonian energies
					tie(found_old_mu, old_chemical_potential, dn_dmu) =
						get_old_mu(old_output_file_name);
					// HERE Horrible bug fix in order to align crystal
					// field and value in scf file!! Has to be changed, and old_chemical_potential
					// only retrieved from crystal_field.h5
					old_chemical_potential = (*chempot)[0];
				}
				MPI_Bcast(&found_old_mu, 1, MPI::BOOL, 0, MPI_COMM_WORLD);
				MPI_Bcast(&old_chemical_potential, 1, MPI::DOUBLE, 0, MPI_COMM_WORLD);
				MPI_Bcast(&dn_dmu, 1, MPI::DOUBLE, 0, MPI_COMM_WORLD);
				boost::timer::auto_cpu_timer mu_calc;
				tie(newton_success, new_chemical_potential, density, new_dn_dmu) =
					dmft_model->get_mu_from_density(old_chemical_potential);
				if (newton_success == false) {
					tie(bisec_success, new_chemical_potential, density) =
						dmft_model->get_mu_from_density_bisec(old_chemical_potential, dn_dmu);
				} else {
					dn_dmu = new_dn_dmu;
				}
				if ((newton_success == false) && (bisec_success == false)) {
					if (world_rank == 0) {
						cout << "MU was not found. Quitting. " << endl;
						throw runtime_error("Unable to find mu !");
					}
				}
				if (world_rank == 0) {
					cout << ":MU " << new_chemical_potential << "    "
					     << abs(dn_dmu) << endl;
					cout << ":NTOTAL " << density << endl;
					cout << "<E_kin> =  " << dmft_model->get_kinetic_energy() << endl;
				}
				chempot->apply_shift(-old_chemical_potential + new_chemical_potential);
				dmft_model->dump_k_resolved_occupation_matrices();
				dmft_model->compute_order_parameter();
				dmft_model->display_occupation_matrix();
				bare_band->compute_bare_dos(new_chemical_potential);
				bare_band->dump_bare_dos();
				chempot->dump_values();
				bool verbose(false);
				boost::shared_ptr<HybFunction> hybridization_function(
					new HybFunction(parms, bare_band, self_energy,
							new_chemical_potential, world_rank,
							compute_bubble, verbose));
				if (compute_bubble) {
					hybridization_function->compute_local_bubble();
					hybridization_function->compute_lattice_bubble();
					hybridization_function->dump_bubble_hdf5();
				}
				if (world_rank == 0) {
					cout << " total " << endl;
				}
			}
		} else {
			// Computation with Alps3 input format
			// Need to read GF and compute Dyson equation
			
		}
	} else if (computation_type == 1) {
		// perform "mix" action
		if (world_rank == 0) {
			// Read the current sigma, and calculate the
			// sigma gotten from QMC + tails
			int ref_site_index = 0;
			std::string old_h5_group_name("/current_sigma");
			boost::shared_ptr<Selfenergy>
				old_self_energy(new Selfenergy(parms, world_rank, h5_archive,
							       old_h5_group_name, false));
			boost::shared_ptr<Selfenergy>
				qmc_self_energy(new Selfenergy(parms, world_rank, chempot, ref_site_index,
							       h5_archive, false));
			h5_archive.close();
			// save old sigma
			alps::hdf5::archive w_h5_archive(input_file, alps::hdf5::archive::WRITE);
			std::string copy_h5_group_name("/old_sigma");
			if (!(old_self_energy->get_is_nil_sigma())) {
				old_self_energy->hdf5_dump(w_h5_archive, copy_h5_group_name);
				for (int tail_order = 0; tail_order < 2; tail_order++) {
					old_self_energy->hdf5_dump_tail(w_h5_archive, copy_h5_group_name,
									ref_site_index, tail_order);
				}
			}
			double alpha = 0.5;
			if (parms.exists("ALPHA") && !(old_self_energy->get_is_nil_sigma())) {
				alpha = parms["ALPHA"];
			} else if (old_self_energy->get_is_nil_sigma()) {
				alpha = 1.0;
			}
			//cout << "alpha is " << endl;
			cout << "Using alpha = " << alpha << " for mixing " << endl;
			// apply mix with parameter alpha
			qmc_self_energy->apply_linear_combination(old_self_energy, alpha);
			// Dump the new current sigma
			std::string new_h5_group_name("/current_sigma");
			qmc_self_energy->hdf5_dump(w_h5_archive, new_h5_group_name);
			for (int tail_order = 0; tail_order < 2; tail_order++) {
				qmc_self_energy->hdf5_dump_tail(w_h5_archive, new_h5_group_name,
								ref_site_index, tail_order);
			}
			// Update seed
			int cur_seed = boost::lexical_cast<int>(parms["SEED"]) + 1;
			if (cur_seed > 1000)
				cur_seed = 100;
			std::stringstream seed_path;
			seed_path << "/parameters/SEED";
			w_h5_archive << alps::make_pvp(seed_path.str(),
						       cur_seed);
			cout << "SEED= " << cur_seed << endl;
			w_h5_archive.close();
		}
	}  else if (computation_type == 2) {
		// dump hamiltonian
		boost::shared_ptr<Bandstructure> bare_band(
			new Bandstructure(parms, world_rank, true));
		bare_band->dump_hamilt(parms);
	} else if (computation_type == 3) {
		// Perform debug action
		bool compute_bubble = false;
		boost::shared_ptr<Bandstructure> bare_band(
			new Bandstructure(parms, world_rank, true));
		string h5_group_name("/current_sigma");
		boost::shared_ptr<Selfenergy> self_energy(
			new Selfenergy(parms, world_rank, h5_archive, h5_group_name, false));
		boost::shared_ptr<DMFTModel> dmft_model(
			new DMFTModel(bare_band, self_energy,
				      parms, world_rank));
		// Restrict reading to process 0, then broadcast.
		if (world_rank == 0) {
			found_old_mu = true;
			old_chemical_potential = 1.3790653;
			dn_dmu = 0.0197718;
		}
		MPI_Bcast(&found_old_mu, 1, MPI::BOOL, 0, MPI_COMM_WORLD);
		MPI_Bcast(&old_chemical_potential, 1, MPI::DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&dn_dmu, 1, MPI::DOUBLE, 0, MPI_COMM_WORLD);
		double debug_density, debug_deriv;
		tie(debug_density, debug_deriv) =
			dmft_model->get_particle_density(old_chemical_potential, dn_dmu);
		cout << "debug_density " << debug_density << endl;
	}
	MPI_Finalize();
	return 0;
}
