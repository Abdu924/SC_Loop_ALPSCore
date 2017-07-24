#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/filesystem.hpp>
#include "band_structure.hpp"
#include "self_energy.hpp"
#include "greens_function.hpp"
#include "dmft_model.hpp"
#include "bare_hamiltonian.hpp"
#include "hybridization_function.hpp"
#include <fstream>
#include <boost/timer/timer.hpp>
#include <mpi.h>
#include "chemical_potential.hpp"
#include <alps/mc/mcbase.hpp>

using namespace std;

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
     //alps::mcbase::define_parameters(parameters);
     // Adds the convenience parameters (for save/load)
     // alps::define_convenience_parameters(parameters);
     // Define ct-hyb related parameters
     parameters
	  .description("hybridization expansion simulation")
	  .define<std::string>("input-file","", "hdf5 input file with relevant data")
	  .define<int >("action", "int describing the action to be performed")
	  .define<bool>("from_alps3", false, "is the input produced by Alps3?")
	  .define<long>("SEED", 42, "PRNG seed")
	  .define<bool>("cthyb.ACCURATE_COVARIANCE", false, "TODO: UNDERSTAND WHAT THIS DOES")
	  .define<bool>("model.compute_spin_current", false, "Compute the spin current components")
	  .define<std::string>("cthyb.BASEPATH","", "path in hdf5 file to which results are stored")
	  .define<double>("model.beta", "inverse temperature")
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
	  .define<int >("N_QBSEQ", 1, "number of q points for lattice bubble calculation")
	  .define<double >("MAX_QBSEQ", 0.5, "max value of q coord for lattice bubble calculation")
	  .define<int >("N_NU_BSEQ", 0, "number of fermionic frequencies for bubble calculation")
	  .define<std::string>("model.hopping_matrix_input_file", "path for local hopping description file")
	  .define<int >("model.sites", 2, "Number of orbitals (Alps3 convention)")
	  .define<bool >("REAL_DELTA", false, "if true, we force the hybridization function to be real")
	  .define<double>("N_ELECTRONS", "electronic density")
	  .define<int >("N_ORBITALS", 4, "total number of spin-orbitals")
	  .define<double>("C_MAX", "parameter for tail adjustment of gf")
	  .define<double>("C_MIN", "parameter for tail adjustment of gf")
	  .define<double>("mixing.ALPHA", 0.5, "parameter for combination of new and old self-energies")	  
	  .define<bool>("cthyb.DMFT_FRAMEWORK", false,
			"true if we need to tie into a dmft framework")
	  .define<bool>("cthyb.GLOBALFLIP", false, "TODO: UNDERSTAND WHAT THIS DOES.")
	  .define<double>("cthyb.J", 0, "interaction value for density-density Hund's coupling term J.")
	  .define<bool>("cthyb.K_IN_HDF5", false, "set to true if retarded interaction K is stored in hdf5.")
	  .define<bool>("cthyb.MEASURE_freq", true, "measure in frequency domain")
	  .define<bool>("cthyb.MEASURE_g2w", false, "measure two-particle Green's function in frequency space")
	  .define<bool>("cthyb.MEASURE_h2w", false, "measure two-particle H Green's function in frequency space")
	  .define<bool>("cthyb.MEASURE_legendre", false, "measure legendre Green's function coefficients")
	  .define<bool>("cthyb.MEASURE_nn", false, "measure static density-density correlation functions")
	  .define<bool>("cthyb.MEASURE_nnt", false, "measure density-density correlation functions <n(0) n(t)>")
	  .define<bool>("cthyb.MEASURE_nnw", false,
			"measure density-density correlation functions in frequency domain")
	  .define<bool>("cthyb.MEASURE_sector_statistics",false, "measure sector statistics")
	  .define<bool>("cthyb.MEASURE_time",false, "measure in the time domain")
	  .define<bool>("mixing.analytic_sigma_tail", true,
			"decide whether to calculate of fit the self energy tail")
	  .define<bool>("mixing.FIX_C1", false,
			"decide whether to fix 1st order of the tail of the GF, when using Legendre sampling")
	  .define<bool>("mixing.FIX_C2", false,
			"decide whether to fix the second order of the tail of the GF, when using legendre sampling")
	  .define<double>("MU", "chemical potential / orbital energy values")
	  .define<std::string>("MU_VECTOR", "file name for file with chemical potential / orbital energy values")
	  .define<bool>("MU_IN_HDF5", false,"true if the file MU_VECTOR points to a hdf5 file")
	  .define<int >("cthyb.N_HISTOGRAM_ORDERS",200, "orders for the histograms of probability per order")
	  .define<int >("cthyb.N_LEGENDRE",0,"number of legendre coefficients")
	  .define<int >("mixing.L_MAX", 0, "Nb of used Legendre coefficients in DMFT")
	  .define<bool>("mixing.LEGENDRE_FOR_SC_LOOP", false, "Rely on Legendre data for hybridization function computation and mixing")
	  .define<int >("N_MATSUBARA", 955, "number of matsubara frequencies")
	  .define<int >("measurement.G1.N_MATSUBARA", 955, "number of matsubara frequencies for alps3")
	  .define<int >("cthyb.N_MEAS","number of updates per measurement")
	  .define<int >("FLAVORS","number of spin-orbitals (sometimes called flavors)")
	  .define<int >("N_TAU","number of imaginary time discretization points")
	  .define<int>("measurement.G2.n_bosonic_freq", 1, "Number of bosonic frequencies for measurement")
	  .define<int >("cthyb.N_nn", 0,
			"number of points for the measurement of the density density correlator")
	  .define<int >("cthyb.N_w2", 0,
			"number of fermionic frequencies for the two-particle measurement (-N_w2 ... N_w2-1)")
	  .define<std::string>("cthyb.RET_INT_K",
			       "file with the retarted interaction information. See doc for format.")
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
	  .define<std::string>("solver.OUTFILE_H5GF", "alps_solver_in.h5gf",
			       "H5GF Green's function input file containing G0(omega,k) "
			       "at /G0 and G0(ij, tau) at /G0_tau_rs")
	  .define<std::string>("solver.INFILE_H5GF", "alps_solver_out.h5gf",
			       "H5GF Green's function output file containing G(omega,k) "
			       "at /G_omega and G(ij, tau) at /G_tau")
	  .define<int >("L", 200, "Number of Brillouin Zone points")
	  .define<double>("t", 1.0, "Hopping element: nearest neighbor")
	  .define<double>("tprime", 0.0, "Hopping element: next-nearest neighbor")
	  .define<int >("measurement.nn_corr.n_tau", 3,
			"number of pts for density-density correlation functions.")
	  .define<int >("measurement.G1.n_tau", 1800, "number of pts for G(tau).")
	  .define<std::string>("measurement.nn_corr.def",
			       "definition of density-density correlation functions.")
	  ;
}

tuple<string, int, bool> handle_command_line(alps::params par) {
     int computation_type(-1);
     bool from_alps3(par["from_alps3"].as<bool>());
     string input_file("");
     if (!par.exists("input-file")) {
	  std::cout << "You must provide the name of the input file" << std::endl;
	  par["help"] = true;
     } else {
	  input_file = par["input-file"].as<string>();
	  cout << "Input file is: " << input_file << "\n";
     }
     if (!par.exists("action")) {
	  std::cout << "You must provide the name of the action to be performed" << std::endl;
	  par["help"] = true;
     } else {
	  if (par["action"].as<int>() == 0) {
	       cout << "Computing hybridization function" << "\n";
	  } else if (par["action"].as<int>() == 1) {
	       cout << "Running mix action" << "\n";
	  } else if (par["action"].as<int>() == 2) {
	       cout << "Dumping Hamiltonian" << "\n";
	  } else if (par["action"].as<int>() == 3) {
	       cout << "Debug mode" << "\n";
	  } else if (par["action"].as<int>() == 4) {
	       cout << "Computing bubble" << "\n";
	  } else {
	       std::cout << "The requested action was not recognized" << std::endl;
	       par["help"] = true;
	  }
     }
     if (!par["help"])
	  computation_type = par["action"];
     if (from_alps3) {
	  cout << "Taking ***MATRIX RESULT*** as input format\n";
     }
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

int main(int argc, const char* argv[]) {
     int world_size, world_rank;
     MPI_Init(NULL, NULL);
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
     try {
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
	  cout << fixed << setprecision(7);
	  alps::params parms(argc, argv);
	  define_parameters(parms);
	  tie(tmp_file_name, computation_type, from_alps3) = handle_command_line(parms);
	  init(world_rank, computation_type, output_file_name, old_output_file_name);
	  //if (world_rank == 0)
	  //	  std::cout << "Parameters : " << std::endl << parms << std::endl;
	  const string input_file(tmp_file_name);
	  // Note: the input (parameter) file also contains the
	  // result of previous runs for sigma.
	  boost::shared_ptr<Chemicalpotential> chempot(
	       new Chemicalpotential(parms, from_alps3, world_rank));
	  // Compute hybridization function
	  if ((computation_type == 0) || (computation_type == 4)) {
	       alps::hdf5::archive h5_archive(input_file, alps::hdf5::archive::READ);
	       bool compute_bubble(computation_type == 4 ? true : false);
	       boost::shared_ptr<Bandstructure> bare_band(
		    new Bandstructure(parms, world_rank, true));
	       string h5_group_name = parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>() ?
		    HybFunction::legendre_self_energy_name : HybFunction::matsubara_self_energy_name;
	       if (world_rank == 0) {
		    if (parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>()) {
			 std::cout << "Using LEGENDRE for SC LOOP " << std::endl << std::endl;
		    } else {
			 std::cout << "Using Matsubara for SC LOOP " << std::endl << std::endl;
		    }
	       }
	       boost::shared_ptr<Selfenergy> self_energy(
		    new Selfenergy(parms, world_rank, h5_archive, h5_group_name, true));
	       h5_archive.close();
	       {
		    boost::timer::auto_cpu_timer all_loop;
		    boost::shared_ptr<DMFTModel> dmft_model(
			 new DMFTModel(bare_band, self_energy, parms, world_rank));
		    // Restrict reading to process 0, then broadcast.
		    if (world_rank == 0) {
			 // HERE Horrible bug fix in order to align crystal
			 // field and value in scf file!! Has to be changed, and old_chemical_potential
			 // only retrieved from crystal_field.h5
			 old_chemical_potential = (*chempot)[0];
			 found_old_mu == true;
			 dn_dmu = chempot->get_dn_dmu();
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
			 chempot->set_dn_dmu(dn_dmu);
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
		    if (parms["model.compute_spin_current"].as<bool>() == true) {
			 dmft_model->get_spin_current();
			 dmft_model->display_spin_current();
		    }
		    bare_band->compute_bare_dos(new_chemical_potential);
		    bare_band->dump_bare_dos();
		    chempot->dump_values();
		    bool verbose(false);
		    MPI_Barrier(MPI_COMM_WORLD);
		    boost::shared_ptr<HybFunction> hybridization_function(
			 new HybFunction(parms, bare_band, self_energy,
					 new_chemical_potential, world_rank,
					 compute_bubble, verbose));
		    if (world_rank == 0)
		    {
			 alps::hdf5::archive w_h5_archive(input_file, alps::hdf5::archive::WRITE);
			 hybridization_function->dump_G0_hdf5(w_h5_archive);
			 w_h5_archive.close();
		    }
		    MPI_Barrier(MPI_COMM_WORLD);
		    if (compute_bubble) {
			 hybridization_function->compute_local_bubble();
			 hybridization_function->compute_lattice_bubble();
			 hybridization_function->dump_bubble_hdf5();
		    }
		    if (world_rank == 0) {
			 cout << " total " << endl;
		    }
	       }
	  } else if (computation_type == 1) {
	       // perform "mix" action
	       alps::hdf5::archive h5_archive(input_file, alps::hdf5::archive::READ);
	       bool verbose = false;
	       std::cout << "do mix" << std::endl;
	       boost::shared_ptr<Bandstructure> bare_band(
		    new Bandstructure(parms, world_rank, true));
	       if (world_rank == 0) {
		    // Read the current sigma, and calculate the
		    // sigma gotten from QMC + tails
		    int ref_site_index = 0;
		    string old_h5_group_name = parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>() ?
			 HybFunction::legendre_self_energy_name : HybFunction::matsubara_self_energy_name;
		    boost::shared_ptr<Selfenergy>
			 old_self_energy(new Selfenergy(parms, world_rank, h5_archive,
							old_h5_group_name, verbose));
		    boost::shared_ptr<Selfenergy> qmc_self_energy;
		    boost::shared_ptr<Selfenergy> legendre_qmc_self_energy;
		    boost::shared_ptr<Greensfunction> legendre_greens_function;
		    if ((parms["cthyb.MEASURE_freq"]) && (!from_alps3)) {
			 std::cout << "generate Matsubara Sigma" << std::endl;
			 // S_omega or S_l_omega
			 int input_type = 0;
			 qmc_self_energy.reset(new Selfenergy(parms, world_rank, chempot,
							      bare_band, ref_site_index,
							      h5_archive, input_type, verbose));
		    }
		    if (parms["cthyb.MEASURE_legendre"]) {
			 int sampling_type = from_alps3 ? 0 : 1;
			 std::cout << "generate Leg GF" << std::endl;
			 legendre_greens_function.reset(
			      new Greensfunction(parms, world_rank, chempot,
						 bare_band, sampling_type, h5_archive));
			 legendre_qmc_self_energy.reset(
			      new Selfenergy(parms, world_rank, chempot, bare_band,
					     ref_site_index, h5_archive, legendre_greens_function));
		    }
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
		    double alpha(parms["mixing.ALPHA"]);
		    if (old_self_energy->get_is_nil_sigma()) {
			 alpha = 1.0;
			 cout << "Old self-energy not found => Forcing alpha tp 1.0" << endl;
		    }
		    cout << "Using alpha = " << alpha << " for mixing " << endl;
		    // apply mix with parameter alpha
		    if ((parms["cthyb.MEASURE_freq"]) && (!from_alps3)) {
			 qmc_self_energy->apply_linear_combination(old_self_energy, alpha);
			 // Dump the new current sigma
			 std::string new_h5_group_name("/current_sigma");
			 qmc_self_energy->hdf5_dump(w_h5_archive, new_h5_group_name);
			 for (int tail_order = 0; tail_order < 2; tail_order++) {
			      qmc_self_energy->hdf5_dump_tail(w_h5_archive, new_h5_group_name,
							      ref_site_index, tail_order);
			 }
		    }
		    if (parms["cthyb.MEASURE_legendre"]) {
			 legendre_greens_function->dump_single_site_full_gf_matsubara(w_h5_archive, ref_site_index);
			 legendre_qmc_self_energy->apply_linear_combination(old_self_energy, alpha);
			 std::string new_h5_group_name("/current_legendre_sigma");
			 legendre_qmc_self_energy->hdf5_dump(w_h5_archive, new_h5_group_name);
			 for (int tail_order = 0; tail_order < 2; tail_order++) {
			      legendre_qmc_self_energy->hdf5_dump_tail(w_h5_archive, new_h5_group_name,
								       ref_site_index, tail_order);
			 }
		    }
		    w_h5_archive.close();
	       }
	  } else if (computation_type == 2) {
	       // dump hamiltonian
	       boost::shared_ptr<Bandstructure> bare_band(
		    new Bandstructure(parms, world_rank, true));
	       bare_band->dump_hamilt(parms);
	  } else if (computation_type == 3) {
	       // Perform debug action
	       // bool compute_bubble = false;
	       // boost::shared_ptr<Bandstructure> bare_band(
	       //      new Bandstructure(parms, world_rank, true));
	       // string h5_group_name("/current_sigma");
	       // boost::shared_ptr<Selfenergy> self_energy(
	       //      new Selfenergy(parms, world_rank, h5_archive, h5_group_name, false));
	       // boost::shared_ptr<DMFTModel> dmft_model(
	       //      new DMFTModel(bare_band, self_energy,
	       // 		     parms, world_rank));
	       // // Restrict reading to process 0, then broadcast.
	       // if (world_rank == 0) {
	       //      found_old_mu = true;
	       //      old_chemical_potential = 1.3790653;
	       //      dn_dmu = 0.0197718;
	       // }
	       // MPI_Bcast(&found_old_mu, 1, MPI::BOOL, 0, MPI_COMM_WORLD);
	       // MPI_Bcast(&old_chemical_potential, 1, MPI::DOUBLE, 0, MPI_COMM_WORLD);
	       // MPI_Bcast(&dn_dmu, 1, MPI::DOUBLE, 0, MPI_COMM_WORLD);
	       // double debug_density, debug_deriv;
	       // tie(debug_density, debug_deriv) =
	       //      dmft_model->get_particle_density(old_chemical_potential, dn_dmu);
	       // cout << "debug_density " << debug_density << endl;
	  }
	  MPI_Finalize();
	  return 0;
     }
     catch(std::exception& exc){
	  std::cerr<<exc.what()<<std::endl;
	  return -1;
     }
     catch(...){
	  std::cerr << "Fatal Error: Unknown Exception!\n";
	  return -2;
     }
}
