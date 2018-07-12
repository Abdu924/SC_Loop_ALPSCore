#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/filesystem.hpp>
#include "../shared_libs/band_structure.hpp"
#include "../shared_libs/self_energy.hpp"
#include "../shared_libs/greens_function.hpp"
#include "../shared_libs/dmft_model.hpp"
#include "hybridization_function.hpp"
#include <fstream>
#include <boost/timer/timer.hpp>
#include <mpi.h>
#include "../shared_libs/chemical_potential.hpp"
#include "../shared_libs/cmd_line.hpp"
#include <alps/params.hpp>
#include <alps/params/convenience_params.hpp>
#include <alps/mc/mcbase.hpp>

using namespace std;

void init(int world_rank, int computation_type,
	  string &output_file_name, string &backup_file_name) {
     string error_file_name;
     if (computation_type == 0) {
	  output_file_name = "c_dmft.out";
	  error_file_name = "error_dmft.txt";
     } else if (computation_type == 1) {
	  output_file_name = "c_mix.out";
	  error_file_name = "error_mix.txt";
     } else if (computation_type == 2) {
	  output_file_name = "c_dump_hamilt.out";
	  error_file_name = "error_dump.txt";
     } else if (computation_type == 3) {
	  output_file_name = "c_lattice_gf.out";
	  error_file_name = "error_lattice_gf.txt";
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
	  auto dummy2 = freopen(error_file_name.c_str(), "a", stderr);
	  cout << "Using Boost "
	       << BOOST_VERSION / 100000     << "."  // major version
	       << BOOST_VERSION / 100 % 1000 << "."  // minor version
	       << BOOST_VERSION % 100                // patch level
	       << endl;
     }
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
	       cout << "Dump lattice gf" << "\n";
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
	  int newton_success = 0;
          int bisec_success = 0;
          int found_old_mu = 0;
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
	  if (computation_type == 0) {
	       alps::hdf5::archive h5_archive(input_file, "r");
	       boost::shared_ptr<Bandstructure> bare_band(
		    new Bandstructure(parms, world_rank, true));
	       string h5_group_name = parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>() ?
		    Selfenergy::legendre_self_energy_name : Selfenergy::matsubara_self_energy_name;
	       if (world_rank == 0) {
		    if (parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>()) {
			 std::cout << "Using LEGENDRE for SC LOOP " << std::endl << std::endl;
		    } else {
			 std::cout << "Using Matsubara for SC LOOP " << std::endl << std::endl;
		    }
	       }
               int ref_site_index = 0;
	       boost::shared_ptr<Selfenergy> self_energy(
		    new Selfenergy(parms, world_rank, ref_site_index, h5_archive, h5_group_name, true));
	       h5_archive.close();
	       {
		    boost::timer::auto_cpu_timer all_loop;
		    // Restrict reading to process 0, then broadcast.
		    if (world_rank == 0) {
			 // HERE Horrible bug fix in order to align crystal
			 // field and value in scf file!! Has to be changed, and old_chemical_potential
			 // only retrieved from crystal_field.h5
			 old_chemical_potential = ((*chempot)[0] + (*chempot)[2]) / 2.0;
			 found_old_mu = 1;
			 dn_dmu = chempot->get_dn_dmu();
		    }
		    MPI_Bcast(&found_old_mu, 1, MPI_INT, 0, MPI_COMM_WORLD);
		    MPI_Bcast(&old_chemical_potential, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		    MPI_Bcast(&dn_dmu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    bool compute_bubble(false);
                    boost::shared_ptr<DMFTModel> dmft_model(
			 new DMFTModel(bare_band, self_energy, parms, old_chemical_potential,
                                       compute_bubble, world_rank));
		    boost::timer::auto_cpu_timer mu_calc;
                    // if we are in the self-consistency test, find the new chemical potential
                    tie(newton_success, new_chemical_potential, density, new_dn_dmu) =
                         dmft_model->get_mu_from_density(old_chemical_potential);
		    if (newton_success == 0) {
			 tie(bisec_success, new_chemical_potential, density) =
			      dmft_model->get_mu_from_density_bisec(old_chemical_potential, dn_dmu);
		    } else {
			 dn_dmu = new_dn_dmu;
			 chempot->set_dn_dmu(dn_dmu);
		    }
		    if ((newton_success == 0) && (bisec_success == 0)) {
			 if (world_rank == 0) {
			      cout << "MU was not found. Quitting. " << endl;
			      throw runtime_error("Unable to find mu !");
			 }
		    }
                    dmft_model->set_chemical_potential(new_chemical_potential);
		    if (world_rank == 0) {
			 cout << ":MU " << new_chemical_potential << "    "
			      << abs(dn_dmu) << endl;
			 cout << ":NTOTAL " << density << endl;
			 cout << "<E_kin> =  " << dmft_model->get_kinetic_energy() << endl;
                         cout << "<U> =  " << dmft_model->get_potential_energy() << endl;
                         cout << "E_tot =  " << dmft_model->get_potential_energy() +
                              dmft_model->get_kinetic_energy() << endl;
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
					 new_chemical_potential, world_rank, verbose));
		    if (world_rank == 0)
		    {
			 alps::hdf5::archive w_h5_archive(input_file, "w");
			 hybridization_function->dump_G0_hdf5(w_h5_archive);
			 hybridization_function->dump_G0_for_ctint_hdf5(w_h5_archive);
			 w_h5_archive.close();
		    }
		    MPI_Barrier(MPI_COMM_WORLD);
		    if (world_rank == 0) {
			 cout << " total " << endl;
		    }
	       }
	  } else if (computation_type == 1) {
	       // perform "mix" action
	       alps::hdf5::archive h5_archive(input_file, "r");
	       bool verbose = false;
	       std::cout << "do mix" << std::endl;
	       boost::shared_ptr<Bandstructure> bare_band(
		    new Bandstructure(parms, world_rank, true));
	       if (world_rank == 0) {
		    // Read the current sigma, and calculate the
		    // sigma gotten from QMC + tails
		    int ref_site_index = 0;
		    string old_h5_group_name = parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>() ?
			 Selfenergy::legendre_self_energy_name : Selfenergy::matsubara_self_energy_name;
		    boost::shared_ptr<Selfenergy>
			 old_self_energy(new Selfenergy(parms, world_rank, ref_site_index, h5_archive,
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
		    alps::hdf5::archive w_h5_archive(input_file, "w");
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
                         legendre_greens_function->dump_single_site_full_gf_legendre(w_h5_archive, ref_site_index);
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
	       alps::hdf5::archive h5_archive(input_file, "r");
	       boost::shared_ptr<Bandstructure> bare_band(
		    new Bandstructure(parms, world_rank, true));
	       string h5_group_name = parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>() ?
		    Selfenergy::legendre_self_energy_name : Selfenergy::matsubara_self_energy_name;
	       if (world_rank == 0) {
		    if (parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>()) {
			 std::cout << "Using LEGENDRE for SC LOOP " << std::endl << std::endl;
		    } else {
			 std::cout << "Using Matsubara for SC LOOP " << std::endl << std::endl;
		    }
	       }
               int ref_site_index = 0;
	       boost::shared_ptr<Selfenergy> self_energy(
		    new Selfenergy(parms, world_rank, ref_site_index, h5_archive, h5_group_name, true));
	       h5_archive.close();
               // Restrict reading to process 0, then broadcast.
               if (world_rank == 0) {
                    old_chemical_potential = ((*chempot)[0] + (*chempot)[2]) / 2.0;
                    found_old_mu = 1;
                    dn_dmu = chempot->get_dn_dmu();
               }
               MPI_Bcast(&found_old_mu, 1, MPI_INT, 0, MPI_COMM_WORLD);
               MPI_Bcast(&old_chemical_potential, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
               MPI_Bcast(&dn_dmu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
               bool compute_bubble(false);
               boost::shared_ptr<DMFTModel> dmft_model(
                    new DMFTModel(bare_band, self_energy, parms, old_chemical_potential,
                                  compute_bubble, world_rank));
               dmft_model->compute_lattice_gf(old_chemical_potential);
               MPI_Barrier(MPI_COMM_WORLD);
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
