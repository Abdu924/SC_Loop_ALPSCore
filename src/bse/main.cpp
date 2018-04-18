#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <vector>
#include <boost/filesystem.hpp>
#include "../shared_libs/band_structure.hpp"
#include "../shared_libs/self_energy.hpp"
#include "../shared_libs/greens_function.hpp"
#include "../shared_libs/dmft_model.hpp"
#include <fstream>
#include <boost/timer/timer.hpp>
#include <mpi.h>
#include "bubble.hpp"
#include "bseq_solver.hpp"
#include "../shared_libs/chemical_potential.hpp"
#include "../shared_libs/cmd_line.hpp"
#include <alps/params.hpp>
#include <alps/params/convenience_params.hpp>
#include <alps/mc/mcbase.hpp>

using namespace std;

void init(int world_rank, int computation_type,
	  string &output_file_name, string &backup_file_name) {
     string error_file_name;
     if ((computation_type == 0) or (computation_type == 1)) {
	  output_file_name = "c_bse.out";
	  error_file_name = "error_bse.txt";
     }
     backup_file_name = output_file_name + ".old";
     if (world_rank == 0) {
	  if (boost::filesystem::exists(output_file_name)) {
	       if (boost::filesystem::is_regular_file(output_file_name)) {
		    boost::filesystem::copy_file(output_file_name, backup_file_name,
						 boost::filesystem::copy_option::overwrite_if_exists);
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
	       cout << "Computing bubbles" << "\n";
	  } else if (par["action"].as<int>() == 1) {
               cout << "Inverting bseq" << "\n";
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

int main(int argc, const char* argv[]) {
     int world_size, world_rank;
     MPI_Init(NULL, NULL);
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
     try {
	  string output_file_name;
	  string old_output_file_name;
	  double density, derivative;
	  double chemical_potential;
          int found_old_mu = 0;
	  // kind of computation
	  // 0 --- compute bubbles
	  int computation_type;
	  bool from_alps3(false);
	  string tmp_file_name;
	  cout << fixed << setprecision(7);
	  alps::params parms(argc, argv);
	  define_parameters(parms);
	  tie(tmp_file_name, computation_type, from_alps3) = handle_command_line(parms);
	  init(world_rank, computation_type, output_file_name, old_output_file_name);
	  const string input_file(tmp_file_name);
	  boost::shared_ptr<Chemicalpotential> chempot(
	       new Chemicalpotential(parms, from_alps3, world_rank));
	  // Compute bubbles
	  if (computation_type == 0) {
               alps::hdf5::archive h5_archive(input_file, "r");
               boost::shared_ptr<Bandstructure> bare_band(
                    new Bandstructure(parms, world_rank, true));
               string h5_group_name = parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>() ?
                    Selfenergy::legendre_self_energy_name : Selfenergy::matsubara_self_energy_name;
	       if (world_rank == 0) {
	            if (parms["mixing.LEGENDRE_FOR_SC_LOOP"].as<bool>()) {
	        	 std::cout << "Using LEGENDRE source for self-energy" << std::endl << std::endl;
	            } else {
	        	 std::cout << "Using Matsubara source for self-energy" << std::endl << std::endl;
	            }
	       }
	       // TODO: self-energy for non interacting case, RPA. Below, constructor for sigma=0
               // boost::shared_ptr<Selfenergy> self_energy(new Selfenergy(parms, world_rank, true));
               boost::shared_ptr<Selfenergy> self_energy(
		    new Selfenergy(parms, world_rank, h5_archive, h5_group_name, true));
               if (world_rank == 0) {
                    //FIXME TODO
                    // HERE Horrible bug fix in order to align crystal
                    // field and value in scf file!! Has to be changed, and old_chemical_potential
                    // only retrieved from crystal_field.h5
                    chemical_potential = ((*chempot)[0] + (*chempot)[2]) / 2.0;
                    found_old_mu = 1;
               }
               MPI_Bcast(&found_old_mu, 1, MPI_INT, 0, MPI_COMM_WORLD);
               MPI_Bcast(&chemical_potential, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
               boost::shared_ptr<Bubble> local_bubble(
                    new Bubble(h5_archive, bare_band, self_energy, parms,
                                    chemical_potential, world_rank));

               h5_archive.close();
               local_bubble->compute_local_bubble();
               local_bubble->compute_lattice_bubble();
               local_bubble->dump_bubble_hdf5(parms);
	  } else if (computation_type == 1) {
               // Solve BSEQ
               boost::shared_ptr<Bandstructure> bare_band(new Bandstructure(parms, world_rank, true));
               alps::hdf5::archive g2_archive(input_file, "r");
               const string bubble_file = parms["bseq.bubbles.filename"].as<string>();
               alps::hdf5::archive bubble_archive(bubble_file, "r");
               int current_bose_freq = 0;
               boost::shared_ptr<BseqSolver> bseq_solver(
                    new BseqSolver(g2_archive, bubble_archive, bare_band,
                                   current_bose_freq, parms, world_rank));
               bseq_solver->inverse_bseq();
               bseq_solver->dump_susceptibility(parms);
               bseq_solver->dump_vertex(parms);
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
