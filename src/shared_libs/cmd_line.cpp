#include <alps/params.hpp>
#include <alps/params/convenience_params.hpp>
#include <alps/mc/mcbase.hpp>

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
	  .define<bool>("model.compute_spin_current", true, "Compute the spin current components")
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
	  .define<std::string>("model.hopping_matrix_input_file", "path for local hopping description file")
          .define<int >("model.space_sites", 1, "Number of real space lattice sites considered")
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
	  .define<int >("bseq.N_QBSEQ", 1, "number of q points for lattice bubble calculation")
	  .define<double >("bseq.MAX_QBSEQ", 0.5, "max value of q coord for lattice bubble calculation")
	  .define<int >("bseq.N_NU_BSEQ", 0, "number of fermionic frequencies for bubble calculation")
          .define<int >("bseq.bubbles.dump_matsubara", 0, "save Matsubara bubbles to hdf5")
          .define<int >("bseq.bubbles.dump_legendre", 0, "save Legendre bubbles to hdf5")
	  ;
}
