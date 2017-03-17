#ifndef CHEM_POT_HPP
#define CHEM_POT_HPP

#include <vector>
#include <fstream>
#include <alps/params.hpp>
#include "band_structure.hpp"

//This class handles the 'mu' term (chemical potential term).
//In the simplest case, 'mu' is just a constant for all orbitals.
//The values can be different through a magnetic or christal field,
//or due to a double counting which is orbitally dependent.
// In that case the values are equal to
// mu - epsilon_k, for orbital k
class Chemicalpotential {
public:
     Chemicalpotential(const alps::params &parms, int world_rank)
	  :world_rank_(world_rank) {
	  val_.resize(parms.exists("N_ORBITALS") ?
		      static_cast<size_t>(parms["N_ORBITALS"]) : 2,
		      parms.exists("MU") ?
		      static_cast<double>(parms["MU"]) : 0.0);
	  if (parms.exists("MU_VECTOR")) {
	       if (!world_rank_) {
		    std::cout << "Warning::parameter MU_VECTOR defined, ignoring parameter MU"
			      << std::flush << std::endl;
	       }
	       // parms delivers a cons attribute...
	       std::string mufilename = parms["MU_VECTOR"];
	       mufilename_ = mufilename;
	       if(parms.exists("MU_IN_HDF5") &&
		  (static_cast<bool>(parms["MU_IN_HDF5"]))) {
		    //attempt to read from h5 archive
		    alps::hdf5::archive ar(mufilename_, alps::hdf5::archive::READ);
		    ar >> alps::make_pvp("/MUvector", val_);
	       } else {
		    // read from text file
		    std::ifstream mu_file(mufilename_.c_str());
		    if(!mu_file.good())
			 throw std::runtime_error("Problem reading in MU_VECTOR.");
		    std::size_t i = 0;
		    double MU_i;
		    for (i = 0; i < n_orbitals(); ++i) {
			 mu_file >> MU_i;
			 val_[i] = MU_i;
			 if(!mu_file.good())
			      throw std::runtime_error("Problem reading in MU_VECTOR.");
		    }
	       }
	  }
     }
     
     std::size_t n_orbitals(void)const {
	  return val_.size();
     }
     
     const double &operator[] (std::size_t flavor)const {
	  return val_[flavor];
     }
     
     void apply_shift(const double shift){
	  for(std::size_t i = 0; i < n_orbitals(); ++i)
	       val_[i] += shift; //apply shift
     }

     void dump_values() {
	  if(world_rank_ == 0) {
	       alps::hdf5::archive ar(mufilename_, alps::hdf5::archive::WRITE);
	       ar << alps::make_pvp("/MUvector", val_);
	       ar.close();
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
     }
     
private:
     std::vector<double> val_;
     int world_rank_;
     std::string mufilename_;
};

#endif
