#ifndef CHEM_POT_HPP
#define CHEM_POT_HPP

#include <vector>
#include <fstream>
#include <iostream>
#include <alps/params.hpp>
#include <boost/filesystem.hpp>
#include "band_structure.hpp"

//This class handles the 'mu' term (chemical potential term).
//In the simplest case, 'mu' is just a constant for all orbitals.
//The values can be different through a magnetic or christal field,
//or due to a double counting which is orbitally dependent.
// In that case the values are equal to
// mu - epsilon_k, for orbital k
class Chemicalpotential {
public:
     Chemicalpotential(const alps::params &parms, bool in_alps3, int world_rank)
	  :is_alps3(in_alps3), world_rank_(world_rank) {
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
	       if (is_alps3) {
		    std::string alps3_mufilename = parms["model.hopping_matrix_input_file"];
		    alps3_mufilename_ = alps3_mufilename;
	       }
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
	       // save hdf5 file for sc_loop
	       alps::hdf5::archive ar(mufilename_, alps::hdf5::archive::WRITE);
	       ar << alps::make_pvp("/MUvector", val_);
	       ar.close();
	       if (is_alps3) {
		    // save txt file for Alps3
		    ifstream filein(alps3_mufilename_); //File to read from
		    ofstream fileout("temp_chempot.txt"); //Temporary file
		    if(!filein || !fileout)
		    {
			 cout << "Error opening files!" << endl;
			 throw runtime_error("Pb in read/write txt version of chempot !");
		    }
		    string strTemp;
		    //bool found = false;
		    int line_idx = 0;
		    while (std::getline(filein, strTemp))
		    {
			 if((line_idx % (val_.size() + 1)) == 0) {
			      std::stringstream correct_line;
			      int orb_index = line_idx / (val_.size() + 1);
			      correct_line << orb_index << " " << orb_index << " "
					   << -val_[orb_index] << " " << 0.0 << endl;
			      fileout << correct_line.str().c_str();
			 } else {
			      cout << "tmp " << strTemp;
			      strTemp += "\n";
			      fileout << strTemp;
			 }
			 line_idx++;
		    }
		    fileout.close();
		    boost::filesystem::copy_file("temp_chempot.txt", alps3_mufilename_,
		    				 boost::filesystem::copy_option::overwrite_if_exists);
	       }
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
     }
     
private:
     bool is_alps3;
     std::vector<double> val_;
     int world_rank_;
     std::string mufilename_;
     std::string alps3_mufilename_;
};

#endif
