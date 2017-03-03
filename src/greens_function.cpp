#include <boost/multi_array.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/hdf5/complex.hpp>
#include "greens_function.hpp"

using namespace std;

Greensfunction::Greensfunction(const alps::params &parms, int world_rank,
			       alps::hdf5::archive h5_archive, string h5_group_name, bool verbose)
     :world_rank_(world_rank) {
}
