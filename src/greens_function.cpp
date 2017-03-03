#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pointer.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/multi_array.hpp>
#include "greens_function.hpp"


using namespace std;

Greensfunction::Greensfunction(const alps::params &parms, int world_rank,
			       alps::hdf5::archive &h5_archive, string h5_group_name,
			       bool verbose)
	:world_rank_(world_rank) {
	//std::map<std::string,boost::any> &ar = static_cast<std::map<std::string,boost::any>(h5_archive);
	//boost::multi_array<std::complex<double>, 3>(boost::extents[4][4][40]) test;
	//= h5_archive["G1_LEGENDRE"];
	//h5_archive["G1_LEGENDRE"] = raw_gf_data;
	//raw_gf_data = static_cast<boost::multi_array<std::complex<double>, 3> >(h5_archive["G1_LEGENDRE"]);
	typedef 	boost::multi_array<double, 4> array_type;
	typedef 	boost::multi_array<complex<double> , 3> cplx_array_type;
	typedef array_type::index index;
	array_type test(boost::extents[4][4][40][2]);

	
	h5_archive["G1_LEGENDRE"] >> test;
	cout << "test element :" << test[0][0][0][0] << endl;

	cplx_array_type test2(boost::extents[955][4][4]);

	
	h5_archive["/gf/data"] >> test2;
	cout << "test element :" << test2[0][0][0] << endl;

	typedef array_type::index_range range;
	array_type::array_view<3>::type myview =
		test2[ boost::indices[range(0,2)][range(1,3)][range(0,4,2)] ];
}
