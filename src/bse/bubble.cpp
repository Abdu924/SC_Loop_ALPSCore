#include "bubble.hpp"

using namespace std;

Bubble::Bubble() {
     values_.resize(boost::extents
		[static_cast<int>(n_orbitals)]
		[static_cast<int>(n_orbitals)]
		[static_cast<int>(n_orbitals)]
		[static_cast<int>(n_orbitals)]
		[static_cast<int>(N_l_G4)]
		[static_cast<int>(N_l_G4)]
		[static_cast<int>(N_W)]);

}
