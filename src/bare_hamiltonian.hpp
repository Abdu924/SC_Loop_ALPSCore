#ifndef BARE_HAMILTONIANH__
#define BARE_HAMILTONIANH__

#include <Eigen/Dense>
#include<vector>
#include<iostream>
#include <complex>
#include <cmath>
#include <alps/params.hpp>

using namespace std;

class BareHamiltonian {
     /*
      * Class for the management of the bare dispersion,
      * starting from the hopping parameters provided in hr.dat.
      */
public:
     void dump_hamilt();
     int get_nb_k_points();
     int get_n_flavors();
     

};

#endif //BARE_HAMILTONIANH__
