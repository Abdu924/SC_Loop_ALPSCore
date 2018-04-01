#pragma once

#include <boost/multi_array.hpp>

using namespace std;

class Bubble {
     /*
      * class providing interface for the bubbles
      */
public:
     Selfenergy();
     /* for each frequency, 
      * data is contained in a 4-dimensional matrix, 
      * each dimension has size tot_orbital_size
      */
     boost::multi_array<std::complex<double>, 5> values_;
     
     virtual ~Selfenergy() {}
     
};
