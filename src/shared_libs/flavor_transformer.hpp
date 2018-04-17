#pragma once

#include <complex>
#include <cmath>
#include <vector>
#include <assert.h>

#include <boost/bimap.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/multi_array.hpp>

#include<Eigen/Dense>

typedef boost::bimap<std::pair<int, int>, int> bm_type;
typedef bm_type::value_type triplet_type;

class FlavorTransformer {
     
public:
     FlavorTransformer();
     virtual ~FlavorTransformer() {};

     int get_col_from_pair(int orb1, int orb2) {return col_from_orbital_pair.left.at(std::make_pair(orb1, orb2)); }
     int get_line_from_pair(int orb1, int orb2) {return line_from_orbital_pair.left.at(std::make_pair(orb1, orb2)); }
     
private:
     bm_type line_from_orbital_pair;
     bm_type col_from_orbital_pair;
};
