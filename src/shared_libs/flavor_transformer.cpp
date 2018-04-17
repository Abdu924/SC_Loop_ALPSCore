#include "flavor_transformer.hpp"

FlavorTransformer::FlavorTransformer() {
     // we go from
     // aup bdown adown bup
     // to
     // aup bup adown bdown
     line_from_orbital_pair.clear();
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 0), 0));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 3), 1));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 0), 2));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 3), 3));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 2), 4));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 1), 5));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 2), 6));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 1), 7));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 2), 8));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 1), 9));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 2), 10));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 1), 11));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 0), 12));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 3), 13));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 0), 14));
     line_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 3), 15));
     col_from_orbital_pair.clear();
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 0), 0));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 0), 1));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 3), 2));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 3), 3));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 2), 4));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 2), 5));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 1), 6));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 1), 7));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 0), 8));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 0), 9));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(2, 3), 10));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(1, 3), 11));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 2), 12));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 2), 13));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(0, 1), 14));
     col_from_orbital_pair.insert(triplet_type(std::pair<int, int>(3, 1), 15));
}
