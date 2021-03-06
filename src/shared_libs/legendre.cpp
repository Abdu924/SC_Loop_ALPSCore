#include "legendre.hpp"

LegendreTransformer::LegendreTransformer(int n_matsubara, int n_legendre):
     n_matsubara_(n_matsubara), n_legendre_(n_legendre),
     Tnl_(n_matsubara, n_legendre), Tnl_neg_(n_matsubara, n_legendre), inv_l_(n_legendre) {
     double sign_tmp = 1.0;
     double neg_tmp = 1.0;
     for (int im = 0; im < n_matsubara_; ++im) {
          std::complex<double> ztmp(0.0, 1.0);
          for (int il = 0; il < n_legendre_; ++il) {
               Tnl_(im, il) = sign_tmp * ztmp * std::sqrt(2 * il + 1.0) * boost::math::sph_bessel(il, 0.5 * (2 * im + 1) * M_PI);
               Tnl_neg_(im, il) = neg_tmp * Tnl_(im, il);
               neg_tmp *= -1.0;
               ztmp *= std::complex<double>(0.0, 1.0);
          }
          sign_tmp *= -1;
     }
     sqrt_2l_1.resize(n_legendre);
     sqrt_2l_1[0] = 1.0;
     for (int l = 1; l < n_legendre_; l++) {
          inv_l_[l] = 1.0 / l;
          sqrt_2l_1[l] = std::sqrt(2.0 * l + 1.0);
     }
};

// The shifted version
// The easiest way to figure out what happens for negative frequencies
// is to use the definition of T_{ol}, right below Eq. 4.5
// in L. Behnke's PhD thesis.
LegendreTransformer::LegendreTransformer(int n_matsubara, int n_legendre, int boson_index):
     n_matsubara_(n_matsubara), n_legendre_(n_legendre),
     Tnl_(n_matsubara, n_legendre), Tnl_neg_(n_matsubara, n_legendre), inv_l_(n_legendre) {
     double neg_tmp = 1.0;
     for (int im = 0; im < n_matsubara_; ++im) {
          for (int il = 0; il < n_legendre_; ++il) {
               int power_index = std::abs(2 * im + boson_index + 1);
               Tnl_(im, il) = std::pow(std::complex<double>(0.0, 1.0), power_index + il)
                    * std::sqrt(2 * il + 1.0) *
                    boost::math::sph_bessel(il, 0.5 * power_index * M_PI);
               int neg_power_index = std::abs(-2 * im + boson_index + 1);
               Tnl_neg_(im, il) = std::pow(std::complex<double>(0.0, 1.0), neg_power_index + il)
                    * std::sqrt(2 * il + 1.0) *
                    boost::math::sph_bessel(il, 0.5 * neg_power_index * M_PI);
               if ((-2 * im + boson_index + 1) < 0)
                    Tnl_neg_(im, il) = std::conj(Tnl_neg_(im, il));
          }
     }
     sqrt_2l_1.resize(n_legendre);
     sqrt_2l_1[0] = 1.0;
     for (int l = 1; l < n_legendre_; l++) {
          inv_l_[l] = 1.0 / l;
          sqrt_2l_1[l] = std::sqrt(2.0 * l + 1.0);
     }
};

void LegendreTransformer::compute_legendre(double x, std::vector<double> &val) const {
     assert(val.size() >= n_legendre_);
     assert(x >= -1.00001 && x <= 1.00001);
     for (int l = 0; l < n_legendre_; l++) {
	  if (l == 0) {
	       val[l] = 1;
	  } else if (l == 1) {
	       val[l] = x;
	  } else {
	       //val[l] = ((2 * l - 1) * x * val[l-1] - (l - 1) * val[l-2]) / static_cast<double>(l);//l
	       val[l] = ((2 * l - 1) * x * val[l - 1] - (l - 1) * val[l - 2]) * inv_l_[l];//l
	  }
     }
}

const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &LegendreTransformer::Tnl() const {
     return Tnl_;
}

const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> &LegendreTransformer::Tnl_neg() const {
     return Tnl_neg_;
}

void LegendreTransformer::compute_legendre(const std::vector<double> &xval,
					   boost::multi_array<double, 2> &val) const {
     assert(val.shape()[0] >= n_legendre_);
     const int nx = xval.size();
#ifndef NDEBUG
     for (int ix = 0; ix < nx; ++ix) {
	  assert(xval[ix] >= -1.00001 && xval[ix] <= 1.00001);
     }
#endif
     for (int l = 0; l < n_legendre_; l++) {
	  if (l == 0) {
	       for (int ix = 0; ix < nx; ++ix) {
		    val[l][ix] = 1;
	       }
	  } else if (l == 1) {
	       for (int ix = 0; ix < nx; ++ix) {
		    val[l][ix] = xval[ix];
	       }
	  } else {
	       //for (int ix=0; ix<nx; ++ix) {
	       //val[ix][l] = ((2 * l - 1) * xval[ix]*val[ix][l - 1] - (l - 1) * val[ix][l - 2]) * inv_l_[l];//l
	       //}
	       const double inv_l_tmp = inv_l_[l];
	       for (int ix = 0; ix < nx; ++ix) {
		    val[l][ix] = ((2 * l - 1) * xval[ix] *
				  val[l - 1][ix] - (l - 1) * val[l - 2][ix]) * inv_l_tmp;//l
	       }
	  }
     }
}
