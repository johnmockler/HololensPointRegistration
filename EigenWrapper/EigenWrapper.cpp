#define EXPORT_API __declspec(dllexport)
#include <Eigen/Dense>
#include "PointRegister.h"

using namespace Eigen;
extern "C" {


	EXPORT_API void registerIsotropic(float* X, float* Y, float* outR, int N, float* outT, float* outFRE) {
		//Register two point clouds, where the correspondences are already known, but are affected by isotropic, zero mean measurement noise
		/*
			* X is the moving set, which is registered to the static set Y. Both are 3
			by N, where N is the number of fiducials.
			Uses method from Least-squares fitting of two 3-D point sets. IEEE T Pattern Anal by Arun et. al.
			Code adapted from matlab example from balachadran et al (see anisometric case)
			*/
		typedef Map<MatrixXf> MapMatrix;
		MapMatrix x_map(X, 3, N);
		MapMatrix y_map(Y, 3, N);
		MapMatrix r_map(outR, 3, 3);
		MapMatrix t_map(outT, 3, 1);

		PointRegister pr(x_map, y_map, N);

		*outFRE = pr.solveIsotropic();
		r_map = pr.getR();
		t_map = pr.getT();

	}

	EXPORT_API void registerAnisotropic(float* X, float* Y, float* W, int N, float threshold, float* outR, float* outT, float* outFRE, float* outN)
	{

		typedef Map<MatrixXf> MapMatrix;
		MapMatrix x_map(X, 3, N);
		MapMatrix y_map(Y, 3, N);
		MapMatrix w_map(W, 3, 3);
		MapMatrix out_r_map(outR, 3, 3);
		MapMatrix out_t_map(outT, 3, 1);

		PointRegister pr(x_map, y_map, N, w_map);

		*outFRE = pr.solveAnisotropic(threshold);
		*outN = pr.getN_Iter();
		out_r_map = pr.getR();
		out_t_map = pr.getT();

	}
}