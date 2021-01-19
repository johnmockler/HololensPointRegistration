#define EXPORT_API __declspec(dllexport)
#include <Eigen/Dense>
#include "PointRegister.h"

using namespace Eigen;
extern "C" {

	EXPORT_API float registerIsotropic(float* X, float* Y,  int N, float* outR, float* outT) {		
		typedef Map<MatrixXf> MapMatrix;
		MapMatrix x_map(X, 3, N);
		MapMatrix y_map(Y, 3, N);
		MapMatrix r_map(outR, 3, 3);
		MapMatrix t_map(outT, 3, 1);

		PointRegister pr(x_map, y_map, N);

		float FRE = pr.solveIsotropic();

		r_map = pr.getR();
		t_map = pr.getT();
		return FRE;
	}

	EXPORT_API float registerAnisotropic(float* X, float* Y, float* W, int N, float threshold, float* outR, float* outT) {
		typedef Map<MatrixXf> MapMatrix;
		MapMatrix x_map(X, 3, N);
		MapMatrix y_map(Y, 3, N);
		MapMatrix w_map(W, 3, 3);
		MapMatrix r_map(outR, 3, 3);
		MapMatrix t_map(outT, 3, 1);

		PointRegister pr(x_map, y_map, N, w_map);

		float FRE =  pr.solveAnisotropic(threshold);

		r_map = pr.getR();
		t_map = pr.getT();
		return FRE;
	}
}