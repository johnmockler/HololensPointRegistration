#include "pch.h"
#include <Eigen/Dense>

#define EXPORT_API __declspec(dllexport)

using namespace Eigen;
extern "C" {


	EXPORT_API void calculateLeastSquares(float* H, float* X) {
		//calculates determinant of H, then finds X = U*V_transpose and calculates determinant
		typedef Map<MatrixXf> MapMatrix;
		MapMatrix H_map(H, 3, 3);

		MapMatrix xMap(X, 3, 3);

		JacobiSVD<MatrixXf> svd(H_map, ComputeFullU | ComputeFullV);

		xMap = svd.matrixV() * svd.matrixU().transpose();

		//float det = X_map.determinant();
		//return X;
	}

	EXPORT_API void calculateTransform(float* model_centroid, float* detected_centroid, float* R, float* H) {
		typedef Map<MatrixXf> MapMatrix;

		MapMatrix model_map(model_centroid, 3, 1);
		MapMatrix detected_map(detected_centroid, 3, 1);

		MapMatrix R_map(R, 3, 3);

		MatrixXf T = detected_map - R_map * model_map;

		//float* H = new float(16);
		MapMatrix H_map(H, 4, 4);

		H_map = MatrixXf::Identity(4, 4);
		H_map.block(0, 0, 3, 3) = R_map;
		H_map.block(0, 3, 3, 1) = T;
		//return H;
	}
}