#define EXPORT_API __declspec(dllexport)
#include <Eigen/Dense>
#include <limits>
#include <unsupported/Eigen/MatrixFunctions>

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

	void C_maker(MatrixXf X, MatrixXf W, int N, Ref<MatrixXf> C) {
		//For homogenous case: Xi is a 3xN array
		
		MatrixXf w1 = W.col(1).replicate(1, N);
		MatrixXf w2 = W.col(2).replicate(1, N);
		MatrixXf w3 = W.col(3).replicate(1, N);
		MatrixXf x1 = X.row(1).replicate(3, 1);
		MatrixXf x2 = X.row(2).replicate(3, 1);
		MatrixXf x3 = X.row(3).replicate(3, 1);
		C << -w2.cwiseProduct(x3) + w3.cwiseProduct(x2), w1.cwiseProduct(x3) - w3.cwiseProduct(x1),
			-w1.cwiseProduct(x2) + w2.cwiseProduct(x1), w1, w2, w3;
		//I am not sure what the permutation does in the main code... (how does it even work, it's not a 3d matrix)
		//I don't think it's necessary. you just need to map this, 3x6N matrix to 3Nx6;
		//Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm;
		//perm.indices() = { 0, 2, 1 };
		//C = C * perm;
		int c_rows = 3 * N;
		C.resize(c_rows, 6);
	}

	void e_maker(MatrixXf X, MatrixXf Y, MatrixXf W, int N, Ref<MatrixXf> e) {
		
		MatrixXf w1 = W.col(1).replicate(1, N);
		MatrixXf w2 = W.col(2).replicate(1, N);
		MatrixXf w3 = W.col(3).replicate(1, N);
		MatrixXf D = Y - X;
		MatrixXf d1 = D.row(1).replicate(3, 1);
		MatrixXf d2 = D.row(2).replicate(3, 1);
		MatrixXf d3 = D.row(3).replicate(3, 1);
		e << w1.cwiseProduct(d1) + w2.cwiseProduct(d2) + w3.cwiseProduct(d3);
		int e_rows = 3 * N;
		e.resize(e_rows, 6);

	}

	EXPORT_API void registerAnisotropic(float* X, float* Y, float* W, int N, float threshold, float* estR, float* estT, float FRE,
		float* outR, float* outT, float* outFRE, float* outN) {
		/*
		* X is the moving set, which is registered to the static set Y. Both are 3
		by N, where N is the number of fiducials. W is a 3-by-3 array, with
		each page containing the weighting matrix.
		THRESHOLD is the size of the change to the moving set above which the
		iteration continues.

		note: here W is estimated to be the same for each fiducial. Without this
		assumption, W would be 3x3xN (and code would need to be modified)
		outputs: Rotation, Translation, FRE and number of iterations
	
		Adapted from Matlab Code authored by:
		R. Balachandran and J. M. Fitzpatrick
		December 2008 
		*/
		typedef Map<MatrixXf> MapMatrix;
		MapMatrix x_map(X, 3, N);
		MapMatrix y_map(Y, 3, N);
		MapMatrix w_map(W, 3, 3);
		MapMatrix r_map(estR, 3, 3);
		MapMatrix t_map(estT, 3, 1);
		MapMatrix out_r_map(outR, 3, 3);
		MapMatrix out_t_map(outT, 3, 1);

		//use when weighting is homogoneous (Same w for each fiducial)
		//MatrixXf w_stretched = w_map.replicate(1, N);

		int n = 0;
		int index = 0;
		float config_change = std::numeric_limits<float>::infinity();
		MatrixXf Xold = r_map * x_map + t_map.replicate(1, N);
		MatrixXf Xnew;
		Vector3f oldq;
		out_r_map = r_map;
		out_t_map = t_map;

		while (config_change > threshold) {
			n = n + 1;

			MatrixXf C;
			C_maker(Xold, w_map, N, C);

			MatrixXf e;
			e_maker(Xold, y_map, w_map, N, e);

			Vector3f q = C.colPivHouseholderQr().solve(e);

			if (n > 1) {
				q = (q + oldq) / 2.0f; //damps osccilations
			}
			oldq = q;
			Vector3f delta_t;
			delta_t << q[4], q[5], q[6];
			delta_t.transpose();
			Matrix3f delta_theta;
			delta_theta << 1, -q[3], q[2],
						q[3], 1, -q[1],
						-q[2], q[1], 1;
			JacobiSVD<Matrix3f> svd(delta_theta, ComputeFullU | ComputeFullV);

			Matrix3f delta_R = svd.matrixU() * svd.matrixV().transpose();

			out_r_map = delta_R * out_r_map; //update rotation
			out_t_map = delta_R * out_t_map + delta_t; //update translation
			Xnew = out_r_map * x_map + t_map.replicate(1, N); //update moving points
		
			float num = (Xnew - Xold).pow(2).colwise().sum().sum();
			float denom = (Xold - Xold.rowwise().mean().replicate(1, N)).pow(2).colwise().sum().sum();
			config_change = std::sqrt( num/denom );

			Xold = Xnew;
		}

		VectorXf FREmatrix(N);
		for (int i = 0; i < N; i++) {
			VectorXf D = w_map * (Xnew.col(i) - y_map.col(i));
			FREmatrix[i] = D.transpose() * D;
		}

		*outFRE = std::sqrt(FREmatrix.mean());
		*outN = n;
		}




}