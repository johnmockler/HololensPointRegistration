#include <Eigen/Dense>
#include <limits>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

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

float register_isotropic(Ref<MatrixXf> X, Ref<MatrixXf> Y, int N, Ref<MatrixXf> R, Ref<MatrixXf> T) {
	Vector3f x_mean = X.rowwise().mean();
	Vector3f y_mean = Y.rowwise().mean();

	Vector3f x_tilde = X - x_mean.replicate(1, N); //x shifted to origin
	Vector3f y_tilde = Y = y_mean.replicate(1, N); //y shifted to origin

	MatrixXf H = x_tilde * y_tilde.transpose(); //cross covariance matrix

	JacobiSVD<MatrixXf> svd(H, ComputeFullU | ComputeFullV);

	R = svd.matrixV() * svd.matrixU().transpose(); //why does the fitzpatrick paper have a diag(1, 1, det(V*U))? its not in the arun paper
	T = y_mean - R * x_mean;

	Vector3f FREvect = R * X - T.replicate(1, N) - Y;
	return std::sqrt(FREvect.pow(2).rowwise().sum().mean());
}
