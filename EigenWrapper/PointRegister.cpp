
#include "PointRegister.h"
using namespace Eigen;


PointRegister::PointRegister(MatrixXf Xin, MatrixXf Yin, int num_points) {
	initialize(Xin, Yin, num_points);
	W.setIdentity(3, 3);

}

PointRegister::PointRegister(MatrixXf Xin, MatrixXf Yin, int num_points, MatrixXf Win) {
	initialize(Xin, Yin, num_points);
	W = Win;
}

void PointRegister::c_maker(Ref<MatrixXf> C)
{
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

void PointRegister::e_maker(Ref<MatrixXf> e)
{
	MatrixXf w1 = W.col(1).replicate(1, N);
	MatrixXf w2 = W.col(2).replicate(1, N);
	MatrixXf w3 = W.col(3).replicate(1, N);
	MatrixXf D = Y - X;
	MatrixXf d1 = D.row(1).replicate(3, 1);
	MatrixXf d2 = D.row(2).replicate(3, 1);
	MatrixXf d3 = D.row(3).replicate(3, 1);
	e << w1.cwiseProduct(d1) + w2.cwiseProduct(d2) + w3.cwiseProduct(d3);

	e.resize(6, 1);
}

int PointRegister::getN_Iter()
{
	return n_iter;
}

MatrixXf PointRegister::getR()
{
	return R;
}
MatrixXf PointRegister::getT()
{
	return T;
}

void PointRegister::initialize(MatrixXf Xin, MatrixXf Yin, int num_points)
{
	X = Xin;
	Y = Yin;
	N = num_points;
	R.setIdentity(3, 3);
	T.setZero(3, 1);
}

float PointRegister::solveIsotropic()
{
	
	MatrixXf x_mean = X.rowwise().mean();
	MatrixXf y_mean = Y.rowwise().mean();
	
	MatrixXf x_tilde = X - x_mean.replicate(1, N); //x shifted to origin
	MatrixXf y_tilde = Y - y_mean.replicate(1, N); //y shifted to origin
	
	MatrixXf H = x_tilde * y_tilde.transpose(); //cross covariance matrix
	
	JacobiSVD<MatrixXf> svd(H, ComputeFullU | ComputeFullV);
	
	

	R = svd.matrixV() * svd.matrixU().transpose(); //why does the fitzpatrick paper have a diag(1, 1, det(V*U))? its not in the arun paper
	T = y_mean - R * x_mean;

	MatrixXf FREvect = R * X + T.replicate(1, N) - Y;
	
	float FRE = std::sqrt(FREvect.pow(2).colwise().sum().mean());


	return FRE;
}

float PointRegister::solveAnisotropic(float threshold)
{
	/*
	X is the moving set, which is registered to the static set Y. Both are 3
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
	
	float iso_fre = solveIsotropic();

	//if there is no, or minimal error, then there's no need to loop
	/*
	if (iso_fre < threshold) {
		return iso_fre;
	}*/

	int n = 0;
	int index = 0;
	float config_change = threshold + 1.0f;
	MatrixXf Xold = R * X + T.replicate(1, N);
	MatrixXf Xnew;
	Vector3f oldq;

	while (config_change > threshold) {
		if (n > MAX_ITERATIONS) {
			break;
		}
		
		n = n + 1;

		MatrixXf C;
		c_maker(C);

		MatrixXf e;
		e_maker(e);

		Vector3f q = C.lu().solve(e);

		if (n > 1) {
			q = (q + oldq) / 2.0f; //damps osccilations
		}
		oldq = q;
		
		Vector3f delta_t;
		delta_t << q(4), q(5), q(6);
		delta_t.transpose();
		
		Matrix3f delta_theta;
		delta_theta << 1, -q(3), q(2),
			q(3), 1, -q(1),
			-q(2), q(1), 1;
		
		JacobiSVD<Matrix3f> svd(delta_theta, ComputeFullU | ComputeFullV);

		Matrix3f delta_R = svd.matrixU() * svd.matrixV().transpose();

		R = delta_R * R; //update rotation
		T = delta_R * T + delta_t; //update translation
		Xnew = R * X + T.replicate(1, N); //update moving points

		float num = (Xnew - Xold).pow(2).colwise().sum().sum();
		float denom = (Xold - Xold.rowwise().mean().replicate(1, N)).pow(2).colwise().sum().sum();
		config_change = std::sqrt(num / denom);

		Xold = Xnew;


	}

	n_iter = n;

	VectorXf FREmatrix(N);
	for (int i = 0; i < N; i++) {
		VectorXf D = W * (Xnew.col(i) - Y.col(i));
		FREmatrix[i] = D.transpose() * D;
	}

	return std::sqrt(FREmatrix.mean());
}






