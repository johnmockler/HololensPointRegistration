#pragma once

#include <Eigen/Dense>
#include <limits>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

class PointRegister
{

public:
	PointRegister(MatrixXf Xin, MatrixXf Yin, int Num_points);
	PointRegister(MatrixXf Xin, MatrixXf Yin, int Num_points, MatrixXf Win);
	int getN_Iter();
	MatrixXf getR();
	MatrixXf getT();
	float solveIsotropic();
	float solveAnisotropic(float threshold = 0.1f);

private:
	MatrixXf R;
	MatrixXf T;
	MatrixXf W;
	MatrixXf X;
	MatrixXf Y;
	int N;
	int n_iter = 0;
	void c_maker(Ref<MatrixXf> C);
	void e_maker(Ref<MatrixXf> e);
	void initialize(MatrixXf Xin, MatrixXf Yin, int NumPoints);

};

