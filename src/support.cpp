#define EIGEN_USE_MKL_ALL
#include <map>
#include <string>
#define _USE_MATH_DEFINES
#include <cmath>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <set>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <omp.h>

#include "Calibration.h"
#include "Support.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

ostream&
print_set(ostream& os, const set<string>& someSet)
{
	for (auto& element : someSet)
	{
		os << element << "; ";
	}
	os << "\n";
	return os;
}

void
infer_matrix_size_from_files(std::ifstream& ifs)
{
	int rows = 0;
	int cols = 0;

	std::string line;
	std::istringstream sin;
	std::string data;

	while (std::getline(ifs, line))
	{
		cols = 0;
		sin.clear();
		sin.str(line);
		data.clear();
		while (std::getline(sin, data, ','))
		{
			++cols;
		}
		++rows;
	}
}

ofstream&
write_tensor_to_file(ofstream& of, const Tensor& ten, const std::string& filename)
{
	of.open(filename, std::ios::trunc | std::ios::out);
	if (of)
	{
		write_tensor_to_file(of, ten).close();
	}
	else
	{
		std::cerr << "Couldn't open " << filename << "!" << std::endl;
		std::exit(0);
	}
	return of;
}

ofstream&
write_tensor_to_file(ofstream& of, const Tensor& ten)
{

	int dim2 = ten.size();

	for (int r = 0; r < dim2; ++r)
	{
		write_matrix_to_file(ten[r], of);
	}
	return of;
}

std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, bool>& someMap, const std::string& filename)
{
	ifs.open(filename);
	if (ifs)
	{
		read_map_from_file(ifs, someMap).close();
	}
	else
	{
		std::cout << "Couldn't open " << filename << "!" << endl;
		std::cout << filesystem::current_path() << endl;
		exit(0);
	}
	return ifs;
}

std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, bool>& someMap)
{
	std::string line;
	std::istringstream sin;
	std::string key;
	std::string value;
	while (std::getline(ifs, line))
	{
		sin.clear();
		sin.str(line);
		key.clear();
		value.clear();
		std::getline(sin, key, ',');
		std::getline(sin, value, '\n');
		someMap.emplace(key, std::stoi(value) != 0);
	}
	return ifs;
}

std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, int>& someMap, const std::string& filename)
{
	ifs.open(filename);
	if (ifs)
	{
		read_map_from_file(ifs, someMap).close();
	}
	else
	{
		std::cout << "Couldn't open " << filename <<"!" << endl;
		std::exit(0);
	}
	return ifs;
}

void
rouwenhorst_AR1(const double& rho, const double& sigma, const int& nsk, Eigen::MatrixXd& markov, 
	Eigen::ArrayXd& grid)
{
	const double p = (1.0 + rho) / 2.0;
	Eigen::MatrixXd m1(nsk, nsk);
	Eigen::MatrixXd m2(nsk, nsk);
	Eigen::MatrixXd m3(nsk, nsk);
	Eigen::MatrixXd m4(nsk, nsk);
	markov.setZero(2, 2);
	markov << p, 1.0 - p,
		1.0 - p, p;
	for (int n = 3; n != nsk + 1; ++n)
	{
		m1.setZero(n, n);
		m2.setZero(n, n);
		m3.setZero(n, n);
		m4.setZero(n, n);

		m1.block(0, 0, n - 1, n - 1) = p * markov;
		m2.block(0, 1, n - 1, n - 1) = (1 - p) * markov;
		m3.block(1, 0, n - 1, n - 1) = (1 - p) * markov;
		m4.block(1, 1, n - 1, n - 1) = p * markov;
		markov = m1 + m2 + m3 + m4;

		markov.block(1, 0, n - 2, n) /= 2.0;
	}

	/* Rouwenhorst method to discretize grid. */
	grid.resize(nsk);
	grid = het::LinSpaced(nsk, -sqrt(nsk - 1) * sigma, sqrt(nsk - 1) * sigma);

}

std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, int>& someMap)
{
	std::string line;
	std::istringstream sin;
	std::string key;
	std::string value;
	while (std::getline(ifs, line))
	{
		sin.clear();
		sin.str(line);
		key.clear();
		value.clear();
		std::getline(sin, key, ',');
		std::getline(sin, value, '\n');
		someMap.emplace(key, std::stoi(value));
	}
	return ifs;
}



std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, double>& someMap)
{
	std::string line;
	std::istringstream sin;
	std::string key;
	std::string value;
	while (std::getline(ifs, line))
	{
		sin.clear();
		sin.str(line);
		key.clear();
		value.clear();
		std::getline(sin, key, ',');
		std::getline(sin, value, '\n');
		//someMap.emplace(key, std::stod(value));
		someMap[key] = std::stod(value);
	}
	return ifs;
}



ifstream&
read_tensor_from_file(ifstream& ifs, Tensor& ten)
{
	int dim0 = ten[0].rows();
	std::string line;
	std::istringstream sin;
	std::string data;
	int row = 0;
	int col = 0;
	int r = 0;
	while (std::getline(ifs, line))
	{
		col = 0;
		sin.clear();
		sin.str(line);
		data.clear();
		while (std::getline(sin, data, ','))
		{
			ten[r](row % dim0, col++) = std::stod(data);
		}
		r = ++row / dim0;
	}
	return ifs;
}

std::ofstream&
write_set_to_file(std::ofstream& of, const std::set<std::string>& someSet, const std::string& filename)
{
	of.open(filename, std::ios::trunc | std::ios::out);
	if (of)
	{
		write_set_to_file(of, someSet).close();
	}
	else
	{
		std::cerr << "Couldn't open " << filename << "!" << std::endl;
		std::exit(0);
	}
	return of;
}

ofstream&
write_set_to_file(ofstream& of, const set<string>& someSet)
{
	for (auto& key : someSet)
	{
		of << key << "\n";
	}
	return of;
}

double
norm_cdf(const double& mu, const double& sigma, const double& x)
{
	return 0.5 * (std::erfc(-(x - mu) / sigma * M_SQRT1_2));
}

double
norm_cdf(const double& x)
{
	return 0.5 * (std::erfc(-x* M_SQRT1_2));
}

double
invlogistic(const double& x, const double& xhigh, const double& xlow, const double& xinit, double curtrure)
{
	return 1.0 / curtrure * log((xhigh - x) / (xhigh - xinit) * (xinit - xlow) / (x - xlow));
}


double
logistic(const double& y, const double& xhigh, const double& xlow, const double& xinit, double curtrure)
{
	return xlow + (xhigh - xlow) / (1.0 + exp(curtrure * y) * (xhigh - xinit) / (xinit - xlow));
}

double
linear_interp_value(const double& x1, const double& x2, const double& y1, const double& y2,
	const double& xa)
{
	return (y2 - y1) / (x2 - x1) * (xa - x1) + y1;
}

double
sign(const double& x)
{
	if (x > 0)
	{
		return 1;
	}
	else if (x < 0)
	{
		return -1;
	}
	else return 0;
}

double
linear_interp_value(const double& weight, const double& y1, const double& y2)
{
	return y2 + weight * (y1 - y2);
}

void
set_grid_auclert(double min, double max, int n, het& grid)
{
	double shift = 0.25;
	grid = het::LinSpaced(n, log10(min + shift), log10(max + shift));
	grid = pow(10, grid) -  shift;
	grid(0) = min;

}

double
calculate_variance(const het& dist, const het& grid)
{
	return (dist * pow(grid - (dist * grid).sum(), 2.0)).sum();
}





void
set_zero_tensor(Tensor& ten)
{
	int size = ten.size();
#pragma omp parallel for
	for (int i = 0; i < size; ++i)
	{
		ten[i].setZero();
	}
}

void init_empty_tensor(Tensor& ten, int dim0, int dim1)
{

	int size = ten.size();
#pragma omp parallel for
	for (int i = 0; i < size; ++i)
	{
		ten[i].setZero(dim0, dim1);
	}
}

void init_empty_tensor(std::vector<Eigen::ArrayXXi>& ten, int dim0, int dim1, int dim2)
{
	ten.resize(dim2);
#pragma omp parallel for
	for (int i = 0; i < dim2; ++i)
	{
		ten[i].setZero(dim0, dim1);
	}
}


void init_empty_tensor(std::vector<SpMat>& ten, int dim0, int dim1, int dim2)
{
	ten.resize(dim2);
#pragma omp parallel for
	for (int i = 0; i < dim2; ++i)
	{
		ten[i].resize(dim0, dim1);
		ten[i].setZero();
	}
}



void init_empty_tensor(Tensor& ten, int dim0, int dim1, int dim2)
{
	ten.resize(dim2);
#pragma omp parallel for
	for (int i = 0; i < dim2; ++i)
	{
		ten[i].setZero(dim0, dim1);
	}
}

void init_empty_tensor(Bool& ten, int dim0, int dim1, int dim2)
{
	ten.resize(dim2);
#pragma omp parallel for
	for (int i = 0; i < dim2; ++i)
	{
		ten[i].resize(dim0, dim1);
	}
}

void
vectorize_tensor(const Tensor& ten, Eigen::VectorXd& vec)
{
	const int dim2 = ten.size();
	const int dim1 = ten[0].cols();
	const int dim0 = ten[0].rows();
	const int size = dim0 * dim1;

	vec.resize(size * dim2);

#pragma omp parallel for
	for (int r = 0; r < dim2; ++r)
	{
		vec.segment(size * r, size) = ten[r].reshaped();
	}
}

//bool
//write_map_matrix_to_files(std::ofstream& of, const std::map<std::string, Eigen::MatrixXd>& map,
//	const std::string& fold)
//{
//	std::string filename;
//	auto output_iter = map.begin();
//	for (output_iter = map.begin(); output_iter != map.end(); ++output_iter)
//	{
//		filename = fold + "/" + output_iter->first + ".csv";
//		write_matrix_to_file(output_iter->second, of, filename);
//	}
//	return 1;
//}

bool
write_map_map_matrix_to_files(ofstream& of, const std::map<std::string, std::map<std::string, Eigen::MatrixXd>>& map,
	const string& fold)
{
	string sfold;
	string filename;
	auto output_iter = map.begin();
	auto input_iter = map.begin()->second.begin();
	for (output_iter = map.begin(); output_iter != map.end(); ++output_iter)
	{
		sfold = fold + "/" + output_iter->first;
		filesystem::create_directories(sfold);
		auto& input_jacs = output_iter->second;
		for (input_iter = input_jacs.begin(); input_iter != input_jacs.end(); ++input_iter)
		{
			filename = sfold + "/" + output_iter->first + "-" + input_iter->first + ".csv";
			write_matrix_to_file(input_iter->second, of, filename);
		}
	}
	return 1;
}

