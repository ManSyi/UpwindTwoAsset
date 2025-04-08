#pragma once
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <omp.h>
#include <vector>

typedef Eigen::ArrayXd het;
typedef Eigen::SparseMatrix<double> SpMat;



template <typename Derived>
bool
check_within_tols(const Eigen::DenseBase<Derived>& x1, const Eigen::DenseBase<Derived>& x2, const double& tols, double& eps)
{

	for (int i = 0, j = 0, rows = x1.rows(), cols = x1.cols(); j != cols; ++j)
	{
		for (i = 0; i != rows; ++i)
		{
			if (std::abs(x1(i, j) - x2(i, j)) >= tols)
			{
				eps = std::abs(x1(i, j) - x2(i, j));
				return 0;
			}
		}

	}
	return 1;
}

template <typename Derived>
bool
check_within_tols(const Eigen::DenseBase<Derived>& x1, const double& tols)
{

	for (int i = 0, j = 0, rows = x1.rows(), cols = x1.cols(); j != cols; ++j)
	{
		for (i = 0; i != rows; ++i)
		{
			if (std::abs(x1(i, j)) >= tols)
			{
				return 0;
			}
		}

	}
	return 1;
}

template <typename Derived>
void
find_stationary_dist(const Eigen::EigenBase<Derived>& trans, const double& tols,
	const int& maxiter, Eigen::VectorXd& dist, Eigen::VectorXd& dist_p)
{
	double eps;
	for (int n = 0; n != maxiter; ++n)
	{
		dist_p.noalias() = trans.derived() * dist;
		dist_p /= dist_p.sum();
		if (n % 10 == 0 && check_within_tols(dist_p, dist, tols, eps))
		{
			goto end;
		}
		std::swap(dist_p, dist);
	}

	std::cout << "Couldn't find the stationary distribution after " << maxiter << " iterations!!" << std::endl;
	std::exit(0);
end:
	std::swap(dist_p, dist);
}

template <typename Derived>
void
find_stationary_dist(const Eigen::EigenBase<Derived>& trans, const double& tols,
	const int& maxiter, Eigen::VectorXd& dist)
{
	Eigen::VectorXd dist_p(dist.rows());
	double eps;
	for (int n = 0; n != maxiter; ++n)
	{
		dist_p.noalias() = trans.derived() * dist.matrix();
		dist_p /= dist_p.sum();
		if (n % 10 == 0 && check_within_tols(dist_p, dist, tols, eps))
		{
			goto end;
		}
		std::swap(dist_p, dist);
	}
	std::cout << "Couldn't find the stationary distribution after " << maxiter << " iterations!!" << std::endl;
	std::exit(0);
end:
	std::swap(dist_p, dist);
}

template <typename Derived>
void
set_grid(const int& n, const double& low, const double& high,
	const double& param, Eigen::DenseBase<Derived> const& grid)
{
	Eigen::DenseBase<Derived>& grid_ref = const_cast<Eigen::DenseBase<Derived>&>(grid);
	grid_ref = het::LinSpaced(n, 0.0, 1.0);
	grid_ref = low + (high - low)  * grid_ref.derived().pow(1.0 / param);
}

template <typename Derived>
void
set_exp_grid(const int& n, const double& low, const double& high,
	Eigen::DenseBase<Derived> const& grid, int times = 3)
{
	Eigen::DenseBase<Derived>& grid_ref = const_cast<Eigen::DenseBase<Derived>&>(grid);

	double grid_max = high - low;
	int i = 0;
	for (i = 0; i < times; ++i)
	{
		grid_max = std::log(grid_max + 1);
	}
	grid_ref = het::LinSpaced(n, 0, grid_max);
	for (i = 0; i < times; ++i)
	{
		grid_ref = grid_ref.derived().exp() - 1;
	}
	grid_ref.derived() += low;
}

template<typename Derived>
void
locate_dist(const Eigen::DenseBase<Derived>& xgrid, const double& xa, const int& limit, int& loc)
{
	if (xgrid(0) >= xa)
	{
		loc = 0;
	}
	while (loc < limit)
	{
		if (xgrid(1 + loc) >= xa)
		{
			++loc;
			break;
		}
		++loc;
	}
}

template<typename Derived>
void
locate_lower_bound(const Eigen::DenseBase<Derived>& xgrid, const double& xa, const int& limit, int& loc)
{
	while (loc < limit)
	{
		if (xgrid(1 + loc) > xa)
		{
			break;
		}
		++loc;
	}

}

template<typename Derived>
void
locate_lower_bound(const Eigen::DenseBase<Derived>& xgrid, const double& xa, const int& limit, int& loc, double& weight)
{
	locate_lower_bound(xgrid, xa, limit, loc);
	weight = (xgrid(loc + 1) - xa) / (xgrid(loc + 1) - xgrid(loc));
}




template <typename Derived_grid, typename Derived_dist>
double
probability(const Eigen::DenseBase<Derived_grid>& dist_grid, const Eigen::DenseBase<Derived_dist>& dist,
	const double& xa)
{
	const int limit = dist_grid.rows() - 2;

	if (dist_grid(0) >= xa)
	{
		return dist(0);
	}

	int i = 0;
	double w = 0.0;
	locate_lower_bound(dist_grid, xa, limit, i, w);
	w = (std::max)(w, 0.0);
	w = (std::min)(w, 1.0);
	double fx1 = dist.segment(0, i + 1).sum();
	double fx2 = fx1 + dist(i + 1);
	//return w * fx1 + (1 - w) * dist.segment(0, i + 2).sum();
	return w * fx1 + (1 - w) * fx2;
}

template <typename Derived_grid1, typename Derived_grid2, typename Derived_dist>
double
probability(const Eigen::DenseBase<Derived_grid1>& a_dist_grid, const Eigen::DenseBase<Derived_grid2>& b_dist_grid,
	const Eigen::DenseBase<Derived_dist>& dist, const double& xa, const double& xb)
{
	if (xa <= a_dist_grid(0) && xb <= b_dist_grid(0))
	{
		return dist(0, 0);
	}
	if (xa <= a_dist_grid(0))
	{
		return probability(b_dist_grid, dist.row(0), xb);
	}
	if (xb <= b_dist_grid(0))
	{
		return probability(a_dist_grid, dist.col(0), xa);
	}
	const int end_rindex = a_dist_grid.rows() - 2;
	const int end_cindex = b_dist_grid.rows() - 2;
	int i = 0;
	int j = 0;
	double w_i = 0.0;
	double w_j = 0.0;
	locate_lower_bound(a_dist_grid, xa, end_rindex, i, w_i);
	locate_lower_bound(b_dist_grid, xb, end_cindex, j, w_j); 
	w_i = (std::max)(w_i, 0.0);
	w_j = (std::max)(w_j, 0.0);
	w_i = (std::min)(w_i, 1.0);
	w_j = (std::min)(w_j, 1.0);
	double fx1 = dist.block(0, 0, i + 1, j + 1).sum();
	double fx2 = fx1 + dist.col(j + 1).segment(0, i + 1).sum();
	double fx3 = fx1 + dist.row(i + 1).segment(0, j + 1).sum();
	double fx4 = fx2 + fx3 + dist(i + 1, j + 1);
	//return w_i * w_j * dist.block(0, 0, i + 1, j + 1).sum()
	//	+ w_i * (1 - w_j) * dist.block(0, 0, i + 1, j + 2).sum()
	//	+ (1 - w_i) * w_j * dist.block(0, 0, i + 2, j + 1).sum()
	//	+ (1 - w_i) * (1 - w_j) * dist.block(0, 0, i + 2, j + 2).sum();

	return w_i * w_j * fx1 + w_i * (1 - w_j) * fx2 
		+ (1 - w_i) * w_j * fx3 + (1 - w_i) * (1 - w_j) * fx4;
}

template <typename Derived>
std::ofstream&
write_matrix_to_file(const Eigen::EigenBase<Derived>& mat, std::ofstream& of)
{
	int rows = mat.rows();
	int cols = mat.cols();

	int j = 0;
	for (int i = 0; i < rows; ++i)
	{
		for (j = 0; j < cols - 1; ++j)
		{
			of << mat.derived()(i, j) << ",";
		}
		of << mat.derived()(i, j);
		of << "\n";
	}
	return of;
}

template <typename Derived, typename File>
std::ofstream&
write_matrix_to_file(const Eigen::EigenBase<Derived>& mat, std::ofstream& of, const File& filename)
{
	of.open(filename, std::ios::trunc | std::ios::out);
	if (of)
	{
		write_matrix_to_file(mat, of).close();
	}
	else
	{
		std::cerr << "Couldn't open " << filename << "!" << std::endl;
		std::exit(0);
	}
	return of;
}

template <typename T>
bool
write_map_matrix_to_files(std::ofstream& of, const T& map,
	const std::string& fold)
{
	std::string filename;
	auto output_iter = map.begin();
	for (output_iter = map.begin(); output_iter != map.end(); ++output_iter)
	{
		filename = fold + "/" + output_iter->first + ".csv";
		write_matrix_to_file(output_iter->second, of, filename);
	}
	return 1;
}

template <typename T>
bool
write_map_tensor_to_files(std::ofstream& of, const T& map,
	const std::string& fold)
{
	std::string filename;
	auto output_iter = map.begin();
	for (output_iter = map.begin(); output_iter != map.end(); ++output_iter)
	{
		filename = fold + "/" + output_iter->first + ".csv";
		write_tensor_to_file(of, output_iter->second, filename);
	}
	return 1;
}


template <typename Derived>
std::ifstream&
read_matrix_from_file(const Eigen::EigenBase<Derived>& mat_, std::ifstream& ifs)
{
	Eigen::EigenBase<Derived>& mat = const_cast<Eigen::EigenBase<Derived>&>(mat_);
	std::string line;
	std::istringstream sin;
	std::string data;
	int row = 0;
	int col = 0;
	while (std::getline(ifs, line))
	{
		col = 0;
		sin.clear();
		sin.str(line);
		data.clear();
		while (std::getline(sin, data, ','))
		{
			mat.derived()(row, col++) = std::stod(data);
		}
		++row;
	}
	return ifs;
}

template <typename Derived, typename File>
std::ifstream&
read_matrix_from_file(const Eigen::EigenBase<Derived>& mat_, std::ifstream& ifs, const File& filename)
{
	ifs.open(filename);
	if (ifs)
	{
		read_matrix_from_file(mat_, ifs).close();
	}
	else
	{
		std::cerr << "Couldn't open " << filename << "!" << std::endl;
		std::exit(0);
	}
	return ifs;
}

template <typename T>
bool
read_files_to_map_matrix(std::ifstream& ifs, const std::string& fold, int rows, int cols,
	std::map<std::string, T>& map_matrix)
{
	std::string filename;
	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator(fold))
	{
		if (dir_entry.is_regular_file())
		{
			filename = dir_entry.path().stem().string();
			map_matrix[filename].setZero(rows, cols);
			read_matrix_from_file(map_matrix[filename], ifs, dir_entry.path());
		}
	}
	return 1;
}




template <typename T>
void
construct_unordered_map_from_list(std::unordered_map<std::string, T>& unmap, const std::unordered_map<std::string, T>& list)
{
	for (auto& elem : list)
	{
		unmap.emplace(elem.first, elem.second);
	}
}
template <typename T>
std::set<std::string>
get_keys(const T& some_map)
{
	std::set<std::string> key_set;
	std::transform(some_map.begin(), some_map.end(),
		std::inserter(key_set, key_set.end()), [](auto pair) { return pair.first; });
	return key_set; 
}

template<typename Derived1, typename Derived2, typename Derived3, typename Derived4>
void
apply_linear_pairs(const Eigen::ArrayBase<Derived1>& index, const Eigen::ArrayBase<Derived2>& weight,
	const Eigen::ArrayBase<Derived3>& ygrid, const Eigen::ArrayBase<Derived4>& ya_)
{
	Eigen::ArrayBase<Derived4>& ya = const_cast<Eigen::ArrayBase<Derived4>&>(ya_);
	const int cols = ya.cols();
	if (ygrid.rows() > 1)
	{
		for (int m = 0; m < cols; ++m)
		{
			ya.col(m) = weight.col(m) * ygrid(index.col(m), m) + (1.0 - weight.col(m)) * ygrid(index.col(m) + 1, m);
		}
	}
	else if (ygrid.rows() == 1)
	{
		for (int m = 0; m < cols; ++m)
		{
			ya.col(m) = weight.col(m) * ygrid(index.col(m), m);
		}
	}
}



template<typename Derived>
void
check_increasing(const Eigen::ArrayBase<Derived>& xgrid)
{
	int size = xgrid.rows();
	if ((xgrid.segment(1, size - 1) - xgrid.segment(0, size - 1) <= 0).any())
	{
		het tar = xgrid;
		std::cerr << "No increasing in grid!!" << std::endl;
		std::exit(0);
	}
}

template<typename Derived1, typename Derived2, typename Derived3, typename Derived4>
void
mono_interpolate_pairs(const Eigen::ArrayBase<Derived1>& xgrid, const Eigen::ArrayBase<Derived2>& xa,
	const Eigen::ArrayBase<Derived3>& lower_index_, const Eigen::ArrayBase<Derived4>& weight_)
{
	const int rows = xa.rows();
	const int cols = xa.cols();
	const int index_limit = xgrid.rows() - 2;
	Eigen::ArrayBase<Derived3>& lower_index = const_cast<Eigen::ArrayBase<Derived3>&>(lower_index_);
	Eigen::ArrayBase<Derived4>& weight = const_cast<Eigen::ArrayBase<Derived4>&>(weight_);
	int k = 0;
	if (index_limit > -1)
	{
		for (int m = 0; m < cols; ++m)
		{
			lower_index(0, m) = 0;

			locate_lower_bound(xgrid.col(m), xa(0, m), index_limit, lower_index(0, m), weight(0, m));
			for (k = 1; k != rows; ++k)
			{
				lower_index(k, m) = lower_index(k - 1, m);
				//lower_index(k, m) = 0;
				locate_lower_bound(xgrid.col(m), xa(k, m), index_limit, lower_index(k, m), weight(k, m));
			}
		}
	}
	else if(index_limit == -1)
	{
		lower_index.setZero();
		weight.setOnes();
	}
}

template<typename Derived1, typename Derived2, typename Derived3, typename Derived4>
void
nonmono_interpolate_pairs(const Eigen::ArrayBase<Derived1>& xgrid, const Eigen::ArrayBase<Derived2>& xa,
	const Eigen::ArrayBase<Derived3>& lower_index_, const Eigen::ArrayBase<Derived4>& weight_)
{
	const int rows = xa.rows();
	const int cols = xa.cols();
	const int index_limit = xgrid.rows() - 2;
	Eigen::ArrayBase<Derived3>& lower_index = const_cast<Eigen::ArrayBase<Derived3>&>(lower_index_);
	Eigen::ArrayBase<Derived4>& weight = const_cast<Eigen::ArrayBase<Derived4>&>(weight_);
	int k = 0;
	if (index_limit > -1)
	{
		for (int m = 0; m < cols; ++m)
		{
			for (k = 0; k != rows; ++k)
			{
				lower_index(k, m) = 0;
				locate_lower_bound(xgrid.col(m), xa(k, m), index_limit, lower_index(k, m), weight(k, m));
			}
		}
	}
	else if (index_limit == -1)
	{
		lower_index.setZero();
		weight.setOnes();
	}
}


template<typename Derived1, typename Derived2>
double
linear_interpolate(const Eigen::DenseBase<Derived1>& xgrid, const Eigen::DenseBase<Derived2>& ygrid, const double& xa)
{
	int i = 0;
	int limit = xgrid.rows() - 2;
	locate_lower_bound(xgrid, xa, limit, i);
	return linear_interp_value(xgrid(i), xgrid(i + 1), ygrid(i), ygrid(i + 1), xa);
}

template<typename Derived1, typename Derived2>
double
linear_interpolate(const Eigen::DenseBase<Derived1>& xgrid, const Eigen::DenseBase<Derived2>& ygrid, const double& xa, int& i)
{
	// extra-interpolate at both side.
	int limit = xgrid.rows() - 2;
	locate_lower_bound(xgrid, xa, limit, i); 
	return linear_interp_value(xgrid(i), xgrid(i + 1), ygrid(i), ygrid(i + 1), xa);
}


template<typename Derived1>
double
quantile_share(const Eigen::VectorXd& cdf, const Eigen::DenseBase<Derived1>& wealth_frac, double expectation, double prob)
{
	int loc = 0;
	double weight = 0;
	locate_lower_bound(cdf, prob, cdf.rows() - 2, loc, weight);
	weight = (std::min)(weight, 1.0);
	weight = (std::max)(weight, 0.0);
	return (wealth_frac.segment(0, loc + 1).sum() * weight
		+ wealth_frac.segment(0, loc + 2).sum() * (1.0 - weight)) / expectation;
}

template <typename Derived1, typename Derived2, typename Derived3>
void
Adjust_dist_rowwise(const Eigen::ArrayXd& grid, double adj,
	const Eigen::DenseBase<Derived1>& dist,
	const Eigen::DenseBase<Derived2>& cdf_next_,
	const Eigen::DenseBase<Derived3>& dist_next_)
{
	Eigen::DenseBase<Derived2>& cdf_next = const_cast<Eigen::DenseBase<Derived2>&>(cdf_next_);
	Eigen::DenseBase<Derived3>& dist_next = const_cast<Eigen::DenseBase<Derived3>&>(dist_next_);
	int rows = dist.rows();
	int cols = dist.cols();
	int i = 0;
	int j = 0;
	double mean1 = 0.0;
	double mean2 = 0.0;
	for (j = 0; j < cols; ++j)
	{
		mean1 = (grid.array() * dist.col(j).array()).sum() / dist.col(j).sum();
		for (i = 0; i < rows; ++i)
		{
			cdf_next(i, j) = probability(grid, dist.col(j).array() / dist.col(j).sum(), grid(i) / adj);
			if (cdf_next(i, j) > 1.0) cdf_next(i, j) = 1.0;
		}
		cdf_next(rows - 1, j) = 1.0;
		dist_next(0, j) = cdf_next(0, j);
		dist_next.col(j).segment(1, rows - 1) = cdf_next.col(j).segment(1, rows - 1) - cdf_next.col(j).segment(0, rows - 1);

		mean2 = (grid.array() * dist_next.col(j).array()).sum();

		dist_next.col(j) *= adj * mean1 / mean2;
		dist_next(0, j) = 1.0 - dist_next.col(j).segment(1, rows - 1).sum();
		dist_next.col(j) *= dist.col(j).sum();
	}

}
