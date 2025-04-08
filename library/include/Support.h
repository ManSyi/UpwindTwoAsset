#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <iomanip>

typedef Eigen::ArrayXd het;

typedef double parameter; typedef double aggregates;

typedef Eigen::SparseMatrix<double> SpMat;

typedef Eigen::ArrayXXd het2;

typedef std::vector<het2> Tensor;

typedef Eigen::ArrayXXd Grid;
typedef Eigen::ArrayXd Seq;
typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> ArrayXXb;

typedef std::vector<ArrayXXb> Bool;


std::ofstream&
write_tensor_to_file(std::ofstream& of, const Tensor& ten);

double
logistic(const double& y, const double& xhigh, const double& xlow, const double& xinit, double curtrure = 0.1);

double
invlogistic(const double& x, const double& xhigh, const double& xlow, const double& xinit, double curtrure = 0.1);

double
linear_interp_value(const double& weight, const double& y1, const double& y2);

double
linear_interp_value(const double& x1, const double& x2, const double& y1, const double& y2,
	const double& xa);

double
calculate_variance(const het& dist, const het& grid);

std::ostream&
print_set(std::ostream&, const std::set<std::string>&);

std::ofstream&
write_set_to_file(std::ofstream& of, const std::set<std::string>& someSet, const std::string& filename);


std::ofstream&
write_set_to_file(std::ofstream&, const std::set<std::string>&);

void 
init_empty_tensor(std::vector<Eigen::ArrayXXi>& ten, int dim0, int dim1, int dim2);

void
init_empty_tensor(Tensor& ten, int dim0, int dim1, int dim2);

void 
init_empty_tensor(Tensor& ten, int dim0, int dim1);

void 
init_empty_tensor(Bool& ten, int dim0, int dim1, int dim2);

void
init_empty_tensor(std::vector<SpMat>& ten, int dim0, int dim1, int dim2);

void
set_zero_tensor(Tensor& ten);

void
set_grid_auclert(double min, double max, int n, het& grid);

void
vectorize_tensor(const Tensor& ten, Eigen::VectorXd& vec);

double
sign(const double& x);

double
norm_cdf(const double& mu, const double& sigma, const double& x);

double
norm_cdf(const double& x);

//bool
//write_map_matrix_to_files(std::ofstream& of, const std::map<std::string, Eigen::MatrixXd>& map,
//	const std::string& fold);

bool
write_map_map_matrix_to_files(std::ofstream& of, const std::map<std::string, std::map<std::string, Eigen::MatrixXd>>& map,
	const std::string& fold);




template <typename T>
std::ostream&
print_map(std::ostream& os, const std::unordered_map<std::string, T>& someMap)
{
	for (auto& element : someMap) {
		os << element.first << ": " << element.second << "\n";
	}
	return os;
}

template <typename T>
std::ofstream&
write_map_to_file(std::ofstream& of, const std::unordered_map<std::string, T>& someMap)
{
	for (auto& element : someMap) {
		of << element.first << "," << element.second << "\n";
	}
	return of;
}

template <typename T>
std::ofstream&
write_map_to_file(std::ofstream& of, const std::unordered_map<std::string, T>& someMap, const std::string& filename)
{
	of.open(filename, std::ios::trunc | std::ios::out);
	if (of)
	{
		write_map_to_file(of, someMap).close();
	}
	else
	{
		std::cerr << "Couldn't open " << filename << "!" << std::endl;
		std::exit(0);
	}
	return of;
}



std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, bool>& someMap);

std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, int>& someMap);

std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, double>& someMap);

std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, int>& someMap, const std::string& filename);

std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, bool>& someMap, const std::string& filename);

std::ofstream&
write_tensor_to_file(std::ofstream& of, const Tensor& ten, const std::string& filename);

std::ifstream&
read_tensor_from_file(std::ifstream& ifs, Tensor& ten);

void
rouwenhorst_AR1(const double& rho, const double& sigma, const int& nsk, Eigen::MatrixXd& markov,
	Eigen::ArrayXd& grid);

template <typename File>
std::ifstream&
read_tensor_from_file(std::ifstream& ifs, Tensor& ten, const File& filename)
{
	ifs.open(filename);
	if (ifs)
	{
		read_tensor_from_file(ifs, ten).close();
	}
	else
	{
		std::cout << "Couldn't open " << filename << "!" << std::endl;
		std::exit(0);
	}
	return ifs;
}

template <typename File>
std::ifstream&
read_map_from_file(std::ifstream& ifs, std::unordered_map<std::string, double>& someMap, const File& filename)
{
	ifs.open(filename);
	if (ifs)
	{
		read_map_from_file(ifs, someMap).close();
	}
	else
	{
		std::cout << "Couldn't open " << filename << "!" << std::endl;
		std::exit(0);
	}
	return ifs;
}

template <typename T>
inline T&
get_value(std::unordered_map<std::string, T>& m, const std::string& s)
{
	return m.find(s)->second;
}

template <typename T>
inline const T&
get_value(const std::unordered_map<std::string, T>& m, const std::string& s)
{
	return m.find(s)->second;
}

template <typename T>
inline T&
get_value(std::map<std::string, T>& m, const std::string& s)
{
	return m.find(s)->second;
}



template <typename T>
inline const T&
get_value(const std::map<std::string, T>& m, const std::string& s)
{
	return m.find(s)->second;
}



template <typename T>
void
renew_map(const T& news, T& old)
{
	for (auto& elem : news)
	{
		if (old.find(elem.first) != old.end()) old[elem.first] = elem.second;
	}
}

template <typename T1, typename T2, typename T3>
void
print_one_root_update(const T1& iter, const T2& var, const T3& error)
{
	std::cout << std::setw(16) << std::left << iter
		<< std::setw(16) << std::left << var
		<< std::setw(16) << std::left << error << std::endl;
}

template <typename T>
void
append_map(const T& news, T& old)
{
	for (auto& elem : news)
	{
		old[elem.first] = elem.second;
	}
}

// visitor
template<typename Func>
struct lambda_as_visitor_wrapper : Func {
	lambda_as_visitor_wrapper(const Func& f) : Func(f) {}
	template<typename S, typename I>
	void init(const S& v, I i, I j) { return Func::operator()(v, i, j); }
};

template<typename Mat, typename Func>
void visit_lambda(const Mat& m, const Func& f)
{
	lambda_as_visitor_wrapper<Func> visitor(f);
	m.visit(visitor);
}

// indexing
template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
	const ArgType& m_arg;
	const RowIndexType& m_rowIndices;
	const ColIndexType& m_colIndices;
public:
	typedef Eigen::Matrix<typename ArgType::Scalar,
		RowIndexType::SizeAtCompileTime,
		ColIndexType::SizeAtCompileTime,
		ArgType::Flags& Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
		RowIndexType::MaxSizeAtCompileTime,
		ColIndexType::MaxSizeAtCompileTime> MatrixType;

	indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
		: m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
	{}

	const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
		return m_arg(m_rowIndices[row], m_colIndices[col]);
	}
};


template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType, RowIndexType, ColIndexType>, typename indexing_functor<ArgType, RowIndexType, ColIndexType>::MatrixType>
mat_indexing(const Eigen::DenseBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
{
	typedef indexing_functor<ArgType, RowIndexType, ColIndexType> Func;
	typedef typename Func::MatrixType MatrixType;
	return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

template <typename Derived>
class logical
{
private:
	const Eigen::Index new_size;
	Eigen::Array<Eigen::Index, Eigen::Dynamic, 1> old_inds;

public:
	logical(const Eigen::ArrayBase<Derived>& keep) : new_size(keep.count()), old_inds(new_size)
	{
		for (Eigen::Index i = 0, j = 0; i < keep.size(); i++)
			if (keep(i))
				old_inds(j++) = i;
	}
	Eigen::Index size() const { return new_size; }
	Eigen::Index operator[](Eigen::Index new_ind) const { return old_inds(new_ind); }
};

#include "support.hpp"