#pragma once
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>



#include "Het_block.h"

typedef Eigen::ArrayXXd het2;
typedef Eigen::SparseMatrix<double> SpMat;
typedef std::vector<het2> Tensor;
class SteadyState;
struct Het_Outputs;
struct Het_Inputs;
struct Het_workspace;
typedef Eigen::ArrayXXd Grid;


namespace TA
{


	bool
		hour_secant(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, Het_workspace& ws);

	bool
		hour_bisec(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, Het_workspace& ws);


	void
		fhour_supply(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, const double& hour, Het_workspace& ws, double& fhour);

	void
		hour_min(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, Het_workspace& ws);

	void
		fhour_foc(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, const Het_workspace& ws, const double& hour, double& fhour);

	void
		solve_convergent_pols(const Het_Outputs& target_params, const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		solve_targeting_moments(const Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp);

	void
		convert_pols_map(const Het_Inputs& het_inputs, Het_workspace& ws, std::unordered_map<std::string, Tensor>& pols_map);
	
	void
		solve_cdf(const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		solve_labor(const Het_Inputs& het_inputs, Het_workspace& ws);



	void
		solve_dist_stat(const Het_Inputs& het_inputs, Het_Outputs& htp, Het_workspace& ws, std::unordered_map<std::string, double>& map);

	void
		solve_res_pols(const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		swap_het_vecs(Het_workspace& ws, std::unordered_map<std::string, Eigen::MatrixXd>& map);

	void
		swap_het_dist(Het_workspace& ws, SteadyState& ss);

	void
		solve_het_one_step(const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		target_HtM(const Calibration& cal, Het_Outputs& htp, const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		solve_convergent_dist(const Het_Outputs& target_params, const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		derivative(const Het_Inputs& het_inputs, const Eigen::MatrixXd& V, het2& VaF, het2& VaB, het2& VbF, het2& VbB, int r);

	void
		solve_deposit(const Het_Inputs& het_inputs, const het2& Va, const het2& Vb,
			het2& d, het2& Hd);


	void
		solve_het_one_step(const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		solve_HJB(const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		solve_KFE(const Het_Inputs& het_inputs, Het_workspace& ws);

	void
		solve_rhs(const Het_Inputs& het_inputs, Het_workspace& ws, int r);

	void
		construct_distKFE(const Het_Inputs& het_inputs, Het_workspace& ws, const std::vector<SpMat>& assetHJB);

	void
		construct_final_assetHJB(const Het_Inputs& het_inputs, Het_workspace& ws,
			const Tensor& sa, const Tensor& sb);

	void
		construct_assetHJB(const Het_Inputs& het_inputs, Het_workspace& ws, int r);

	void
		print_parameter(const Het_Inputs& het_inputs, const Het_workspace& ws);

	void
		construct_transpose_distKFE(const Het_Inputs& het_inputs, Het_workspace& ws, const std::vector<SpMat>& assetHJB);


	namespace Adj_fun_auclert
	{
		template <typename Derived1>
		auto
			adj(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived1>& d_abs)
		{
			return het_inputs.chi1 / het_inputs.chi2
				* (d_abs / (het_inputs.agrid + het_inputs.a_kink)).pow(het_inputs.chi2)
				* (het_inputs.agrid + het_inputs.a_kink);
		}
		double
			adj(const Het_Inputs& het_inputs, const double& d_abs, int k, int m);

		template <typename Derived1>
		auto
			adj1(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived1>& d)
		{
			return het_inputs.chi1
				* (d.abs() / (het_inputs.agrid + het_inputs.a_kink)).pow(het_inputs.chi2 - 1)
				* d.sign();
		}
		double
			adj1(const Het_Inputs& het_inputs, const double& d, int k, int m);

		template <typename Derived1>
		auto
			adj1inv(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived1>& ratio)
		{
			return ((ratio.abs() / het_inputs.chi1).pow(1 / (het_inputs.chi2 - 1))
				* ratio.sign() * (het_inputs.agrid + het_inputs.a_kink)).min(het_inputs.dmax);
		}
	}

	namespace Adj_fun_kaplan
	{

		template <typename Derived1>
		auto
			adj(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived1>& d_abs)
		{
			return het_inputs.chi0 * d_abs * het_inputs.innaction
				+ het_inputs.chi1 * (d_abs / het_inputs.akinkgrid).pow(het_inputs.chi2)
				* het_inputs.akinkgrid;
		}

		double
			adj(const Het_Inputs& het_inputs, const double& d_abs, int k, int m);

		template <typename Derived1>
		auto
			adj1(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived1>& d)
		{
			return (het_inputs.chi1 * het_inputs.chi2 
				* (d.abs() / het_inputs.akinkgrid).pow(het_inputs.chi2 - 1)
				+ het_inputs.chi0 * het_inputs.innaction) * d.sign();
		}

		double
			adj1(const Het_Inputs& het_inputs, const double& d, int k, int m);

		template <typename Derived1>
		auto
			adj1inv(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived1>& ratio)
		{
			return (((ratio.abs() - het_inputs.chi0 * het_inputs.innaction).max(0)
				/ (het_inputs.chi1 * het_inputs.chi2)).pow(1 / (het_inputs.chi2 - 1))
				* ratio.sign() * het_inputs.akinkgrid).min(het_inputs.dmax);
		}
	}
}