#pragma once
#include <map>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::ArrayXXd het2;
typedef std::vector<het2> Tensor;
struct Het_Outputs;
struct Het_Inputs;
struct Het_equm;
struct Het_workspace;

typedef Eigen::ArrayXXd Grid;

class VexNode;
typedef std::vector<VexNode> DAG;

typedef Eigen::ArrayXd het;
struct BlockParams;

class Calibration;
class SteadyState;

namespace TA
{
	namespace KAPLAN
	{
		struct Capital_params
		{
			double& Y;
			double Elabor;
			double Ea;
			double mc;
			double alpha;
			double tfp;
			double profdistfrac;
			double ra;
			double high;
			double low;
			double tols;
			int maxiter = 0;
		};

		void
			set_parameters(Calibration& cal, SteadyState& ss);

		void
			solve_res_aggregators(const Het_Inputs& het_inputs, const Het_Outputs& htp,
				SteadyState& ss);
		void
			solve_equm_res_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp);

		void
			solve_ss_het_one_step_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp);

		void
			solve_targeting_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp);

		void
			solve_rhs(const Het_Inputs& het_inputs, Het_workspace& ws, int r);

		double
			capital_targets(double K, void* params_);

		void
			print_equm_iteration(const Het_equm& params, const Het_Inputs& het_inputs);

		void
			print_equm_head();

		void
			update_equm_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp);

		void
			solve_equm_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp);

		void
			solve_ss_het_equm(Het_equm& params, Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp);

		void
			solve_both_binding_pols(const Het_Inputs& het_inputs, Het_workspace& ws);

		void
			solve_FdVaF(const Het_Inputs& het_inputs, Het_workspace& ws, double& droot, double& H, int r, int k, int m);
		bool
			Fd_bisec(const Het_Inputs& het_inputs, Het_workspace& ws, const double& Va,
				const double& dlow, const double& dhigh, const double& Flow, double& droot, int r, int k, int m);
		double
			FdVa(const Het_Inputs& het_inputs, Het_workspace& ws, const double& Va, const double& d, int r, int k, int m);

		double
			FdVa(const Het_Inputs& het_inputs, Het_workspace& ws, const double& Va, const double& d, double& Fd, int r, int k, int m);

		void
			solve_binding_cons(const Het_Inputs& het_inputs, const double& d, Het_workspace& ws, int r, int k, int m);

		void
			solve_binding_pols(const Het_Inputs& het_inputs, Het_workspace& ws, int r);

		void
			solve_unbinding_cons(const Het_Inputs& het_inputs, const Het_workspace& ws, const het2& Vb, int r,
				het2& c, het2& sc, het2& hour, het2& Hc);
		void
			solve_unbinding_res(const Het_Inputs& het_inputs, const Het_workspace& ws,
				const het2& Vb, const het2& Va, const het2& sc, const het2& utility, int r,
				het2& d, het2& sd, het2& sa, het2& sb, het2& H);

		void
			check_all_market_clearing(const Calibration& cal, SteadyState& ss);



	}
}