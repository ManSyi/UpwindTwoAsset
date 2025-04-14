#pragma once
#include <map>
#include <unordered_map>
#include <string>
#include <set>
#include <iostream>
#include <fstream>
#include <gsl/gsl_vector.h>
#include "newuoa_h.h"
#include "Support.h"
#include "Calibration.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<double> SpMat;
struct Het_workspace;
struct Het_Outputs;
struct Het_Inputs;
struct Het_equm;

typedef double parameter;
typedef double aggregates;

typedef Eigen::SparseMatrix<double> SpMat;

typedef Eigen::ArrayXXd het2;
typedef std::vector<het2> Tensor;

typedef double (*Root_target)(double ra, void* params);
typedef int
(*Roots_target)(const gsl_vector* x, void* params, gsl_vector* f);
typedef double
(*Min_target)(const gsl_vector* x, void* params);


class SteadyState {
public:
	SteadyState() = default;
	friend bool write_steadystate(const SteadyState& ss, std::ofstream& of);
	friend bool read_steadystate(SteadyState& ss, std::ifstream& in);

	
	void init(const Calibration& cal);

	const aggregates& get_aggregator(const std::string& name) const 
	{ 
		return get_value(aggregators_, name);
	}
	std::unordered_map<std::string, aggregates>& get_aggregators()
	{
		return aggregators_;
	}
	const std::unordered_map<std::string, aggregates>& get_aggregators() const 
	{
		return aggregators_;
	}
	std::unordered_map<std::string, Tensor>& get_policies(const std::string& name)
	{ 
		return get_value(het_policies_, name);
	}

	const Tensor& get_policy(const std::string& sector, const std::string& policy) const
	{
		return get_value(get_value(het_policies_, sector), policy);
	}

	Eigen::VectorXd& get_marg_dist(const std::string& het_name)
	{
		return get_value(dist_marg_, het_name);
	}
	const Eigen::VectorXd& get_marg_dist(const std::string& het_name) const
	{
		return get_value(dist_marg_, het_name);
	}

	Eigen::MatrixXd& get_joint_dist(const std::string& het_name)
	{
		return get_value(dist_joint_, het_name);
	}
	const Eigen::MatrixXd& get_joint_dist(const std::string& het_name) const
	{
		return get_value(dist_joint_, het_name);
	}

	std::vector<SpMat>& get_trans_mat(const std::string& het_name)
	{
		return get_value(assetHJB_, het_name);
	}
	const std::vector<SpMat>& get_trans_mat(const std::string& het_name) const
	{
		return get_value(assetHJB_, het_name);
	}

	std::unordered_map<std::string, Eigen::MatrixXd>& get_het_vecs(const std::string& het_name)
	{
		return get_value(het_vecs_, het_name);
	}
	const std::unordered_map<std::string, Eigen::MatrixXd>& get_het_vecs(const std::string& het_name) const
	{
		return get_value(het_vecs_, het_name);
	}

private:

	std::unordered_map<std::string, aggregates> aggregators_;
	std::unordered_map<std::string, Eigen::VectorXd> dist_marg_;
	std::unordered_map<std::string, Eigen::MatrixXd> dist_joint_;
	std::unordered_map<std::string, std::vector<SpMat>> assetHJB_;
	std::unordered_map<std::string, std::unordered_map<std::string, Eigen::MatrixXd>> het_vecs_;
	std::unordered_map<std::string, std::unordered_map<std::string, Tensor>> het_policies_;
};

typedef void (*Het_one_step_fun)(const Het_Inputs& het_inputs, Het_workspace& ws);

typedef void (*Het_convergent_fun)(const Het_Outputs& target_params,
	const Het_Inputs& het_inputs, Het_workspace& ws);

typedef void (*Het_equm_fun)(Het_equm& het_equm, Het_Inputs& het_inputs, Het_workspace& ws,
	Het_Outputs& targets_params);

typedef void (*Het_moments_fun)(Het_Inputs& het_inputs, Het_workspace& ws,
	Het_Outputs& targets_params);

typedef void (*Print_equm_head_fun)();

typedef void (*Print_equm_iter_fun)(const Het_equm& params, const Het_Inputs& het_inputs);

struct Params_range
{
	double init;
	double low;
	double high;
	double xcur;
};

struct Targets_state
{
	double actual;
	double weight;
	double implied;
};

struct Het_equm
{
	const Calibration& cal;
	Het_Inputs& het_inputs;
	Het_workspace& ws;
	Het_Outputs& het_outputs;
	Het_equm_fun het_fun;
	Het_equm_fun het_fun_one_step;
	Het_moments_fun moments_fun;
	Het_moments_fun moments_update_fun;
	Het_moments_fun moments_res_fun;
	Print_equm_head_fun print_head;
	Print_equm_iter_fun print_iter;
	const Model& model = cal.options().model();
	void init();
	double tols_equm;
	double step_equm;
	double eps = 0;
	int maxiter_equm;
	int iter = 0;
	double x = 0;
	double x_next = 0;
};

struct Het_calibration
{
	const Calibration& cal;
	Het_Inputs& het_inputs;
	Het_workspace& ws;
	Het_Outputs& het_outputs;
	Het_convergent_fun het_fun;
	Het_moments_fun moments_fun;
	void init();
	 
	const Opt& opt = cal.options().opt();
	const Het_opt_params& estimate_opt_params = cal.options().estimate_opt_params();
	const Het_target& targets = cal.options().dist_target();
	const std::set<std::string>& target_set = cal.options().dist_targets_set();
	const Model& model = cal.options().model();
	std::map<std::string, Params_range> init_map;
	std::map<std::string, Targets_state> target_map;

	std::string name;
	bool display = 0;
	Roots_target roots_target;
	Min_target min_target;
	newuoa_dfovec* dfls_target;



	int maxfun_multip = 0;
	int ndfls = 0;
	int num_thread = 0;
	double rhobeg;
	double rhoend;

	double tols_roots;
	int maxiter_roots;
	double step_size;
	int maxiter_mins;
	double tols_mins;
	int iter = 0;

};

void
solve_HA(const Het_convergent_fun& solve_pols, const Het_moments_fun& moments,
	Het_Inputs& het_inputs, Het_Outputs& targets_params, Het_workspace& ws);

void
solve_ss(Calibration& cal, SteadyState& ss);

void
solve_ss_equm(const Calibration& cal, SteadyState& ss);

void
solve_ss_het_one_step(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp);

//void
//calibrate_ss_het_block(const Calibration& cal, Het_Inputs& het_inputs,
//	Het_workspace& ws, Het_Outputs& het_outputs, SteadyState& ss);

void
construct_ss_indiv_map(const Calibration& cal, const Het_Inputs& het_inputs,
	Het_Outputs& htp, Het_workspace& ws, SteadyState& ss);

bool 
write_steadystate(const SteadyState& ss, std::ofstream& of);

bool
solve_model(std::ofstream& of, std::ifstream& ifs, Calibration& cal);

void 
solve_ss_from_files(Calibration& cal, SteadyState& ss, std::ifstream& ifs);


bool 
read_steadystate(SteadyState& ss, std::ifstream& in);

std::ostream&
print_init(const std::string& name, const double& value, const double& low,
	const double& high, const int& weidth, std::ostream& os);

std::ostream&
print_head(std::ostream& os, const std::map<std::string, Params_range>& init_map, const std::map<std::string, Targets_state>& target_map);

std::ostream&
print_iteration(std::ostream& os, const std::map<std::string, Params_range>& init_map, const std::map<std::string, Targets_state>& target_map, const int& iter);

void
display_init_params(const std::unordered_map<std::string, parameter>& params, const int& wedth,
	const std::set<std::string>& init_names, std::map<std::string, Params_range>& init_map);

void
display_dist_targets(const std::unordered_map<std::string, parameter>& target_value, int width,
	const std::set<std::string>& target_set, std::map<std::string, Targets_state>& target_map);

void
check_bound(const std::string& name, double low, double high, double xa);

double norm(double* v, int size);

namespace ROOTS
{


	int
		target_HA_moments(const gsl_vector* x, void* params, gsl_vector* f);

	gsl_vector*
		init_parameters(const std::map<std::string, Params_range>& init_map);

	void
		assign_parameters(const gsl_vector* x, Het_calibration* het_params);



}
namespace Nelder_Mead
{
	double
		target_HA_moments(const gsl_vector* x, void* params);

	gsl_vector*
		init_parameters(const std::map<std::string, Params_range>& init_map);

	void
		assign_parameters(const gsl_vector* x, Het_calibration* het_params);
}

namespace DFNLS
{

	void
		assign_parameters(const double* y, Het_calibration* het_params);

	void
		target_HA_moments(const long n, const long mv, const double* y,
			double* v_err, void* data);

	double*
		init_parameters(const std::map<std::string, Params_range>& init_map);

}

