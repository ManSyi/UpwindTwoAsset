#include <map>
#include <string>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <vector>
#include <memory>
#include <iterator>
#include <chrono>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_blas.h>

#include <omp.h>

#include "newuoa_h.h"
#include "Solver.h"

//#define EIGEN_USE_MKL_ALL
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct> 

#include "Support.h"
#include "Calibration.h"
#include "Steady_state.h"

#include "Het_block.h"

#include "Two_asset.h"
#include "Kaplan.h"





namespace ta = TA;
namespace ta_kpl = TA::KAPLAN;


using Eigen::MatrixXd;
using Eigen::VectorXd;
//using Eigen::kroneckerProduct;
//typedef Eigen::SparseMatrix<double> SpMat;

using namespace std;

void
SteadyState::init(const Calibration& cal)
{
	const int na = cal.size("na");
	const int nb = cal.size("nb");
	const int nesk = cal.size("nesk");
	int nab = na * nb;
	const int n = nesk * nb * na;

	dist_joint_["HA"].setZero(nab, nesk);
	dist_marg_["HA_B"].setZero(nb);
	dist_marg_["HA_B_CDF"].setZero(nb);
	init_empty_tensor(assetHJB_["HA"], nab, nab, nesk); 

	het_vecs_["HA"]["V"].setZero(nab, nesk);
	het_vecs_["HA"]["ccum1"].setZero(nab, nesk);
	het_vecs_["HA"]["ccum2"].setZero(nab, nesk);
	het_vecs_["HA"]["ccum4"].setZero(nab, nesk);
	het_vecs_["HA"]["subeff1ass"].setZero(nab, nesk);
	het_vecs_["HA"]["subeff2ass"].setZero(nab, nesk);
	het_vecs_["HA"]["wealtheff1ass"].setZero(nab, nesk);
	het_vecs_["HA"]["wealtheff2ass"].setZero(nab, nesk);
	init_empty_tensor(het_policies_["HA"]["sb"], na, nb, nesk);
	init_empty_tensor(het_policies_["HA"]["c"], na, nb, nesk);
	init_empty_tensor(het_policies_["HA"]["b"], na, nb, nesk);
	switch (cal.options().model())
	{
	case Model::KAPLAN:
		init_empty_tensor(het_policies_["HA"]["sa"], na, nb, nesk);
		init_empty_tensor(het_policies_["HA"]["adj"], na, nb, nesk);
		init_empty_tensor(het_policies_["HA"]["d"], na, nb, nesk);
		init_empty_tensor(het_policies_["HA"]["a"], na, nb, nesk);
		
		dist_marg_["HA_A"].setZero(na);
		dist_marg_["HA_A_CDF"].setZero(na);

		dist_joint_["HA_AB"].setZero(na, nb);
		break;
	default:
		break;
	}



	switch (cal.options().model())
	{
	case Model::KAPLAN:
		init_empty_tensor(het_policies_["HA"]["hour"], na, nb, nesk);
		init_empty_tensor(het_policies_["HA"]["labor"], na, nb, nesk);
		break;
	default:
		break;
	}

}


bool
write_steadystate(const SteadyState& ss, ofstream& of)
{
	string fold = "SteadyState";
	filesystem::remove_all(fold);
	filesystem::create_directories(fold);
	string filename = fold + "/aggregates.csv";
	write_map_to_file(of, ss.aggregators_, filename);

	string sfold = fold + "/het_inputs/";
	string ssfold;

	sfold = fold + "/dist_marg/";
	filesystem::create_directories(sfold);
	for (auto& elem : ss.dist_marg_)
	{
		ssfold = sfold + elem.first + "/";
		filesystem::create_directories(ssfold);
		filename = ssfold + "dist.csv";
		write_matrix_to_file(elem.second, of, filename);
	}
	sfold = fold + "/dist_joint/";
	filesystem::create_directories(sfold);
	for (auto& elem : ss.dist_joint_)
	{
		ssfold = sfold + elem.first + "/";
		filesystem::create_directories(ssfold);
		filename = ssfold + "dist.csv";
		write_matrix_to_file(elem.second, of, filename);
	}
	sfold = fold + "/het_vecs/";

	for (auto& elem : ss.het_vecs_)
	{
		ssfold = sfold + elem.first + "/";
		filesystem::create_directories(ssfold);

		for (auto& init : elem.second)
		{
			filename = ssfold + init.first + ".csv";
			write_matrix_to_file(init.second, of, filename);
		}
	}

	sfold = fold + "/policies/";
	
	for (auto& elem : ss.het_policies_)
	{
		ssfold = sfold + elem.first + "/";
		filesystem::create_directories(ssfold);
		for (auto& policy : elem.second)
		{
			filename = ssfold + policy.first + ".csv";
			write_tensor_to_file(of, policy.second, filename); 
		}
	}



	return 1;
}

bool
read_steadystate(SteadyState& ss, ifstream& ifs)
{
	string fold = "SteadyState";
	string filename = fold + "/aggregates.csv";
	read_map_from_file(ifs, ss.aggregators_, filename);
	string sfold;
	string parent;
	string sector;
	string item;
	sfold = fold + "/dist_marg/";
	if (filesystem::exists(sfold))
	{
		
		for (auto const& dir_entry : filesystem::recursive_directory_iterator(sfold))
		{
			if (dir_entry.is_regular_file())
			{
				parent = dir_entry.path().parent_path().string();
				sector = parent.substr(parent.rfind("/") + 1);
				read_matrix_from_file(ss.dist_marg_[sector], ifs, dir_entry.path());

			}
		}
	}
	sfold = fold + "/dist_joint/";
	if (filesystem::exists(sfold))
	{
		
		for (auto const& dir_entry : filesystem::recursive_directory_iterator(sfold))
		{
			if (dir_entry.is_regular_file())
			{
				parent = dir_entry.path().parent_path().string();
				sector = parent.substr(parent.rfind("/") + 1);
				read_matrix_from_file(ss.dist_joint_[sector], ifs, dir_entry.path());

			}
		}
	}
	sfold = fold + "/het_vecs/";
	if (filesystem::exists(sfold))
	{
		
		for (auto const& dir_entry : filesystem::recursive_directory_iterator(sfold))
		{
			if (dir_entry.is_regular_file())
			{
				parent = dir_entry.path().parent_path().string();
				sector = parent.substr(parent.rfind("/") + 1);
				item = dir_entry.path().stem().string();
				read_matrix_from_file(ss.het_vecs_[sector][item], ifs, dir_entry.path());
			}
		}
	}
	sfold = fold + "/policies/";
	if (filesystem::exists(sfold))
	{
		
		for (auto const& dir_entry : filesystem::recursive_directory_iterator(sfold))
		{
			if (dir_entry.is_regular_file())
			{
				parent = dir_entry.path().parent_path().string();
				sector = parent.substr(parent.rfind("/") + 1);
				item = dir_entry.path().stem().string();
				read_tensor_from_file(ifs, ss.het_policies_[sector][item], dir_entry.path());
			}
		}
	}
	
	return 1;
}

void
solve_ss_from_files(Calibration& cal, SteadyState& ss, ifstream& ifs)
{
	if (read_steadystate(ss, ifs)) renew_parameters(ss.get_aggregators(), cal);
}

void
solve_ss(Calibration& cal, SteadyState& ss)
{
	switch (cal.options().model())
	{
	case Model::KAPLAN:
		solve_ss_equm(cal, ss);
		renew_parameters(ss.get_aggregators(), cal);
		ta_kpl::check_all_market_clearing(cal, ss);
		break;
	}
}

std::ostream&
print_init(const std::string& name, const double& value, const double& low,
	const double& high, const int& weidth, std::ostream& os)
{
	os << std::setw(weidth) << std::left << name << std::setw(weidth) << std::left << value
		<< "[" << std::setw(weidth / 2) << std::left
		<< std::setprecision(2) << low << std::setw(weidth / 2) << std::right << std::setprecision(2) << high << "]";
	return os;
}



std::ostream&
print_head(std::ostream& os, const std::map<std::string, Params_range>& init_map, const std::map<std::string, Targets_state>& target_map)
{
	os << std::setw(16) << std::left << "Iterations";
	for (auto& pair : init_map)
	{
		os << std::setw(16) << std::left << pair.first;
	}
	string name = "Target";
	int pos = name.size();
	for (auto& pair : target_map)
	{
		name = pair.first.substr(pos);
		os << std::setw(20) << std::left << name;
	}
	os << std::setw(16) << std::left << "Error(norm-2)";
	return os;
}

std::ostream&
print_iteration(std::ostream& os, const std::map<std::string, Params_range>& init_map, const std::map<std::string, Targets_state>& target_map, const int& iter)
{
	os << std::setw(16) << std::left << iter;
	for (auto& pair : init_map)
	{
		os << std::setw(16) << std::left << pair.second.xcur;
	}
	for (auto& pair : target_map)
	{
		os << std::setw(20) << std::left << pair.second.implied;
	}
	return os;
}

void
display_init_params(const std::unordered_map<std::string, parameter>& params, const int& wedth,
	const std::set<std::string>& init_names, std::map<std::string, Params_range>& init_map)
{
	cout.setf(ios_base::fixed, ios_base::floatfield);
	cout.setf(ios_base::showpoint);
	cout.precision(3);
	
	cout << "\n\nInitial parameters and bounds:\n" << endl;
	cout << std::setw(6) << std::left << "    " << std::setw(wedth) << std::left << "Params" <<
		std::setw(wedth) << std::left << "Value" << "Bounds" << endl;
	int i = 0;

	string high;
	string low;
	string index;
	for (auto& e : init_names)
	{
		cout.precision(3);

		high = e + "_high";
		low = e + "_low";
		Params_range& value = init_map[e];
		value.init = get_value(params, e);
		value.low = get_value(params, low);
		value.high = get_value(params, high);
		check_bound(e, value.low, value.high, value.init);
		index = " " + std::to_string(++i) + ".";
		cout << std::setw(6) << std::left << index;
		print_init(e, value.init, value.low, value.high, wedth, cout) << endl;
	}
}

void
display_dist_targets(const std::unordered_map<std::string, parameter>& target_value, int width,
	const std::set<std::string>& target_set, std::map<std::string, Targets_state>& target_map)
{
	int i = 0;
	std::cout << "\n\nTargets options:\n" << endl;
	string index;
	for (auto& e : target_set)
	{
		target_map[e].actual = get_value(target_value, e);
		target_map[e].weight = get_value(target_value, e + "Weight");
		index = " " + std::to_string(++i) + ".";
		std::cout << std::setw(6) << std::left << index;
		std::cout << std::setw(width) << std::left << e << std::setw(5) << std::left << " = " << target_map[e].actual << endl;
	}
}


void
Het_equm::init()
{
	maxiter_equm = cal.size("maxiter_equm");
	tols_equm = cal.params("tols_equm");
	step_equm = cal.params("step_equm");

	switch (model)
	{
	case Model::KAPLAN:
		het_fun = ta_kpl::solve_ss_het_equm;
		moments_fun = ta_kpl::solve_equm_moments;
		moments_update_fun = ta_kpl::update_equm_moments;
		moments_res_fun = ta_kpl::solve_equm_res_moments;
		print_head = ta_kpl::print_equm_head;
		print_iter = ta_kpl::print_equm_iteration;
		break;
	default:
		break;
	}
}

void
Het_calibration::init()
{
	int wedth = 16;
	display_init_params(cal.parameter_map(), wedth, cal.options().estimate_opt_params_set(), init_map);
	display_dist_targets(het_inputs.cal.parameter_map(), 20, target_set, target_map);

	name = "HA_BLOCK";
	maxiter_roots = cal.size("maxiter_roots");
	tols_roots = cal.params("tols_roots");
	step_size = cal.params("step_size");
	maxiter_mins = cal.size("maxiter_mins");
	tols_mins = cal.params("tols_mins");

	maxfun_multip = 200;
	rhobeg = cal.params("rhobeg");
	rhoend = cal.params("rhoend");
	ndfls = cal.size("ndfls");
	num_thread = cal.size("num_thread");
	
	roots_target = ROOTS::target_HA_moments;
	min_target = Nelder_Mead::target_HA_moments;
	dfls_target = DFNLS::target_HA_moments;

	switch (model)
	{
	case Model::KAPLAN:
		het_fun = ta::solve_convergent_pols;
		break;
	}

	switch (model)
	{
	case Model::KAPLAN:
		moments_fun = ta_kpl::solve_targeting_moments;
		break;
	default:
		break;
	}

	
}


void
Het_Outputs::init()
{
	tfp = cal.params("tfp");
	mc = cal.params("mc");
	b0_lb = cal.params("b0_lb");
	b0_lb = cal.params("b0_ub");
	alpha = cal.params("alpha");
	targetKYratio = cal.params("targetKYratio");
	targetKNratio = cal.params("targetKNratio");
	deprec = cal.params("deprec");
	govdebtYratio = cal.params("govdebtYratio");
	maxiter_dist = cal.size("maxiter_dist");
	maxiter_policy = cal.size("maxiter_policy");

	tols_dist = cal.params("tols_dist");
	tols_policy = cal.params("tols_policy");

	a0_close_frac = cal.params("a0_close_frac");
	b0_close_frac = cal.params("b0_close_frac");
	output_data = cal.params("data_annual_output") / 4.0;

	switch (cal.options().model())
	{
	case Model::KAPLAN:
		het_fun = ta::solve_convergent_pols;
		het_one_step_fun = ta::solve_het_one_step;
		break;
	}

	switch (cal.options().model())
	{
	case Model::KAPLAN:
		moments_fun = ta_kpl::solve_ss_het_one_step_moments;
		break;
	default:
		break;
	}

}

void
solve_ss_equm(const Calibration& cal, SteadyState& ss)
{
	Het_Inputs het_inputs = { cal };
	het_inputs.init();
	het_inputs.set_hh();
	Het_workspace ws = { het_inputs };
	ws.init();
	ws.init_dist();
	het_inputs.solve_both_binding_pols(het_inputs, ws);
	ws.init_bellman();
	Het_Outputs het_outputs = { cal, cal.b(), cal.a()};
	het_outputs.init();


	if (cal.options().run().calibration)
	{
		Het_calibration het_cal = { cal, het_inputs, ws, het_outputs };
		het_cal.init();
		iteration(het_cal);
		for (auto& e : het_cal.init_map) ss.get_aggregators().emplace(e.first, e.second.xcur);
	}

	if (cal.options().run().solve_equm)
	{
		Het_equm het_equm = { cal, het_inputs, ws, het_outputs };
		het_equm.init();
		cout.setf(ios_base::fixed, ios_base::floatfield);
		cout.precision(8);
		chrono::high_resolution_clock::time_point tp1 = chrono::high_resolution_clock::now();
		chrono::high_resolution_clock::time_point tp2 = chrono::high_resolution_clock::now();
		chrono::duration<long long, nano> dur1 = tp2 - tp1;
		ofstream fout("Iterations_equlibrium.txt");
		streambuf* oldcout = cout.rdbuf(fout.rdbuf());
		het_equm.het_fun(het_equm, het_inputs, ws, het_outputs);
		tp2 = chrono::high_resolution_clock::now();
		dur1 = tp2 - tp1;
		cout << "\n\nIterations cost " << chrono::duration_cast<chrono::milliseconds>(dur1).count()
			<< " milliseconds" << endl;
		cout.rdbuf(oldcout);
	}

	if (!cal.options().run().calibration && !cal.options().run().solve_equm)
		solve_ss_het_one_step(het_inputs, ws, het_outputs);

	

	if (cal.options().run().solve_equm || !cal.options().run().calibration)
	{
		solve_cum(het_inputs, ws);
		solve_ss_decomp(het_inputs, ws);
	}
	het_inputs.swap_vecs(ws, ss.get_het_vecs("HA"));
	het_inputs.solve_res_pols(het_inputs, ws);
	het_inputs.solve_cdf(het_inputs, ws);
	het_inputs.target_htm(cal, het_outputs, het_inputs, ws);
	het_inputs.solve_dist_stat(het_inputs, het_outputs, ws, ss.get_aggregators());
	het_inputs.solve_res_aggregators(het_inputs, het_outputs, ss);
	het_inputs.swap_dist(ws, ss);
	het_inputs.map_pols(het_inputs, ws, ss.get_policies("HA"));
	//construct_ss_indiv_map(cal, het_inputs, het_outputs, ws, ss);

	ws.delete_solvers();
}



void
solve_ss_het_one_step(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp)
{
	ifstream ifs;
	string filename_V = "SteadyState/het_vecs/HA/V.csv";
	string filename_dist = "SteadyState/dist_joint/HA/dist.csv";
	if (std::filesystem::exists(filename_V) && std::filesystem::exists(filename_dist))
	{
		read_matrix_from_file(ws.V, ifs, filename_V);
		read_matrix_from_file(ws.dist, ifs, filename_dist);
		htp.het_one_step_fun(het_inputs, ws);
	}
	else
	{
		htp.het_fun(htp, het_inputs, ws);
	}
	htp.moments_fun(het_inputs, ws, htp);
}


void
construct_ss_indiv_map(const Calibration& cal, const Het_Inputs& het_inputs,
	Het_Outputs& htp, Het_workspace& ws, SteadyState& ss)
{
	std::swap(ss.get_het_vecs("HA")["V"], ws.V);
	std::swap(ss.get_het_vecs("HA")["ccum1"], ws.ccum1);
	std::swap(ss.get_het_vecs("HA")["ccum2"], ws.ccum2);
	std::swap(ss.get_het_vecs("HA")["ccum4"], ws.ccum4);
	std::swap(ss.get_het_vecs("HA")["subeff1ass"], ws.subeff1ass);
	std::swap(ss.get_het_vecs("HA")["wealtheff1ass"], ws.wealtheff1ass);

	switch (cal.options().model())
	{

	case Model::KAPLAN:
		std::swap(ss.get_het_vecs("HA")["subeff2ass"], ws.subeff2ass);
		std::swap(ss.get_het_vecs("HA")["wealtheff2ass"], ws.wealtheff2ass);
		ta::solve_res_pols(het_inputs, ws);
		ta::solve_cdf(het_inputs, ws);
		ta::target_HtM(cal, htp, het_inputs, ws);
		ta::solve_dist_stat(het_inputs, htp, ws, ss.get_aggregators());
		ta_kpl::solve_res_aggregators(het_inputs, htp, ss);
		ta::swap_het_dist(ws, ss);
		ta::convert_pols_map(het_inputs, ws, ss.get_policies("HA"));
		break;
	default:
		break;
	}


}

void
check_bound(const string& name, double low, double high, double xa)
{
	if (xa > high || xa < low)
	{
		cout << "No meaning parameter " << name << " = " << xa << endl;
		cout << "Not in interval [" << low << "  " << high << "]." << endl;
		exit(0);
	}
}





namespace ROOTS
{


	gsl_vector*
		init_parameters(const std::map<std::string, Params_range>& init_map)
	{

		gsl_vector* x_init = gsl_vector_alloc(init_map.size());
		int i = 0;
		for (auto& pair : init_map)
		{
			gsl_vector_set(x_init, i++, pair.second.init);
		}
		return x_init;
	}

	int
		target_HA_moments(const gsl_vector* x, void* params, gsl_vector* f)
	{
		Het_calibration* het_params = (Het_calibration*)params;

		assign_parameters(x, het_params);

		solve_HA(het_params->het_fun, het_params->moments_fun,
			het_params->het_inputs, het_params->het_outputs, het_params->ws);

		const Het_Outputs& het_outputs = het_params->het_outputs;

		auto ptr = het_params->target_map.begin();

		if (het_params->targets.targetLiqYratio)
		{
			ptr = het_params->target_map.find("targetLiqYratio");
			ptr->second.implied = het_outputs.LiqYratio;
			gsl_vector_set(f, std::distance(het_params->target_map.begin(), ptr), ptr->second.implied - ptr->second.actual);
		}
		if (het_params->targets.targetIllYratio)
		{
			ptr = het_params->target_map.find("targetIllYratio");
			ptr->second.implied = het_outputs.IllYratio;
			gsl_vector_set(f, std::distance(het_params->target_map.begin(), ptr), ptr->second.implied - ptr->second.actual);
		}
		if (het_params->targets.targetKYratio)
		{
			ptr = het_params->target_map.find("targetKYratio");
			ptr->second.implied = het_outputs.KYratio;
			gsl_vector_set(f, std::distance(het_params->target_map.begin(), ptr), ptr->second.implied - ptr->second.actual);
		}
		if (het_params->targets.targetFracIll0)
		{
			ptr = het_params->target_map.find("targetFracIll0");
			ptr->second.implied = het_outputs.FracIll0;
			gsl_vector_set(f, std::distance(het_params->target_map.begin(), ptr), ptr->second.implied - ptr->second.actual);
		}
		if (het_params->targets.targetFracLiq0Ill0)
		{
			ptr = het_params->target_map.find("targetFracLiq0Ill0");
			ptr->second.implied = het_outputs.FracLiq0Ill0;
			gsl_vector_set(f, std::distance(het_params->target_map.begin(), ptr), ptr->second.implied - ptr->second.actual);
		}
		if (het_params->targets.targetFracLiqNeg)
		{
			ptr = het_params->target_map.find("targetFracLiqNeg");
			ptr->second.implied = het_outputs.FracLiqNeg;
			gsl_vector_set(f, std::distance(het_params->target_map.begin(), ptr), ptr->second.implied - ptr->second.actual);
		}
		if (het_params->targets.targetHour)
		{
			ptr = het_params->target_map.find("targetHour");
			ptr->second.implied = het_outputs.Ehour;
			gsl_vector_set(f, std::distance(het_params->target_map.begin(), ptr), ptr->second.implied - ptr->second.actual);
		}
		if (het_params->targets.targetLabor)
		{
			ptr = het_params->target_map.find("targetLabor");
			ptr->second.implied = het_outputs.Elabor;
			gsl_vector_set(f, std::distance(het_params->target_map.begin(), ptr), ptr->second.implied - ptr->second.actual);
		}

		print_iteration(cout, het_params->init_map, het_params->target_map, het_params->iter) 
			<< std::setw(16) << std::left << gsl_blas_dnrm2(f) << endl;

		return GSL_SUCCESS;
	}


	void
		assign_parameters(const gsl_vector* x, Het_calibration* het_params)
	{
		auto ptr = het_params->init_map.begin();
		++het_params->iter;
		if (het_params->estimate_opt_params.rho)
		{
			ptr = het_params->init_map.find("rho");
			ptr->second.xcur = gsl_vector_get(x, std::distance(het_params->init_map.begin(), ptr));
			check_bound("rho", ptr->second.low, ptr->second.high, ptr->second.xcur);
			het_params->het_inputs.rho = ptr->second.xcur;
		}

		if (het_params->estimate_opt_params.chi1)
		{
			ptr = het_params->init_map.find("chi1");
			ptr->second.xcur = gsl_vector_get(x, std::distance(het_params->init_map.begin(), ptr));
			check_bound("chi1", ptr->second.low, ptr->second.high, ptr->second.xcur);
			het_params->het_inputs.chi1 = ptr->second.xcur;
		}

		if (het_params->estimate_opt_params.chi0)
		{
			ptr = het_params->init_map.find("chi0");
			ptr->second.xcur = gsl_vector_get(x, std::distance(het_params->init_map.begin(), ptr));
			check_bound("chi0", ptr->second.low, ptr->second.high, ptr->second.xcur);
			het_params->het_inputs.chi0 = ptr->second.xcur;
		}

		if (het_params->estimate_opt_params.chi2)
		{
			ptr = het_params->init_map.find("chi2");
			ptr->second.xcur = gsl_vector_get(x, std::distance(het_params->init_map.begin(), ptr));
			check_bound("chi2", ptr->second.low, ptr->second.high, ptr->second.xcur);
			het_params->het_inputs.chi2 = ptr->second.xcur;
		}

		if (het_params->estimate_opt_params.rb_wedge)
		{
			ptr = het_params->init_map.find("rb_wedge");
			ptr->second.xcur = gsl_vector_get(x, std::distance(het_params->init_map.begin(), ptr));
			check_bound("rb_wedge", ptr->second.low, ptr->second.high, ptr->second.xcur);
			het_params->het_inputs.rb_wedge = ptr->second.xcur;
		}

		if (het_params->estimate_opt_params.vphi)
		{
			ptr = het_params->init_map.find("vphi");
			ptr->second.xcur = gsl_vector_get(x, std::distance(het_params->init_map.begin(), ptr));
			check_bound("vphi", ptr->second.low, ptr->second.high, ptr->second.xcur);
			het_params->het_inputs.vphi = ptr->second.xcur;
		}
		
	}
}


namespace Nelder_Mead
{
	gsl_vector*
		init_parameters(const std::map<std::string, Params_range>& init_map)
	{
		return ROOTS::init_parameters(init_map);
	}

	void
		assign_parameters(const gsl_vector* x, Het_calibration* het_params)
	{
		ROOTS::assign_parameters(x, het_params);
	}

	double
		target_HA_moments(const gsl_vector* x, void* params)
	{
		Het_calibration* het_params = (Het_calibration*)params;

		assign_parameters(x, het_params);

		solve_HA(het_params->het_fun, het_params->moments_fun,
			het_params->het_inputs, het_params->het_outputs, het_params->ws);

		const Het_Outputs& het_outputs = het_params->het_outputs;

		double obj_fun = 0.0;
		auto ptr = het_params->target_map.begin();

		if (het_params->targets.targetLiqYratio)
		{
			ptr = het_params->target_map.find("targetLiqYratio");
			ptr->second.implied = het_outputs.LiqYratio;
			obj_fun += pow(ptr->second.implied / ptr->second.actual - 1.0, 2.0);
		}
		if (het_params->targets.targetIllYratio)
		{
			ptr = het_params->target_map.find("targetIllYratio");
			ptr->second.implied = het_outputs.IllYratio;
			obj_fun += pow(ptr->second.implied / ptr->second.actual - 1.0, 2.0);
		}
		if (het_params->targets.targetKYratio)
		{
			ptr = het_params->target_map.find("targetKYratio");
			ptr->second.implied = het_outputs.KYratio;
			obj_fun += pow(ptr->second.implied / ptr->second.actual - 1.0, 2.0);
		}
		if (het_params->targets.targetFracIll0)
		{
			ptr = het_params->target_map.find("targetFracIll0");
			ptr->second.implied = het_outputs.FracIll0;
			obj_fun += pow(ptr->second.implied / ptr->second.actual - 1.0, 2.0);
		}
		if (het_params->targets.targetFracLiq0Ill0)
		{
			ptr = het_params->target_map.find("targetFracLiq0Ill0");
			ptr->second.implied = het_outputs.FracLiq0Ill0;
			obj_fun += pow(ptr->second.implied / ptr->second.actual - 1.0, 2.0);
		}
		if (het_params->targets.targetFracLiqNeg)
		{
			ptr = het_params->target_map.find("targetFracLiqNeg");
			ptr->second.implied = het_outputs.FracLiqNeg;
			obj_fun += pow(ptr->second.implied / ptr->second.actual - 1.0, 2.0);
		}
		if (het_params->targets.targetHour)
		{
			ptr = het_params->target_map.find("targetHour");
			ptr->second.implied = het_outputs.Ehour;
			obj_fun += pow(ptr->second.implied / ptr->second.actual - 1.0, 2.0);
		}
		if (het_params->targets.targetLabor)
		{
			ptr = het_params->target_map.find("targetLabor");
			ptr->second.implied = het_outputs.Elabor;
			obj_fun += pow(ptr->second.implied / ptr->second.actual - 1.0, 2.0);
		}
		return obj_fun;
	}
}

void
solve_HA(const Het_convertgent_fun& solve_pols, const Het_moments_fun& moments,
	Het_Inputs& het_inputs, Het_Outputs& targets_params, Het_workspace& ws)
{
	if (targets_params.targets.targetFracLiqNeg)
	{
		het_inputs.set_interest();
		het_inputs.set_income();
	}
	if (targets_params.targets.targetHour || targets_params.targets.targetLabor || targets_params.targets.targetFracLiqNeg)
	{
		het_inputs.solve_both_binding_pols(het_inputs, ws);
	}
	het_inputs.set_HJBdiags();

	solve_pols(targets_params, het_inputs, ws); 
	moments(het_inputs, ws, targets_params);
}


namespace DFNLS
{


	void
		assign_parameters(const double* y, Het_calibration* het_params)
	{
		++het_params->iter;
		auto p = het_params->init_map.begin();

		// Lexicographical order
		if (het_params->estimate_opt_params.rho)
		{
			p = het_params->init_map.find("rho");
			p->second.xcur = logistic(y[std::distance(het_params->init_map.begin(), p)],
				p->second.high, p->second.low, p->second.init);
			het_params->het_inputs.rho = p->second.xcur;
		}

		if (het_params->estimate_opt_params.chi1)
		{
			p = het_params->init_map.find("chi1");
			p->second.xcur = logistic(y[std::distance(het_params->init_map.begin(), p)],
				p->second.high, p->second.low, p->second.init);
			het_params->het_inputs.chi1 = p->second.xcur;
		}

		if (het_params->estimate_opt_params.chi0)
		{
			p = het_params->init_map.find("chi0");
			p->second.xcur = logistic(y[std::distance(het_params->init_map.begin(), p)],
				p->second.high, p->second.low, p->second.init);
			het_params->het_inputs.chi0 = p->second.xcur;
			if (!het_params->estimate_opt_params.chi1)
			{
				het_params->het_inputs.chi1 = 1.0 - het_params->het_inputs.chi0;
			}
		}

		if (het_params->estimate_opt_params.chi2)
		{
			p = het_params->init_map.find("chi2");
			p->second.xcur = logistic(y[std::distance(het_params->init_map.begin(), p)],
				p->second.high, p->second.low, p->second.init);
			het_params->het_inputs.chi2 = p->second.xcur;
		}

		if (het_params->estimate_opt_params.rb_wedge)
		{
			p = het_params->init_map.find("rb_wedge");
			p->second.xcur = logistic(y[std::distance(het_params->init_map.begin(), p)],
				p->second.high, p->second.low, p->second.init);
			het_params->het_inputs.rb_wedge = p->second.xcur;
		}

		if (het_params->estimate_opt_params.vphi)
		{
			p = het_params->init_map.find("vphi");
			p->second.xcur = logistic(y[std::distance(het_params->init_map.begin(), p)],
				p->second.high, p->second.low, p->second.init);
			het_params->het_inputs.vphi = p->second.xcur;
		}

	}

	double*
		init_parameters(const std::map<std::string, Params_range>& init_map)
	{
		int i = 0;
		int size = init_map.size();
		double* y = new double[size];
		// Lexicographical order
		for (auto& pair : init_map)
		{
			y[i++] = invlogistic(pair.second.init, pair.second.high, pair.second.low, pair.second.init);
		}
		return y;
	}

	void
		target_HA_moments(const long n, const long mv, const double* y,
			double* v_err, void* data)
	{
		Het_calibration* het_params = (Het_calibration*)(data);

		assign_parameters(y, het_params);

		solve_HA(het_params->het_fun, het_params->moments_fun,
			het_params->het_inputs, het_params->het_outputs, het_params->ws);

		const Het_Outputs& het_outputs = het_params->het_outputs;

		auto ptr = het_params->target_map.begin();

		if (het_params->targets.targetLiqYratio)
		{
			ptr = het_params->target_map.find("targetLiqYratio");
			ptr->second.implied = het_outputs.LiqYratio;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetIllYratio)
		{
			ptr = het_params->target_map.find("targetIllYratio");
			ptr->second.implied = het_outputs.IllYratio;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetKYratio)
		{
			ptr = het_params->target_map.find("targetKYratio");
			ptr->second.implied = het_outputs.KYratio;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetFracIll0)
		{
			ptr = het_params->target_map.find("targetFracIll0");
			ptr->second.implied = het_outputs.FracIll0;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetFracLiq0)
		{
			ptr = het_params->target_map.find("targetFracLiq0");
			ptr->second.implied = het_outputs.FracLiq0;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetFracLiq0Illpos)
		{
			ptr = het_params->target_map.find("targetFracLiq0Illpos");
			ptr->second.implied = het_outputs.FracLiq0Illpos;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetFracLiq0Ill0)
		{
			ptr = het_params->target_map.find("targetFracLiq0Ill0");
			ptr->second.implied = het_outputs.FracLiq0Ill0;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetFracLiqNeg)
		{
			ptr = het_params->target_map.find("targetFracLiqNeg");
			ptr->second.implied = het_outputs.FracLiqNeg;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetHour)
		{
			ptr = het_params->target_map.find("targetHour");
			ptr->second.implied = het_outputs.Ehour;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (het_params->targets.targetLabor)
		{
			ptr = het_params->target_map.find("targetLabor");
			ptr->second.implied = het_outputs.Elabor;
			v_err[std::distance(het_params->target_map.begin(), ptr)]
				= ptr->second.weight * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		print_iteration(cout, het_params->init_map, het_params->target_map, het_params->iter)
			<< std::setw(16) << std::left << norm(v_err, het_params->target_map.size()) << endl;
	}
}

double norm(double* v, int size)
{
	double x = 0.0;
	for (int i = 0; i < size; ++i)
	{
		x += v[i] * v[i];
	}
	return std::sqrt(x);
}