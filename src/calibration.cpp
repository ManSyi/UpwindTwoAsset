#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <filesystem>
#include <map>
#include <unordered_map>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct> 
#include <unordered_map>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <omp.h>
#include <numeric>
#include <algorithm>

#include "Calibration.h"
#include "Support.h"
#include "Earning_process.h"

using namespace std;

void
Earning_Options::init(std::ifstream& ifs)
{
	if (!read_options(*this, ifs))
	{
		std::cout << "Cannot read earning options from file! Use default options." << endl;
	}
}

void
Options::init(std::ifstream& ifs)
{
	if (!read_options(*this, ifs))
	{
		std::cout << "Cannot read options from file! Use default options." << endl;
	}
}

void
Calibration::init(std::ifstream& ifs)
{
	if (!read_calibrations(*this, ifs))
	{
		std::cout << "Cannot read calibrations from files!" << endl;
		exit(0);
	}
}

void
Calibration::make_grid()
{
	set_Delta_grid();
	set_asset_grid();
	set_skill_grid();
}

void
Calibration::set_Delta_grid()
{
	int ntrans = get_value(size_cal_, "ntrans");
	DeltaTrans_.setZero(ntrans);


	double DeltaTrans = get_value(parameter_cal_, "DeltaTrans");
	DeltaTrans_.setConstant(DeltaTrans);
	// To use the algrithom of Auclert, etc(2019), DeltaTrans must be a constant number.

	DeltaTransCum_.setZero(ntrans);
	partial_sum(DeltaTrans_.cbegin(), DeltaTrans_.cend(), DeltaTransCum_.begin(), plus<double>());
}

void
Calibration::set_asset_grid()
{
	
	int nb = get_value(size_cal_, "nb");
	int na = get_value(size_cal_, "na");

	parameter bmin = get_value(parameter_cal_, "bmin");
	parameter bmax = get_value(parameter_cal_, "bmax");
	if (options_.dist_target().targetFracLiqNeg)
	{
		bmin = get_value(parameter_cal_, "bmin_neg");
	}
	b_.setZero(nb);



	switch (options_.model())
	{
		case Model::KAPLAN:
	{
		parameter amin = get_value(parameter_cal_, "amin");
		parameter amax = get_value(parameter_cal_, "amax");
		a_.setZero(na);
		//set_exp_grid(na, amin, amax, a_);
		parameter bcur = get_value(parameter_cal_, "bcur");
		parameter acur = get_value(parameter_cal_, "acur");
		set_grid(na, amin, amax, acur, a_);
		/* make equal space arround zero */
		a_.segment(0, 10).setLinSpaced(10, a_(0), a_(9));
		int nb_neg = 0;
		if (options_.dist_target().targetFracLiqNeg)
		{
			double nbl = -get_value(parameter_cal_, "transfer") 
				/ (get_value(parameter_cal_, "rb") + get_value(parameter_cal_, "rb_wedge") 
					+ get_value(parameter_cal_, "deathrate"));
			if (bmin < nbl)
			{
				std::cout << "Natual borrowing limit violated!" << "\n";
				std::cout << "natual borrowing limit = " << nbl << "\n";
				std::cout << "actual borrowing limit = " << bmin << endl;
			}
			bmin = std::max(nbl + 1e-5, bmin);
			if (bmin > 0)
			{
				set_grid(nb, bmin, bmax, bcur, b_);
			}
			else
			{
				parameter bcur_neg = get_value(parameter_cal_, "bcur_neg");
				nb_neg = get_value(size_cal_, "nb_neg");
				set_grid(nb - nb_neg, 0, bmax, bcur, b_.segment(nb_neg, nb - nb_neg));
				set_grid(nb_neg / 2 + 1, bmin, bmin / 2.0, bcur_neg, b_.segment(0, nb_neg / 2 + 1));
				/* make b grid dense at FRACbNeg limit and zero */
				b_.segment(nb_neg / 2 + 1, nb_neg / 2 - 1) = bmin - b_.segment(1, nb_neg / 2 - 1).reverse();
			}
		}
		else
		{
			set_grid(nb, bmin, bmax, bcur, b_);
		}
		db_.setZero(nb - 1);
		db_ = b_.segment(1, nb - 1) - b_.segment(0, nb - 1);
		tdb_.setZero(nb);
		tdb_(0) = 0.5 * db_(0);
		tdb_.segment(1, nb - 2) = 0.5 * (db_.segment(0, nb - 2) + db_.segment(1, nb - 2));
		tdb_(nb - 1) = 0.5 * db_(nb - 2);
		da_.setZero(na - 1);
		da_ = a_.segment(1, na - 1) - a_.segment(0, na - 1);
		tda_.setZero(na);
		tda_(0) = 0.5 * da_(0);
		tda_.segment(1, na - 2) = 0.5 * (da_.segment(0, na - 2) + da_.segment(1, na - 2));
		tda_(na - 1) = 0.5 * da_(na - 2);
		agrid_.setZero(na, nb);
		agrid_ = a_.rowwise().replicate(nb);
		bgrid_.setZero(na, nb);
		bgrid_ = b_.transpose().colwise().replicate(na);
		dagrid_.setZero(na - 1, nb);
		dagrid_ = da_.rowwise().replicate(nb);
		dbgrid_.setZero(na, nb - 1);
		dbgrid_ = db_.transpose().colwise().replicate(na);
		tdagrid_.setZero(na, nb);
		tdagrid_ = tda_.rowwise().replicate(nb);
		tdbgrid_.setZero(na, nb);
		tdbgrid_ = tdb_.transpose().colwise().replicate(na);
	}
		break;
	default:
		break;
	}
}

void
Calibration::set_ui_rule()
{
	ui_rule_ = sgrid_.min(get_value(parameter_cal_, "meanlabeff"));
	ui_rule_ /= (ui_rule_ * skill_ss_dist_.array()).sum();
}

void
write_earning_results(std::ofstream& of, const Earning_results& results, const std::string& filename)
{
	string params_name = filename + "_param_results.csv";
	string moments_name = filename + "_moment_results.csv";
	write_map_to_file(of, results.params, params_name);
	write_map_to_file(of, results.moments, moments_name);
}

void
Calibration::set_skill_grid()
{
	filesystem::path model_path = filesystem::current_path();
	filesystem::path process_path;
	int nsk = 0;
	const double tols = get_value(parameter_cal_, "tols_dist");
	const int maxiter = get_value(size_cal_, "maxiter_dist");
	ifstream ifs;
	ofstream of;
	of.precision(16);
	string process = options_.process_name();
	process_path = model_path;
	filesystem::current_path(process_path += "/EarningProcess/" + process);

	read_map_from_file(ifs, earning_size_cal_, "size.csv");
	nsk = size_cal_["nsk"] = earning_size_cal_["nsk"];

	
	switch (options_.process())
	{
	case Process::AR1:case Process::JUMP_DRIFT_MIX:
		read_map_from_file(ifs, earning_parameter_cal_, "parameter.csv");
		break;
	case Process::AR1_MIX:
		break;
	case Process::KAPLAN:
		break;
	default:
		break;
	}
	sgrid_.resize(nsk);
	skill_ss_dist_.resize(nsk);
	markov_.resize(nsk, nsk);

	if (options_.run().estimate_process)
	{
		Earning_results results;
		switch (options_.process())
		{
		case Process::AR1:
			break;
		case Process::AR1_MIX:
			break;
		case Process::KAPLAN:
			break;
		case Process::JUMP_DRIFT_MIX:
			cout << "Estimating jump-drift earning process.." << endl;
			EARNING_PROCESS::ESTIMATION::estimation(*this, results);
			cout << "Done." << endl;
			cout << "\nWriting jump-drift earnig estimating results to params.csv and moments.csv..." << endl;
			write_earning_results(of, results, "estimate");
			renew_map(results.params, earning_parameter_cal_);
			break;
		default:
			break;
		}
	}
	else
	{
		switch (options_.process())
		{
		case Process::AR1:
			break;
		case Process::AR1_MIX:
			break;
		case Process::KAPLAN:
			break;
		case Process::JUMP_DRIFT_MIX:
			cout << "Read parameters of jump-drift earning process.." << endl;
			read_map_from_file(ifs, earning_parameter_cal_, "estimate_param_results.csv");
			cout << "Done." << endl;
			break;
		default:
			break;
		}
	}

	switch (options_.process())
	{
	case Process::AR1_MIX:
		break;
	case Process::KAPLAN:
		read_matrix_from_file(sgrid_, ifs, "ygrid_combined.csv");
		read_matrix_from_file(markov_, ifs, "ymarkov_combined.csv");
		read_matrix_from_file(skill_ss_dist_, ifs, "ydist_combined.csv");
		sgrid_ /= (1.0 + 0.85 * get_value(parameter_cal_, "frisch"));
		for (int n = 0; n < nsk; ++n)
		{
			markov_(n, n) -= markov_.row(n).sum();
		}
		break;
	case Process::JUMP_DRIFT_MIX:
		if (options_.run().discrete_process)
		{
			earning_size_cal_["nsim"] *= 10;
			if (options_.earning_options().discrete_opt_params().width1)
			{
				earning_parameter_cal_["width1"] *= earning_parameter_cal_["sigma1"];
				earning_parameter_cal_["width1_low"] *= earning_parameter_cal_["sigma1"];
				earning_parameter_cal_["width1_high"] *= earning_parameter_cal_["sigma1"];
			}
			if (options_.earning_options().discrete_opt_params().width2)
			{
				earning_parameter_cal_["width2"] *= earning_parameter_cal_["sigma2"];
				earning_parameter_cal_["width2_low"] *= earning_parameter_cal_["sigma2"];
				earning_parameter_cal_["width2_high"] *= earning_parameter_cal_["sigma2"];
			}
			Earning_results results;
			cout << "Estimating jump-drift discrete earning process.." << endl;
			EARNING_PROCESS::DISCRETE::estimation(*this, results, of);
			cout << "Done." << endl;
			cout << "\nWriting jump-drift earnig discrete results to params.csv and moments.csv..." << endl;
			write_earning_results(of, results, "discrete");
			renew_map(results.params, earning_parameter_cal_);
		}
		read_matrix_from_file(sgrid_, ifs, "skill_grid.csv");
		read_matrix_from_file(markov_, ifs, "generator.csv");
		read_matrix_from_file(skill_ss_dist_, ifs, "skill_dist.csv");

		switch (options_.model())
		{
		case Model::KAPLAN:
			
			switch (options_.hour_supply())
			{
			case Hour_supply::Seq:case Hour_supply::GHH:
				sgrid_ /= (1.0 + 0.85 * get_value(parameter_cal_, "frisch"));
				break;
			case Hour_supply::NoSupply:
				break;
			default:
				break;
			}
			break;
		}

		for (int n = 0; n < nsk; ++n)
		{
			markov_(n, n) -= markov_.row(n).sum();
		}



		break;
	case Process::AR1:
		if (options_.run().discrete_process)
		{
			rouwenhorst_AR1(earning_parameter_cal_["rho"], earning_parameter_cal_["sigma"],
				nsk, markov_, sgrid_);
			skill_ss_dist_.setConstant(nsk, 1.0 / nsk);
			find_stationary_dist(markov_.transpose(), tols, maxiter, skill_ss_dist_);
			write_grids(*this, of);
		}
		else
		{
			read_matrix_from_file(sgrid_, ifs, "skill_grid.csv");
			read_matrix_from_file(markov_, ifs, "markov.csv");
			read_matrix_from_file(skill_ss_dist_, ifs, "skill_dist.csv");
		}
		for (int n = 0; n < nsk; ++n)
		{
			markov_.row(n) /= markov_.row(n).sum();
		}
		break;
	default:
		break;
	}
	sgrid_ = sgrid_.exp();
	sgrid_ *= parameter_cal_["meanlabeff"] / skill_ss_dist_.dot(sgrid_.matrix());
	switch (options_.model())
	{
	case Model::KAPLAN:
		size_cal_["nesk"] = nsk;
		break;
	default:
		break;
	}
	size_cal_["ne"] = size_cal_["nesk"] / nsk;
	filesystem::current_path(model_path);
}

bool
write_grids(const Calibration& cal, std::ofstream& of)
{
	write_matrix_to_file(cal.markov_, of, "markov.csv");
	write_matrix_to_file(cal.sgrid_, of, "skill_grid.csv");
	write_matrix_to_file(cal.skill_ss_dist_, of, "skill_dist.csv");
	return 1;
}

bool
read_calibrations(Calibration& cal, std::ifstream& ifs)
{
	read_map_from_file(ifs, cal.parameter_cal_, "parameter.csv");
	read_map_from_file(ifs, cal.size_cal_, "size.csv");
	return 1;

}

void
selct_bool_map(const std::set<std::string>& keys, const std::unordered_map<std::string, bool>& from,
	std::set<std::string>& to)
{
	for (auto& k : keys)
	{
		if (get_value(from, k))
		{
			to.emplace(k);
		}
	}
}

void
Model_Options::init(std::ifstream& ifs)
{
	unordered_map<string, bool> m;
	string filename = "options.csv";
	read_map_from_file(ifs, m, filename);

	if (get_value(m, "KAPLAN"))
	{
		model_ = Model::KAPLAN;
		model_string_ = "KAPLAN";
	}
	string model_dir = model_string_;
	string filename_params = model_dir + "/EXP-PARAMS.csv";
	m.clear();
	read_map_from_file(ifs, m, filename_params);
	for (auto& e : m)
	{
		if (e.second) experiment = e.first;
	}
}

bool
read_options(Earning_Options& options, std::ifstream& ifs)
{
	unordered_map<string, bool> m;
	read_map_from_file(ifs, m, "options.csv");
	set<string> names;

	if (get_value(m, "Earning_target"))
	{
		names = { "targetVarLog", "targetVarD1Log", "targetSkewD1Log", "targetKurtD1Log", "targetVarD5Log", "targetSkewD5Log",
		"targetKurtD5Log", "targetFracD1Less10", "targetFracD1Less20", "targetFracD1Less50"};
		selct_bool_map(names, m, options.dist_targets_set_);

		options.dist_target_.targetVarLog = get_value(m, "targetVarLog");
		options.dist_target_.targetVarD1Log = get_value(m, "targetVarD1Log");
		options.dist_target_.targetSkewD1Log = get_value(m, "targetSkewD1Log");
		options.dist_target_.targetKurtD1Log = get_value(m, "targetKurtD1Log");
		options.dist_target_.targetVarD5Log = get_value(m, "targetVarD5Log");
		options.dist_target_.targetSkewD5Log = get_value(m, "targetSkewD5Log");
		options.dist_target_.targetKurtD5Log = get_value(m, "targetKurtD5Log");
		options.dist_target_.targetFracD1Less10 = get_value(m, "targetFracD1Less10");
		options.dist_target_.targetFracD1Less20 = get_value(m, "targetFracD1Less20");
		options.dist_target_.targetFracD1Less50 = get_value(m, "targetFracD1Less50");

	}
	if (get_value(m, "Earning_opt_params"))
	{
		names = { "beta1", "beta2", "lambda1", "lambda2", "sigma1", "sigma2" };
		selct_bool_map(names, m, options.estimate_opt_params_set_);
		options.estimate_opt_params_.beta1 = get_value(m, "beta1");
		options.estimate_opt_params_.beta2 = get_value(m, "beta2");
		options.estimate_opt_params_.lambda1 = get_value(m, "lambda1");
		options.estimate_opt_params_.lambda2 = get_value(m, "lambda2");
		options.estimate_opt_params_.sigma1 = get_value(m, "sigma1");
		options.estimate_opt_params_.sigma2 = get_value(m, "sigma2");

		names = { "width1", "width2", "curv1", "curv2" };
		selct_bool_map(names, m, options.discrete_opt_params_set_);
		options.discrete_opt_params_.width1 = get_value(m, "width1");
		options.discrete_opt_params_.width2 = get_value(m, "width2");
		options.discrete_opt_params_.curv1 = get_value(m, "curv1");
		options.discrete_opt_params_.curv2 = get_value(m, "curv2");
	}
	if (get_value(m, "Opt"))
	{
		if (get_value(m, "Nelder_Mead"))
		{
			options.opt_ = Opt::Nelder_Mead;
			options.opt_name_ = "Nelder_Mead";
		}
		else if (get_value(m, "ROOTS"))
		{
			options.opt_ = Opt::ROOTS;
			options.opt_name_ = "ROOTS";
		}
		else if (get_value(m, "DFNLS"))
		{
			options.opt_ = Opt::DFNLS;
			options.opt_name_ = "DFNLS";
		}
	}
	return 1;
}

bool
read_options(Options& options, std::ifstream& ifs)
{
	unordered_map<string, bool> m;
	read_map_from_file(ifs, m, "options.csv");
	set<string> names;

	if (get_value(m, "Het_target"))
	{
		names = { "targetLiqYratio", "targetIllYratio", "targetKYratio","targetFracLiq0", "targetFracLiq0Ill0",
			"targetFracLiqNeg", "targetFracIll0", "targetFracLiq0Illpos","targetHour", "targetLabor"};
		selct_bool_map(names, m, options.dist_targets_set_);

		options.dist_target_.targetLiqYratio = get_value(m, "targetLiqYratio");
		options.dist_target_.targetIllYratio = get_value(m, "targetIllYratio");
		options.dist_target_.targetKYratio = get_value(m, "targetKYratio");
		options.dist_target_.targetFracLiq0Ill0 = get_value(m, "targetFracLiq0Ill0");
		options.dist_target_.targetFracLiqNeg = get_value(m, "targetFracLiqNeg");
		options.dist_target_.targetFracIll0 = get_value(m, "targetFracIll0");
		options.dist_target_.targetFracLiq0 = get_value(m, "targetFracLiq0");
		options.dist_target_.targetFracLiq0Illpos = get_value(m, "targetFracLiq0Illpos");
		options.dist_target_.targetHour = get_value(m, "targetHour");
		options.dist_target_.targetLabor = get_value(m, "targetLabor");


	}
	if (get_value(m, "Het_opt_params"))
	{
		names = { "rho", "chi0", "chi1", "chi2", "rb_wedge", "vphi" };
		selct_bool_map(names, m, options.opt_params_set_);
		options.opt_params_.rho = get_value(m, "rho");
		options.opt_params_.chi0 = get_value(m, "chi0");
		options.opt_params_.chi1 = get_value(m, "chi1");
		options.opt_params_.chi2 = get_value(m, "chi2");
		options.opt_params_.rb_wedge = get_value(m, "rb_wedge");
		options.opt_params_.vphi = get_value(m, "vphi");
	}
	if (get_value(m, "Opt"))
	{
		if (get_value(m, "Nelder_Mead"))
		{
			options.opt_ = Opt::Nelder_Mead;
			options.opt_name_ = "Nelder_Mead";
		}
		else if (get_value(m, "ROOTS"))
		{
			options.opt_ = Opt::ROOTS;
			options.opt_name_ = "ROOTS";
		}
		else if (get_value(m, "DFNLS"))
		{
			options.opt_ = Opt::DFNLS;
			options.opt_name_ = "DFNLS";
		}
	}
	if (get_value(m, "Adj_fun"))
	{
		if (get_value(m, "KAPLAN_ADJ"))
		{
			options.adj_fun_ = Adj_fun::KAPLAN_ADJ;
		}
		else if (get_value(m, "AUCLERT_ADJ"))
		{
			options.adj_fun_ = Adj_fun::AUCLERT_ADJ;
		}
	}
	if (get_value(m, "Hour_supply"))
	{
		if (get_value(m, "Seq"))
		{
			options.hour_supply_ = Hour_supply::Seq;
			options.hour_supply_name_ = "Seq";
		}
		else if (get_value(m, "GHH"))
		{
			options.hour_supply_ = Hour_supply::GHH;
			options.hour_supply_name_ = "GHH";
		}
		else if (get_value(m, "NoSupply"))
		{
			options.hour_supply_ = Hour_supply::NoSupply;
			options.hour_supply_name_ = "NoSupply";
		}
	}
	if (get_value(m, "Process"))
	{
		if (get_value(m, "AR1"))
		{
			options.process_ = Process::AR1;
			options.process_name_ = "AR1";
		}
		else if (get_value(m, "AR1_MIX"))
		{
			options.process_ = Process::AR1_MIX;
			options.process_name_ = "AR1_MIX";
		}
		else if (get_value(m, "KAPLAN"))
		{
			options.process_ = Process::KAPLAN;
			options.process_name_ = "KAPLAN";
		}
		else if (get_value(m, "JUMP_DRIFT_MIX"))
		{
			options.process_ = Process::JUMP_DRIFT_MIX;
			options.process_name_ = "JUMP_DRIFT_MIX";
			filesystem::path model_path = filesystem::current_path();
			filesystem::path process_path = model_path;
			filesystem::current_path(process_path += "/EarningProcess/JUMP_DRIFT_MIX");
			options.earning_options_.init(ifs);
			filesystem::current_path(model_path);
		}
	}
	if (get_value(m, "Run"))
	{
		options.run_.calibration = get_value(m, "calibration");
		options.run_.solve_ss = get_value(m, "solve_ss");
		options.run_.solve_equm = get_value(m, "solve_equm");
		options.run_.solve_non_convex = get_value(m, "solve_non_convex");
		options.run_.add_innaction = get_value(m, "add_innaction");
		options.run_.pin_chi1 = get_value(m, "pin_chi1");
		options.run_.pin_meanlabeff = get_value(m, "pin_meanlabeff");
		options.run_.discrete_process = get_value(m, "discrete_process");
		options.run_.estimate_process = get_value(m, "estimate_process");
	}


	return 1;
}

bool
write_calibrations(const Calibration& cal, ofstream& of)
{
	string fold = "Calibration";

	filesystem::remove_all(fold);
	filesystem::create_directories(fold);

	filesystem::path root = filesystem::current_path();
	filesystem::current_path(filesystem::absolute(fold));

	write_map_to_file(of, cal.parameter_cal_, "parameter.csv");
	write_map_to_file(of, cal.size_cal_, "size.csv");
	write_matrix_to_file(cal.agrid_, of, "a_grid.csv");
	write_matrix_to_file(cal.bgrid_, of, "b_grid.csv");
	write_matrix_to_file(cal.DeltaTrans_, of, "DeltaTrans.csv");
	write_matrix_to_file(cal.DeltaTransCum_, of, "DeltaTransCum.csv");
	write_matrix_to_file(cal.markov_, of, "markov.csv");
	write_matrix_to_file(cal.sgrid_, of, "skill_grid.csv");
	filesystem::current_path(root);
	return 1;
}

void
renew_parameters(const std::unordered_map<std::string, parameter>& parameters, Calibration& cal)
{
	renew_map(parameters, cal.parameter_cal_);
}

void
renew_earning_parameters(const std::unordered_map<std::string, parameter>& parameters, Calibration& cal)
{
	renew_map(parameters, cal.earning_parameter_cal_);
}

void
append_parameters(const std::unordered_map<std::string, parameter>& parameters, Calibration& cal)
{
	append_map(parameters, cal.parameter_cal_);
}

aggregates
Calibration::filling_rate(aggregates theta) const
{
	parameter match_els = get_value(parameter_cal_, "match_els");
	parameter match_scale = get_value(parameter_cal_, "match_scale");
	return match_scale * std::pow(theta, -match_els);
}

