#pragma once
#include <Eigen/Dense>
#include <unordered_map>
#include <set>
#include <string>
#include <fstream>

#include "Support.h"

typedef Eigen::ArrayXXd het2;
typedef std::vector<het2> Tensor;
typedef Eigen::ArrayXd het;
typedef Eigen::ArrayXXd Grid;
typedef Eigen::ArrayXd Seq;

struct Het_target
{
	bool targetLiqYratio = 1;
	bool targetIllYratio = 1;
	bool targetKYratio = 0;
	bool targetFracLiq0Ill0 = 0;
	bool targetFracLiqNeg = 0;
	bool targetFracIll0 = 0;
	bool targetFracLiq0Illpos = 0;
	bool targetFracLiq0 = 0;
	bool targetHour = 0;
	bool targetLabor = 0;
};

struct Het_opt_params
{
	bool rho = 1;
	bool chi0 = 0;
	bool chi1 = 1;
	bool chi2 = 0;
	bool rb_wedge = 0;
	bool vphi = 0;
};

struct Earning_target
{
	bool targetVarLog = 1;
	bool targetVarD1Log = 1;
	bool targetSkewD1Log = 1;
	bool targetKurtD1Log = 1;
	bool targetVarD5Log = 1;
	bool targetSkewD5Log = 1;
	bool targetKurtD5Log = 1;
	bool targetFracD1Less10 = 1;
	bool targetFracD1Less20 = 1;
	bool targetFracD1Less50 = 1;
};



struct Earning_estimate_opt_params
{
	bool beta1 = 1;
	bool beta2 = 1;
	bool lambda1 = 1;
	bool lambda2 = 1;
	bool sigma1 = 1;
	bool sigma2 = 1;

};

struct Earning_discrete_opt_params
{
	bool width1 = 1;
	bool width2 = 1;
	bool curv1 = 1;
	bool curv2 = 1;
};

struct Run
{
	bool estimate_process = 1;
	bool discrete_process = 1;
	bool add_innaction = 1;
	bool pin_chi1 = 1;
	bool pin_meanlabeff = 1;
	bool fiscal_shock = 1;
	bool calibration = 1;
	bool solve_equm = 1;
	bool solve_ss = 1;
};



enum class Opt
{
	Nelder_Mead,
	DFNLS,
	ROOTS
};


enum class Process
{
	AR1,
	AR1_MIX,
	KAPLAN,
	JUMP_DRIFT_MIX
};

enum class Model
{
	KAPLAN,              // Two asset HANK, Kaplan, etc. (2018)
};





enum class Adj_fun
{
	KAPLAN_ADJ,
	AUCLERT_ADJ
};

enum class Hour_supply
{
	Seq,
	GHH,
	NoSupply
};







class Model_Options
{
public:
	Model_Options() = default;
	void init(std::ifstream& ifs);
	const Model& model() const
	{
		return model_;
	}
	const std::string& model_string() const
	{
		return model_string_;
	}
	const std::string& model_experiment() const
	{
		return experiment;
	}

private:
	Model model_ = Model::KAPLAN;
	std::string model_string_ = "KAPLAN";
	std::string experiment = "Baseline";
};

class Earning_Options
{
public:
	Earning_Options() = default;
	friend bool
		read_options(Earning_Options& options, std::ifstream& in);

	void init(std::ifstream& ifs);

	const Opt& opt() const
	{
		return opt_;
	}
	const Earning_target& dist_target() const
	{
		return dist_target_;
	}
	const Earning_estimate_opt_params& estimate_opt_params() const
	{
		return estimate_opt_params_;
	}
	const Earning_discrete_opt_params& discrete_opt_params() const
	{
		return discrete_opt_params_;
	}
	const std::string& opt_name() const
	{
		return opt_name_;
	}
	const std::set<std::string>& estimate_opt_params_set() const
	{
		return estimate_opt_params_set_;
	}
	const std::set<std::string>& discrete_opt_params_set() const
	{
		return discrete_opt_params_set_;
	}

	const std::set<std::string>& dist_targets_set() const
	{
		return dist_targets_set_;
	}
private:
	Opt opt_ = Opt::DFNLS;
	Earning_target dist_target_;
	Earning_estimate_opt_params estimate_opt_params_;
	Earning_discrete_opt_params discrete_opt_params_;
	std::string opt_name_;
	std::set<std::string> estimate_opt_params_set_;
	std::set<std::string> discrete_opt_params_set_;
	std::set<std::string> dist_targets_set_;
};

class Options
{
public:
	Options() = default; 
	Options(const Model_Options& mod) : model_options_(mod) {}
	friend bool
		read_options(Options& options, std::ifstream& in);

	void init(std::ifstream& ifs);

	const Model& model() const
	{
		return model_options_.model();
	}
	const Earning_Options& earning_options() const
	{
		return earning_options_;
	}
	const Het_target& dist_target() const
	{
		return dist_target_;
	}

	const Opt& opt() const
	{
		return opt_;
	}
	const Adj_fun& adj_fun() const
	{
		return adj_fun_;
	}
	const Hour_supply& hour_supply() const
	{
		return hour_supply_;
	}
	const Het_opt_params& estimate_opt_params() const
	{
		return opt_params_;
	}
	const Process& process() const
	{
		return process_;
	}
	const Run& run() const
	{
		return run_;
	}

	const std::string& fiscal_name() const
	{
		return fiscal_name_;
	}
	const std::string& fiscal_var() const
	{
		return fiscal_var_;
	}
	const std::string& model_name() const
	{
		return model_options_.model_string();
	}

	const std::string& shock_name() const
	{
		return shock_name_;
	}
	const std::string& opt_name() const
	{
		return opt_name_;
	}
	const std::string& process_name() const
	{
		return process_name_;
	}
	const std::string& hour_supply_name() const
	{
		return hour_supply_name_;
	}
	const std::set<std::string>& estimate_opt_params_set() const
	{
		return opt_params_set_;
	}
	const std::set<std::string>& dist_targets_set() const
	{
		return dist_targets_set_;
	}
private:
	Model_Options model_options_;
	Earning_Options earning_options_;
	Het_target dist_target_;

	Opt opt_ = Opt::ROOTS;
	Adj_fun adj_fun_ = Adj_fun::AUCLERT_ADJ;
	Het_opt_params opt_params_;
	Process process_ = Process::AR1;
	Hour_supply hour_supply_ = Hour_supply::Seq;

	Run run_;
	std::string shock_name_;
	std::string fiscal_name_;
	std::string fiscal_var_;
	std::string opt_name_;
	std::string process_name_;
	std::string hour_supply_name_;
	std::set<std::string> dist_targets_set_;
	std::set<std::string> opt_params_set_;


};

typedef double parameter; typedef double aggregates;
typedef Eigen::ArrayXd het;




class Calibration {
public:
	Calibration() = default;
	Calibration(const Options& opt) : options_(opt) { }

	friend bool
		read_calibrations(Calibration& cal, std::ifstream& in);

	friend bool
		write_calibrations(const Calibration& cal, std::ofstream& of);

	friend bool
		write_grids(const Calibration& cal, std::ofstream& of);

	friend void
		renew_parameters(const std::unordered_map<std::string, parameter>&, Calibration&);

	friend void
		append_parameters(const std::unordered_map<std::string, parameter>&, Calibration&);

	friend void
		renew_earning_parameters(const std::unordered_map<std::string, parameter>&, Calibration&);

	void init(std::ifstream& ifs);

	void make_grid();
	void set_Delta_grid();
	/*Interfaces to get specific calibration value. */
	const parameter&
		params(const std::string& name) const
	{
		return get_value(parameter_cal_, name);
	}

	const parameter&
		earning_params(const std::string& name) const
	{
		return get_value(earning_parameter_cal_, name);
	}

	const int&
		earning_size(const std::string& name) const
	{
		return get_value(earning_size_cal_, name);
	}

	const int&
		size(const std::string& name) const
	{
		return get_value(size_cal_, name);
	}

	/*Interfaces to get calibration details.*/
	const std::unordered_map<std::string, parameter>&
		parameter_map() const
	{
		return parameter_cal_;
	}

	const std::unordered_map<std::string, parameter>&
		earning_parameter_map() const
	{
		return earning_parameter_cal_;
	}

	const Grid&
		agrid() const
	{
		return agrid_;
	}

	const Seq& 
		a() const
	{
		return a_;
	}

	const Seq&
		b() const
	{
		return b_;
	}

	const Seq&
		DeltaTransCum() const
	{
		return DeltaTransCum_;
	}

	const Seq&
		DeltaTrans() const
	{
		return DeltaTrans_;
	}

	const Grid&
		bgrid() const
	{
		return bgrid_;
	}

	const Grid&
		dbgrid() const
	{
		return dbgrid_;
	}
	const Grid&
		dagrid() const
	{
		return dagrid_;
	}
	const Grid&
		tdbgrid() const
	{
		return tdbgrid_;
	}
	const Grid&
		tdagrid() const
	{
		return tdagrid_;
	}


	const Seq&
		sgrid() const
	{
		return sgrid_;
	}

	const Seq&
		tda() const
	{
		return tda_;
	}

	const Seq&
		tdb() const
	{
		return tdb_;
	}

	const Seq&
		da() const
	{
		return da_;
	}

	const Seq&
		db() const
	{
		return db_;
	}

	const het&
		ui_rule()const
	{
		return ui_rule_;
	}

	const Eigen::VectorXd&
		ss_skill_dist() const
	{
		return skill_ss_dist_;
	}

	const Eigen::MatrixXd&
		markov() const
	{
		return markov_;
	}

	aggregates
		filling_rate(aggregates theta) const;

	aggregates
		match(aggregates theta, aggregates v) const 
	{
		return filling_rate(theta) * v;
	}
	
	const Options& 
		options() const
	{
		return options_;
	}

private:

	void
		set_asset_grid();

	void
		set_skill_grid();

	void
		set_ui_rule();




	Options options_;
	std::unordered_map<std::string, parameter> parameter_cal_;
	std::unordered_map<std::string, int> size_cal_;
	std::unordered_map<std::string, parameter> earning_parameter_cal_;
	std::unordered_map<std::string, int> earning_size_cal_;	
	Grid agrid_;
	Grid bgrid_;
	Grid dagrid_;
	Grid dbgrid_;
	Grid tdagrid_;
	Grid tdbgrid_;

	Seq sgrid_;
	//Seq DeltaTrans_;
	Seq DeltaTransCum_;
	Seq DeltaTrans_;
	Seq a_;
	Seq b_;
	Seq da_;
	Seq db_;
	Seq tda_;
	Seq tdb_;

	Eigen::MatrixXd markov_;
	Eigen::VectorXd skill_ss_dist_;
	het ui_rule_;
};


struct Earning_results
{
	std::unordered_map<std::string, parameter> params;
	std::unordered_map<std::string, parameter> moments;
};

bool
write_calibrations(const Calibration& cal, std::ofstream& of);

void
write_earning_results(std::ofstream& of, const Earning_results& results, const std::string& filename);

bool
write_grids(const Calibration& cal, std::ofstream& of);

void
renew_parameters(const std::unordered_map<std::string, parameter>&, Calibration&);

void
renew_earning_parameters(const std::unordered_map<std::string, parameter>&, Calibration&);

void
append_parameters(const std::unordered_map<std::string, parameter>&, Calibration&);

bool
read_calibrations(Calibration& cal, std::ifstream& in);

bool
read_options(Options& options, std::ifstream& in);

bool
read_options(Earning_Options& options, std::ifstream& in);
