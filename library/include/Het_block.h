#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>
#include <Eigen/SparseLU>
#include <unordered_map>
#include <string>
#include <memory>

#include "Calibration.h"
#include "Steady_state.h"

struct Het_SSJ_outputs;
typedef Eigen::ArrayXd het;
typedef double parameter;
typedef double aggregates;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> ArrayXXb;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;
typedef std::vector<ArrayXXb> Bool;
typedef Eigen::ArrayXXd het2;
typedef std::vector<het2> Tensor;
struct Het_Inputs;
struct Het_workspace;
struct Het_Outputs;

enum class Adj
{
	ADJ,
	ADJ_DERIV,
	ADJ_DERIV_NEXT,
};


typedef void (*MatrixFun) (const Het_Inputs&, Het_workspace&, int);

typedef void (*BindPols) (const Het_Inputs&, Het_workspace&, int);

typedef void (*BindBothPols) (const Het_Inputs&, Het_workspace& );

typedef void (*UnbindCons) (const Het_Inputs&, const Het_workspace&, const het2&, int,
	het2&, het2&, het2&, het2&);

typedef void(*UnbindRes) (const Het_Inputs&, const Het_workspace&,
	const het2&, const het2&, const het2&, const het2&, int,
	het2&, het2&, het2&, het2&);

typedef void (*ResPols) (const Het_Inputs&, Het_workspace& ws);

typedef void (*TargetHtM)(const Calibration& cal, Het_Outputs& htp, const Het_Inputs& het_inputs, Het_workspace& ws);


typedef void (*CDF)(const Het_Inputs& het_inputs, Het_workspace& ws);
typedef void (*DistStats)(const Het_Inputs& het_inputs, Het_Outputs& htp, Het_workspace& ws, std::unordered_map<std::string, double>& map);

typedef void (*ResAgg)(const Het_Inputs& het_inputs, const Het_Outputs& htp,
	SteadyState& ss);

typedef void (*SwapDist)(Het_workspace& ws, SteadyState& ss);
typedef void (*SwapVecs)(Het_workspace& ws, std::unordered_map<std::string, Eigen::MatrixXd>& map);


typedef void (*MapPols)(const Het_Inputs& het_inputs, Het_workspace& ws, std::unordered_map<std::string, Tensor>& pols_map);



struct Het_Inputs
{
	const Calibration& cal;
	void init();
	void set_income();
	void set_labor_income();
	void set_interest();
	void set_HJBdiags();
	void set_KFEdiags();
	void set_Feymandiags();
	void set_diags();
	void set_extensive_mat();
	void set_hh();
	void set_dag_inputs();

	BindPols solve_binding_pols;
	BindBothPols solve_both_binding_pols;
	UnbindCons solve_unbinding_cons;
	UnbindRes solve_unbinding_res;
	MatrixFun solve_rhs;

	SwapVecs swap_vecs;
	ResPols solve_res_pols;
	CDF solve_cdf;
	TargetHtM target_htm;
	DistStats solve_dist_stat;
	ResAgg solve_res_aggregators;
	SwapDist swap_dist;
	MapPols map_pols;

	aggregates ra;
	aggregates rb;
	aggregates rb_borrow;
	aggregates theta;
	aggregates theta_f;
	aggregates wage;
	aggregates ui;
	aggregates labor;
	aggregates profit;
	aggregates tax;
	aggregates x;
	aggregates u;
	aggregates meanlabeff;

	aggregates transfer;

	parameter rb_wedge;
	parameter illpremium;
	
	// new params
	parameter AL;
	parameter AH;
	parameter a_k;
	parameter a_alpha;
	parameter deathrate;
	parameter rho;
	parameter rho_zeta;

	parameter profdistfrac;
	parameter sigma;
	parameter delta;
	parameter chi0;
	parameter chi1;
	parameter chi2;
	parameter a_kink;
	parameter muw;
	parameter frisch;
	parameter vphi;
	parameter nu;
	parameter unemployedloss;

	parameter DeltaFeyman;
	parameter DeltaHJB;
	parameter DeltaKFE;
	parameter DeltaDecomp;
	parameter dVamin;
	parameter dVbmin;
	parameter dmax;
	int ne = 0;
	int nb = cal.size("nb");
	int na = cal.size("na");
	int nsk = cal.size("nsk");
	int nesk = cal.size("nesk");
	int nb_neg = cal.size("nb_neg");
	int nab = na * nb;
	int maxiter_hour;
	int maxiter_d;
	int maxiter_discmpc;
	double tols_discmpc;
	double tols_hour;
	double hour_high;
	double H_min;
	double tols_d;

	const Seq& a = cal.a();
	const Seq& b = cal.b();
	const Seq& da = cal.da();
	const Seq& db = cal.db();
	const Seq& tda = cal.tda();
	const Seq& tdb = cal.tdb();
	Seq dainv_diag;
	Seq dbinv_diag;
	Seq dainv;
	Seq dbinv;

	const Options& options = cal.options();
	const Model& model = options.model();
	bool FRACbNeg = options.dist_target().targetFracLiqNeg;
	bool innaction = options.run().add_innaction;
	bool isTransition = false;
	const Adj_fun& adj_fun = options.adj_fun();
	const Hour_supply& hour_supply = options.hour_supply();
	const Grid& agrid = cal.agrid();
	const Grid& bgrid = cal.bgrid();
	const Grid& dagrid = cal.dagrid();
	const Grid& dbgrid = cal.dbgrid();
	Grid dagridinv;
	Grid dbgridinv;
	het2 akinkgrid;
	het2 dmin;
	double ra_max;
	const Seq& sgrid = cal.sgrid();

	const het& ui_rule = cal.ui_rule();
	const Eigen::VectorXd& skill_dist = cal.ss_skill_dist();
	Eigen::VectorXd skill_dist_extend;
	het2 rb_structure;
	Eigen::MatrixXd status_transition;
	SpMat status_transition_expand;
	SpMat markov_transition_expand;

	Eigen::MatrixXd markov = cal.markov();

	Eigen::MatrixXd markov_diag;
	Eigen::MatrixXd markov_offdiag;
	Eigen::MatrixXd offdiagKFE;
	Eigen::MatrixXd offdiagHJB;
	Eigen::MatrixXd idnesk;
	SpMat idnab;
	SpMat idne;
	SpMat idnsk;

	Eigen::MatrixXd death_alloc;
	std::vector<SpMat> diagHJB;
	std::vector<SpMat> diagKFE;
	std::vector<SpMat> diagFeyman;
	std::vector<SpMat> markov_diag_spmat;
	het inc_labor;
	het after_tax_wage;
	het hour_ghh;
	het disutility_hour_ghh;
	het2 a_drift;
	het2 b_drift;
	Tensor inc;
	
};

struct Het_policies
{
	const Het_Inputs& het_inputs;
	void init();
	Tensor c;
	Tensor cb;
	Tensor cem;
	Tensor cb_em;
	Tensor d;
	Tensor hour;
	Tensor labor;
	Tensor adj;
	Tensor sa;
	Tensor sb;
};

struct Het_workspace
{
	const Het_Inputs& het_inputs;
	const Options& options = het_inputs.options;
	const Model& model = options.model();
	bool FRACbNeg = options.dist_target().targetFracLiqNeg;
	void init();
	void init_bellman();
	void init_dist();
	void delete_solvers();
	Eigen::MatrixXd V_init;
	Eigen::MatrixXd V;

	double eps;
	int iter = 0;
	double tols_pols;
	int maxiter_policy;

	/*Upwinds*/
	Tensor VaB;
	Tensor VaF;
	Tensor VbB;
	Tensor VbF;

	//New scheme
	Tensor cF;
	Tensor cB;
	Tensor hF;
	Tensor hB;

	Tensor scB;
	Tensor scF;
	Tensor sbB;
	Tensor sbF;

	Tensor sdFB;
	Tensor sdFF;
	Tensor sdBF;
	Tensor sdBB;

	Tensor sd0;
	Tensor sdtemp;


	Tensor FdB;
	Tensor FdF;

	Tensor dBB;
	Tensor dFB;
	Tensor dBF;
	Tensor dFF;

	Tensor sbFB;
	Tensor sbFF;
	Tensor sbBB;
	Tensor sbBF;
	Tensor sb0B;
	Tensor sb0F;

	Tensor saFB;
	Tensor saFF;
	Tensor saBB;
	Tensor saBF;

	Tensor Fd0;
	//Halmilton
	Tensor HFB;
	Tensor HFF;
	Tensor HBF;
	Tensor HBB;
	Tensor H0F;
	Tensor H0B;
	Tensor H0;
	Tensor H00;
	Tensor c00;
	Tensor hour00;
	Tensor Fd00;


	Tensor HF0;
	Tensor HB0;
	Tensor dF0;
	Tensor dB0;
	Tensor cF0;
	Tensor cB0;
	Tensor hourF0;
	Tensor hourB0;

	//Policies
	Tensor c;
	Tensor d;
	Tensor sa;
	Tensor sb;
	Tensor sd;
	Tensor adriftF;
	Tensor adriftB;
	Tensor bdriftF;
	Tensor bdriftB;
	Tensor hour;
	Tensor adj;
	Tensor labor;
	Tensor c0;
	Tensor d0;
	Tensor h0;

	Tensor d_high;
	Tensor d_low;
	Tensor Fd_low;

	Tensor utilityF;
	Tensor utilityB;
	Tensor utility0;
	//ind
	Bool indFB;
	Bool indFF;
	Bool indBB;
	Bool indBF;
	Bool ind0B;
	Bool ind0F;
	Bool ind0;

	Bool validFB;
	Bool validFF;
	Bool validBB;
	Bool validBF;
	Bool valid0B;
	Bool valid0F;

	Tensor hour_min;
	Tensor fhour_max;
	Tensor fhour_min;

	Tensor hour_high;
	Tensor hour_low;
	Tensor fhour_high;
	Tensor fhour_low;
	Tensor fhour;
	Tensor df;


	Tensor adj1;
	Tensor effdisc;
	Eigen::MatrixXd muc;
	Eigen::MatrixXd badj;

	Tensor mpc;
	Tensor mpd_a;
	Tensor mpd_b;

	Eigen::MatrixXd next;
	Eigen::MatrixXd rhs;

	Eigen::MatrixXd subeff1ass;
	Eigen::MatrixXd subeff2ass;
	Eigen::MatrixXd wealtheff1ass;
	Eigen::MatrixXd wealtheff2ass;
	Eigen::MatrixXd ccum1;
	Eigen::MatrixXd ccum2;
	Eigen::MatrixXd ccum4;
	Eigen::MatrixXd dist_init;
	Eigen::MatrixXd dist;

	Eigen::MatrixXd abdist;
	Eigen::VectorXd adist;
	Eigen::VectorXd bdist;
	Eigen::VectorXd acdf;
	Eigen::VectorXd bcdf;



	Eigen::VectorXd illiquid_wealth_dist;
	Eigen::VectorXd liquid_wealth_dist;

	std::vector<Eigen::PardisoLU<SpMat>*> solvers;
	std::vector<SpMat> diagdiscmpc;
	std::vector<SpMat> assetHJB;
	std::vector<SpMat> distKFE;
	std::vector<SpMat> zeros; // Used for SSJ expectations
	std::vector<SpMat> FeynmanKac;
	std::vector<SpMat> mpcHJB;
	double decomp_diag;
};


struct Het_Outputs
{
	const Calibration& cal;
	const het& b_dist_grid;
	const het& a_dist_grid;
	const Het_target& targets = cal.options().dist_target();
	Het_convergent_fun het_fun;
	Het_one_step_fun het_one_step_fun;
	Het_moments_fun moments_fun;
	void init();
	double a0_close_frac;
	double b0_close_frac;
	double a0_close_model;
	double b0_close_model;
	double b0_lb;
	double b0_ub;
	double output_data;
	double tols_dist;
	double tols_policy;

	int maxiter_dist;
	int maxiter_policy;

	/*Parameters when computing outputs*/
	double tfp = 1;
	double deprec;
	double govdebtYratio;
	double mc = 0;
	double emkp = 0;
	double delta = 0;
	double alpha = 0;
	double Kub = 0;
	double targetKYratio;
	double targetKNratio;
	/************/

	/*Ouput from het block*/
	double Ea = 0;
	double Ea_net = 0;
	double Eb = 0;
	double Ea_EM;
	double Eb_EM;
	double Ea_UNEM;
	double Eb_UNEM;
	double K = 0;
	double Y = 0;
	double KNratio = 0;
	double LiqYratio = 0;
	double IllYratio = 0;
	double KYratio = 0;
	double FracLiq0Ill0 = 0;
	double FracLiqNeg = 0;
	double FracIll0 = 0;
	double FracLiq0Illpos = 0;
	double FracLiq0 = 0;
	double FracLiq0Ill0_close = 0;
	double FracLiqNeg_close = 0;
	double FracLiq0Illpos_close = 0;
	double FracLiq0_close = 0;

	double FracLiq0Ill0_close_EM = 0;
	double FracLiqNeg_close_EM = 0;
	double FracLiq0Illpos_close_EM = 0;
	double FracLiq0_close_EM = 0;

	double FracLiq0Ill0_close_UNEM = 0;
	double FracLiqNeg_close_UNEM = 0;
	double FracLiq0Illpos_close_UNEM = 0;
	double FracLiq0_close_UNEM = 0;

	double A_GINI = 0.0;
	double A_TOP_01 = 0.0;
	double A_TOP_1 = 0.0;
	double A_TOP_10 = 0.0;
	double A_BOTTOM_50 = 0.0;
	double A_BOTTOM_25 = 0.0;

	double B_GINI = 0.0;
	double B_TOP_01 = 0.0;
	double B_TOP_1 = 0.0;
	double B_TOP_10 = 0.0;
	double B_BOTTOM_50 = 0.0;
	double B_BOTTOM_25 = 0.0;

	double A_GINI_EM = 0.0;
	double A_TOP_01_EM = 0.0;
	double A_TOP_1_EM = 0.0;
	double A_TOP_10_EM = 0.0;
	double A_BOTTOM_50_EM = 0.0;
	double A_BOTTOM_25_EM = 0.0;

	double B_GINI_EM = 0.0;
	double B_TOP_01_EM = 0.0;
	double B_TOP_1_EM = 0.0;
	double B_TOP_10_EM = 0.0;
	double B_BOTTOM_50_EM = 0.0;
	double B_BOTTOM_25_EM = 0.0;

	double A_GINI_UNEM = 0.0;
	double A_TOP_01_UNEM = 0.0;
	double A_TOP_1_UNEM = 0.0;
	double A_TOP_10_UNEM = 0.0;
	double A_BOTTOM_50_UNEM = 0.0;
	double A_BOTTOM_25_UNEM = 0.0;

	double B_GINI_UNEM = 0.0;
	double B_TOP_01_UNEM = 0.0;
	double B_TOP_1_UNEM = 0.0;
	double B_TOP_10_UNEM = 0.0;
	double B_BOTTOM_50_UNEM = 0.0;
	double B_BOTTOM_25_UNEM = 0.0;



	double Ehour = 0;
	double Elabor = 0;
	double Egrosslabinc = 0;

};



struct Final_ouputs
{
	double K;
	double Elabor;
	double pi;
	double Gov_debt;
	double tfp;
	double eqI;
	double eqK;
	double res_bond;
	double eqW_job;
	double TobinQ;
	double u;
	double theta;
};





void
init_map_tensor_like(std::unordered_map<std::string, Tensor>& tens, const std::unordered_map<std::string, Tensor>& tar);


template <typename Derived>
auto
util_cons(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived>& c)
{
	bool b = (het_inputs.sigma == 1.0);
	double x = (b ? 0 : 1 / (1 - het_inputs.sigma));
	return b * c.log() + !b * x * c.pow(1 - het_inputs.sigma);
}

template <typename Derived>
auto
disutil_hour(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived>& hour)
{
	return -het_inputs.vphi * hour.pow(1 + 1 / het_inputs.frisch) / (1 + 1 / het_inputs.frisch);
}

template <typename Derived>
auto
utility1(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived>& c)
{
	return c.pow(-het_inputs.sigma);
}

template <typename Derived>
auto
utility1inv(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived>& Vb)
{
	return Vb.pow(-1 / het_inputs.sigma);
}

template <typename Derived>
auto
utility2(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived>& hour)
{
	return -het_inputs.vphi * hour.pow(1 / het_inputs.frisch);
}

template <typename Derived>
auto
utility2inv(const Het_Inputs& het_inputs, const Eigen::ArrayBase<Derived>& Vh)
{
	return (Vh / het_inputs.vphi).pow(het_inputs.frisch);
}


double
util_cons(const Het_Inputs& het_inputs, double c);


double
utility2(const Het_Inputs& het_inputs, double hour);

double
utility2inv(const Het_Inputs& het_inputs, double Vh);

double
utility1inv(const Het_Inputs& het_inputs, double Vb);

double
utility1(const Het_Inputs& het_inputs, double c);

double
disutil_hour(const Het_Inputs& het_inputs, double hour);

void
solve_cum(const Het_Inputs& het_inputs, Het_workspace& ws);

void
solve_cum(const Het_Inputs& het_inputs, const Tensor& pols, Het_workspace& ws, int period, Eigen::MatrixXd& cum);

void
solve_convergent_decomp(const Het_Inputs& het_inputs, const Eigen::MatrixXd& rhs, Het_workspace& ws, Eigen::MatrixXd& decomp);

void
init_decomposition(const Het_Inputs& het_inputs, const Eigen::MatrixXd& rhs, Het_workspace& ws, Eigen::MatrixXd& decomp);

void
solve_cumHJB(const Het_Inputs& het_inputs, const Tensor& pols, Het_workspace& ws, Eigen::MatrixXd& cum);

void
construct_cumHJB(const Het_Inputs& het_inputs, Het_workspace& ws);

void
solve_ss_decomp(const Het_Inputs& het_inputs, Het_workspace& ws);

template<typename Derived>
void
solve_decompHJB(const Het_Inputs& het_inputs, const Eigen::DenseBase<Derived>& rhs, const Eigen::MatrixXd& decomp, Het_workspace& ws)
{
#pragma omp parallel for
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		ws.rhs.col(r) = rhs.col(r) * het_inputs.DeltaDecomp + het_inputs.DeltaDecomp * decomp * het_inputs.markov_offdiag.row(r).transpose();
		ws.next.col(r) = ws.solvers[r]->solve(ws.rhs.col(r));
	}
}
void
construct_diag_dismpc(const Het_Inputs& het_inputs, const Tensor& pols, double dig, Het_workspace& ws);

void
construct_decompHJB(const Het_Inputs& het_inputs, Het_workspace& ws);

void
prepare_decomposition(const Het_Inputs& het_inputs, Het_workspace& ws);

void
solve_mpcs(const Het_Inputs& het_inputs, Het_workspace& ws);


template <typename Derived>
double
probability_reshape(const het& adist_grid, const het& bdist_grid, const Eigen::DenseBase<Derived>& dist, int size,
	int rows, int cols, double a, double b)
{
	int n = rows * cols;

	double prob = 0.0;
#pragma omp parallel for reduction(+: prob)
	for (int r = 0; r < size; ++r)
	{
		prob += probability(adist_grid, bdist_grid, dist.segment(n * r, n).reshaped(rows, cols), a, b);
	}
	return prob;
}

template <typename Derived>
double
probability_reshape_colwise(const het& dist_grid, const Eigen::DenseBase<Derived>& dist, int size,
	int rows, int cols, double xa)
{
	int n = rows * cols;

	double prob = 0.0;
#pragma omp parallel for reduction(+: prob)
	for (int r = 0; r < size; ++r)
	{
		prob += probability(dist_grid, dist.segment(n * r, n).reshaped(rows, cols).colwise().sum(), xa);
	}
	return prob;
}

template <typename Derived>
double
probability_reshape_rowwise(const het& dist_grid, const Eigen::DenseBase<Derived>& dist, int size,
	int rows, int cols, double xa)
{
	int n = rows * cols;

	double prob = 0.0;
#pragma omp parallel for reduction(+: prob)
	for (int r = 0; r < size; ++r)
	{
		prob += probability(dist_grid, dist.segment(n * r, n).reshaped(rows, cols).rowwise().sum(), xa);
	}
	return prob;
}

template <typename Derived, typename Derived1>
double
expectation(const Eigen::DenseBase<Derived1>& grid, const Eigen::DenseBase<Derived>& dist, int nesk, int nab)
{

	double sum = 0.0;
#pragma omp parallel for reduction(+: sum)
	for (int r = 0; r < nesk; ++r)
	{
		sum += (grid.reshaped().array() * dist.segment(nab * r, nab).array()).sum();
	}
	return sum;
}

template <typename Derived>
double
expectation(const Tensor& ten, const Eigen::DenseBase<Derived>& dist, int nesk, int nab)
{

	double sum = 0.0;
#pragma omp parallel for reduction(+: sum)
	for (int r = 0; r < nesk; ++r)
	{
		sum += (ten[r].reshaped() * dist.segment(nab * r, nab).array()).sum();
	}
	return sum;
}

template <typename Derived>
double
expectation(const Tensor& ten, const Eigen::DenseBase<Derived>& dist, int beg, int end, int nab)
{

	double sum = 0.0;
#pragma omp parallel for reduction(+: sum)
	for (int r = beg; r < end; ++r)
	{
		sum += (ten[r].reshaped() * dist.segment(nab * r, nab).array()).sum();
	}
	return sum;
}

template <typename Derived, typename Derived1>
double
expectation(const Tensor& ten, const Eigen::DenseBase<Derived1>& dist, int ns, int nab, const Eigen::ArrayBase<Derived>& skill)
{

	double sum = 0.0;
#pragma omp parallel for reduction(+: sum)
	for (int r = 0; r < ns; ++r)
	{
		sum += (ten[r].reshaped() * dist.segment(nab * r, nab).array()).sum() * skill(r);
	}
	return sum;
}

