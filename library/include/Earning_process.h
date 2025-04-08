#pragma once
#include <map>
#include <string>
#include <math.h>
#include <unordered_map>

#include "newuoa_h.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "Calibration.h"
#include "Steady_state.h"
#include "Support.h"

namespace EARNING_PROCESS
{
	struct Moments
	{
		double m1;
		double m2;
		double m3;
		double m4;
		double skew;
		double kurt;
	};

	struct Earning_moments
	{
		Moments mlog;
		Moments mlev;
		Moments mlog_dz1;
		Moments mlog_dz5;

		double frac_dy1_10pct;
		double frac_dy1_20pct;
		double frac_dy1_50pct;
	};

	namespace DISCRETE
	{

		struct Inputs
		{
			const Calibration& cal;
			void init();
			int Tsim;
			int nsim = cal.earning_size("nsim");
			int Quaters = cal.earning_size("Quarters");
			int years = Quaters / 4;
			int nz1 = cal.earning_size("nsk1");
			int nz2 = cal.earning_size("nsk2");
			double dt = cal.earning_params("dt");
			double dtkfe = cal.earning_params("dtkfe");
			double dann;
			double lambda1 = cal.earning_params("lambda1");
			double lambda2 = cal.earning_params("lambda2");
			double beta1 = cal.earning_params("beta1");
			double beta2 = cal.earning_params("beta2");
			double sigma1 = cal.earning_params("sigma1");
			double sigma2 = cal.earning_params("sigma2");
			double width1 = cal.earning_params("width1");
			double width2 = cal.earning_params("width2");
			double curv1 = cal.earning_params("curv1");
			double curv2 = cal.earning_params("curv2");
			double tols_dist = cal.earning_params("tols_dist");
			int maxiter_dist = cal.earning_size("maxiter_dist");
			Eigen::ArrayXXd z1_rand;
			Eigen::ArrayXXd z2_rand;
			Eigen::MatrixXd id1;
			Eigen::MatrixXd id2;

		};

		struct Workspace
		{
			const Inputs& inputs;
			void init();
			Eigen::ArrayXd z1_grid;
			Eigen::ArrayXd z2_grid;
			Eigen::ArrayXd z_grid;
			Eigen::ArrayXd dz1_grid;
			Eigen::ArrayXd dz2_grid;
			Eigen::ArrayXXd z_sim;
			Eigen::ArrayXXd z_ann_sim;
			Eigen::ArrayXXd z_lev_sim;
			Eigen::ArrayXXi z1_sim_index;
			Eigen::ArrayXXi z2_sim_index;
			Eigen::ArrayXXd z_ann_lev_sim;
			Eigen::MatrixXd generator1;
			Eigen::MatrixXd generator2;
			Eigen::MatrixXd generator;
			Eigen::MatrixXd generator1_jump;
			Eigen::MatrixXd generator2_jump;
			Eigen::MatrixXd generator1_drift;
			Eigen::MatrixXd generator2_drift;
			Eigen::MatrixXd markov1_dt;
			Eigen::MatrixXd markov2_dt;
			Eigen::MatrixXd markov1_dt_cdf;
			Eigen::MatrixXd markov2_dt_cdf;
			Eigen::MatrixXd markov1_quarter;
			Eigen::MatrixXd markov2_quarter;
			Eigen::MatrixXd markov_quarter;
			Eigen::VectorXd dist1;
			Eigen::VectorXd dist2;
			Eigen::VectorXd dist;
			Eigen::VectorXd cdf1;
			Eigen::VectorXd cdf2;
			Eigen::VectorXd dist1_next;
			Eigen::VectorXd dist2_next;
		};

		struct Discrete_calibration
		{
			const Calibration& cal;
			Inputs& earning_inputs;
			Workspace& ws;
			Earning_moments& earning_moments;
			const Earning_Options& options = cal.options().earning_options();
			const Opt& opt = options.opt();
			const Earning_discrete_opt_params& discrete_opt_params = options.discrete_opt_params();
			const Earning_target& targets = options.dist_target();

			void init();

			std::map<std::string, Params_range> init_map;
			std::map<std::string, Targets_state> target_map;

			std::string name;
			bool display = 0;
			Roots_target roots_target = nullptr;
			Min_target min_target = nullptr;
			newuoa_dfovec* dfls_target = nullptr;

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
			symmetric_grid(const int& size, const double& low, const double& high, const double& curv, Eigen::ArrayXd& grid);

		void
			generator_jump(const int& rows, const double& mu, const double& sigma,
				const Eigen::ArrayXd& grid, const Eigen::ArrayXd& dgrid, Eigen::MatrixXd& generator);

		void
			generator_drift(const int& rows, const double& beta, const Eigen::ArrayXd& grid,
				Eigen::MatrixXd& generator);

		void
			discretization(const Inputs& params, Workspace& ws);

		void
			stationary_dist(const Inputs& params, Workspace& ws);

		void
			write_grids(const Calibration& cal, const Workspace& ws, std::ofstream& of);

		void
			estimation(const Calibration& cal, Earning_results& results, std::ofstream& of);

		void
			simulate(const Inputs& params, Workspace& ws);

		namespace DFNLS
		{
			void
				assign_parameters(const double* y, Discrete_calibration* params);

			void
				target_estimation_moments(const long n, const long mv, const double* y,
					double* v_err, void* data);
		}
	}

	namespace ESTIMATION
	{
		struct Inputs
		{
			const Calibration& cal;
			void init();
			int Tsim;
			int nsim = cal.earning_size("nsim");
			int Qburn = cal.earning_size("Qburn");
			int Tburn;
			int Quaters = cal.earning_size("Quarters");
			int Tann;
			double dt = cal.earning_params("dt");
			double dann;
			double lambda1 = cal.earning_params("lambda1");
			double lambda2 = cal.earning_params("lambda2");
			double beta1 = cal.earning_params("beta1");
			double beta2 = cal.earning_params("beta2");
			double sigma1 = cal.earning_params("sigma1");
			double sigma2 = cal.earning_params("sigma2");
			Eigen::ArrayXXd z1_rand;
			Eigen::ArrayXXd z2_rand;
			Eigen::ArrayXXd z1_jump;
			Eigen::ArrayXXd z2_jump;
		};

		struct Workspace
		{
			const Inputs& params;
			void init();
			void stationary_params();
			double mean1;
			double mean2;
			double var1;
			double var2;
			double stay_prob1;
			double stay_prob2;
			double drift1;
			double drift2;
			Eigen::ArrayXXd z1;
			Eigen::ArrayXXd z2;
			Eigen::ArrayXXd z;
			Eigen::ArrayXXd z_ann;
			Eigen::ArrayXXd z_lev;
			Eigen::ArrayXXd z_ann_lev;
		};

		struct Estimate_calibration
		{
			const Calibration& cal;
			Inputs& earning_inputs;
			Workspace& ws;
			Earning_moments& earning_moments;
			const Earning_Options& options = cal.options().earning_options();
			const Opt& opt = options.opt();
			const Earning_estimate_opt_params& estimate_opt_params = options.estimate_opt_params();
			const Earning_target& targets = options.dist_target();

			void init();

			std::map<std::string, Params_range> init_map;
			std::map<std::string, Targets_state> target_map;
			std::string name;
			bool display = 0;
			Roots_target roots_target = nullptr;
			Min_target min_target = nullptr;
			newuoa_dfovec* dfls_target = nullptr;

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
			simulate(const Inputs& params, Workspace& ws);



		void
			estimation(const Calibration& cal, Earning_results& results);

		void
			simulate_one_step(const double& jump_rand, const double& z_rand, const double& sigma,
				const double& drift, const double& stay_prob, double& pre, double& next);

		namespace DFNLS
		{
			void
				assign_parameters(const double* y, Estimate_calibration* params);


			void
				target_estimation_moments(const long n, const long mv, const double* y,
					double* v_err, void* data);
		}
	}



	template <typename Derived>
	void
		compute_moments(const Eigen::ArrayBase<Derived>& sample, const int& nsim, Moments& m)
	{
		m.m1 = sample.sum() / nsim;
		m.m2 = (sample - m.m1).pow(2).sum() / nsim;
		m.m3 = (sample - m.m1).pow(3).sum() / nsim;
		m.m4 = (sample - m.m1).pow(4).sum() / nsim;
		if (m.m2 > 0.0)
		{
			m.skew = m.m3 / (std::pow(m.m2, 1.5));
			m.kurt = m.m4 / (std::pow(m.m2, 2));
		}
		else
		{
			m.kurt = m.skew = 0.0;
		}
	}

	void
		compute_earning_moments(const Eigen::ArrayXXd& earning, const int& nsim, Earning_moments& moments);

	void
		dfls_objective(std::map<std::string, double>& target_map, const Earning_target& targets,
			const Earning_moments& moments, double* v_err);

	void
		construct_result_map(const std::map<std::string, Params_range>& init_map,
			Earning_results& results, const Earning_moments& moments);
}