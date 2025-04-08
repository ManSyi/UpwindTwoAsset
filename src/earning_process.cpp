#include <map>
#include <string>
#include <math.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <memory>
#include <iterator>
#include <chrono>
#include <unordered_map>
#include <omp.h>
#include "newuoa_h.h"
#include "Solver.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/KroneckerProduct> 
#include <EigenRand/EigenRand>

#include "Earning_process.h"
#include "Support.h"
#include "Calibration.h"
#include "Steady_state.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

namespace EARNING_PROCESS
{


	void
		dfls_objective(std::map<std::string, Targets_state>& target_map, const Earning_target& targets,
			const Earning_moments& moments, double* v_err)
	{

		auto ptr = target_map.begin();

		if (targets.targetVarLog)
		{
			ptr = target_map.find("targetVarLog");
			ptr->second.implied = moments.mlog.m2;
			v_err[distance(target_map.begin(), ptr)]
				= ptr->second.implied / ptr->second.actual - 1.0;
		}
		if (targets.targetVarD1Log)
		{
			ptr = target_map.find("targetVarD1Log");
			ptr->second.implied = moments.mlog_dz1.m2;
			v_err[distance(target_map.begin(), ptr)]
				= ptr->second.implied / ptr->second.actual - 1.0;
		}
		if (targets.targetSkewD1Log)
		{
			ptr = target_map.find("targetSkewD1Log");
			ptr->second.implied = moments.mlog_dz1.skew;
			v_err[distance(target_map.begin(), ptr)]
				= ptr->second.implied / ptr->second.actual - 1.0;
		}
		if (targets.targetKurtD1Log)
		{
			ptr = target_map.find("targetKurtD1Log");
			ptr->second.implied = moments.mlog_dz1.kurt;
			v_err[distance(target_map.begin(), ptr)]
				= std::sqrt(0.5) * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (targets.targetVarD5Log)
		{
			ptr = target_map.find("targetVarD5Log");
			ptr->second.implied = moments.mlog_dz5.m2;
			v_err[distance(target_map.begin(), ptr)]
				= ptr->second.implied / ptr->second.actual - 1.0;
		}
		if (targets.targetSkewD5Log)
		{
			ptr = target_map.find("targetSkewD5Log");
			ptr->second.implied = moments.mlog_dz5.skew;
			v_err[distance(target_map.begin(), ptr)]
				= std::sqrt(0.5) * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (targets.targetKurtD5Log)
		{
			ptr = target_map.find("targetKurtD5Log");
			ptr->second.implied = moments.mlog_dz5.kurt;
			v_err[distance(target_map.begin(), ptr)]
				= std::sqrt(0.5) * (ptr->second.implied / ptr->second.actual - 1.0);
		}
		if (targets.targetFracD1Less10)
		{
			ptr = target_map.find("targetFracD1Less10");
			ptr->second.implied = moments.frac_dy1_10pct;
			v_err[distance(target_map.begin(), ptr)]
				= ptr->second.implied / ptr->second.actual - 1.0;
		}
		if (targets.targetFracD1Less20)
		{
			ptr = target_map.find("targetFracD1Less20");
			ptr->second.implied = moments.frac_dy1_20pct;
			v_err[distance(target_map.begin(), ptr)]
				= ptr->second.implied / ptr->second.actual - 1.0;
		}
		if (targets.targetFracD1Less50)
		{
			ptr = target_map.find("targetFracD1Less50");
			ptr->second.implied = moments.frac_dy1_50pct;
			v_err[distance(target_map.begin(), ptr)]
				= ptr->second.implied / ptr->second.actual - 1.0;
		}
	}


	void
		compute_earning_moments(const Eigen::ArrayXXd& earning, const int& nsim, Earning_moments& moments)
	{
		compute_moments(earning.row(0), nsim, moments.mlog);
		compute_moments(earning.row(1) - earning.row(0), nsim, moments.mlog_dz1);
		compute_moments(earning.row(4) - earning.row(0), nsim, moments.mlog_dz5);
		moments.frac_dy1_10pct = double(((earning.row(1) - earning.row(0)).abs() < 0.1).count()) / nsim;
		moments.frac_dy1_20pct = double(((earning.row(1) - earning.row(0)).abs() < 0.2).count()) / nsim;
		moments.frac_dy1_50pct = double(((earning.row(1) - earning.row(0)).abs() < 0.5).count()) / nsim;
	}



	void
		construct_result_map(const std::map<std::string, Params_range>& init_map, 
			Earning_results& results, const Earning_moments& moments)
	{
		for (auto& e : init_map)
		{
			results.params.emplace(e.first, e.second.xcur);
		}
		results.moments.emplace("VarLog", moments.mlog.m2);
		results.moments.emplace("VarD1Log", moments.mlog_dz1.m2);
		results.moments.emplace("SkewD1Log", moments.mlog_dz1.skew);
		results.moments.emplace("KurtD1Log", moments.mlog_dz1.kurt);
		results.moments.emplace("VarD5Log", moments.mlog_dz5.m2);
		results.moments.emplace("SkewD5Log", moments.mlog_dz5.skew);
		results.moments.emplace("KurtD5Log", moments.mlog_dz5.kurt);
		results.moments.emplace("FracD1Less10", moments.frac_dy1_10pct);
		results.moments.emplace("FracD1Less20", moments.frac_dy1_20pct);
		results.moments.emplace("FracD1Less50", moments.frac_dy1_50pct);
	}

	namespace DISCRETE
	{
		void
			Inputs::init()
		{
			Tsim = Quaters / dt;
			dann = 4 / dt;
			Eigen::Rand::P8_mt19937_64 urng1{ 10142 };
			Eigen::Rand::P8_mt19937_64 urng2{ 11445 };
			Eigen::Rand::UniformRealGen<double> uniform_gen1;
			Eigen::Rand::UniformRealGen<double> uniform_gen2;
			z1_rand.resize(Tsim, nsim);
			z2_rand.resizeLike(z1_rand);
			z1_rand = uniform_gen1.generateLike(z1_rand, urng1);
			z2_rand = uniform_gen2.generateLike(z2_rand, urng2);
			id1.setIdentity(nz1, nz1);
			id2.setIdentity(nz2, nz2);
		}
		void Workspace::init()
		{
			z1_grid.setZero(inputs.nz1);
			z2_grid.setZero(inputs.nz2);
			z_grid.setZero(inputs.nz1 * inputs.nz2);
			dz1_grid.resize(inputs.nz1 - 1);
			dz2_grid.resize(inputs.nz2 - 1);
			z_sim.resize(inputs.Tsim, inputs.nsim);
			z_lev_sim.resizeLike(z_sim);
			z1_sim_index.resizeLike(z_sim);
			z2_sim_index.resizeLike(z_sim);
			z_ann_sim.resize(inputs.years, inputs.nsim);
			z_ann_lev_sim.resizeLike(z_ann_sim);

			generator1.resize(inputs.nz1, inputs.nz1);
			generator1_jump.resizeLike(generator1);
			generator1_drift.resizeLike(generator1);
			generator.setZero(inputs.nz1 * inputs.nz2, inputs.nz1 * inputs.nz2);
			generator2.resize(inputs.nz2, inputs.nz2);
			generator2_jump.resizeLike(generator2);
			generator2_drift.resizeLike(generator2);

			markov1_dt.resizeLike(generator1);
			markov2_dt.resizeLike(generator2);

			markov1_dt_cdf.resizeLike(generator1);
			markov2_dt_cdf.resizeLike(generator2);
			markov1_quarter.resizeLike(generator1);
			markov2_quarter.resizeLike(generator2);
			markov_quarter.setZero(inputs.nz1 * inputs.nz2, inputs.nz1 * inputs.nz2);

			generator1_drift.setZero();
			generator2_drift.setZero();
			generator1_jump.setZero();
			generator2_jump.setZero();
			dist1.setZero(inputs.nz1);
			dist2.setZero(inputs.nz2);
			dist.setZero(inputs.nz1 * inputs.nz2);
			cdf1.resizeLike(dist1);
			cdf2.resizeLike(dist2);

			dist1_next.setZero(inputs.nz1);
			dist2_next.setZero(inputs.nz2);
		}

		void
			Discrete_calibration::init()
		{
			int wedth = 16;
			display_init_params(cal.earning_parameter_map(), wedth,
				options.discrete_opt_params_set(), init_map);
			display_dist_targets(cal.earning_parameter_map(), wedth, options.dist_targets_set(), target_map);

			name = "EARNING_DISCRETE";
			tols_roots = cal.earning_params("tols_roots");
			maxiter_roots = cal.earning_size("maxiter_roots");
			step_size = cal.earning_params("step_size");
			maxiter_mins = cal.earning_size("maxiter_mins");
			tols_mins = cal.earning_params("tols_mins");

			maxfun_multip = 200;
			rhobeg = cal.earning_params("rhobeg");
			rhoend = cal.earning_params("rhoend");
			ndfls = cal.earning_size("ndfls");
			num_thread = cal.size("num_thread");


			dfls_target = DFNLS::target_estimation_moments;
		}

		void
			symmetric_grid(const int& size, const double& low, const double& high, const double& curv, Eigen::ArrayXd& grid)
		{
			int n2 = size / 2 + 1;
			if (size % 2 != 0)
			{
				set_grid(n2, 0.0, high, curv, grid.segment(size / 2, n2));
				grid.segment(0, n2 - 1) = -grid.segment(size / 2 + 1, n2 - 1).reverse();
			}
			else
			{
				set_grid(n2, 0.0, high, curv, grid.segment(size / 2 - 1, n2));
				grid.segment(0, n2 - 1) = -grid.segment(size / 2, n2 - 1).reverse();
			}

		}

		void
			generator_jump(const int& rows,  const double& mu, const double& sigma,
				const Eigen::ArrayXd& grid, const Eigen::ArrayXd& dgrid, Eigen::MatrixXd& generator)
		{
			int j = 0;
#pragma omp parallel for private(j)
			for (int i = 0; i < rows; ++i)
			{
				generator(i, 0) = norm_cdf(mu, sigma, grid(0) + 0.5 * dgrid(0));
				for (j = 1; j < rows - 1; ++j)
				{
					generator(i, j) = norm_cdf(mu, sigma, grid(j) + 0.5 * dgrid(j))
						- norm_cdf(mu, sigma, grid(j) - 0.5 * dgrid(j - 1));
				}
				generator(i, j) = 1.0 - norm_cdf(mu, sigma, grid(j) - 0.5 * dgrid(j - 1));
				generator.row(i) /= generator.row(i).sum();
			}
		}

		void
			generator_drift(const int& rows, const double& beta, const Eigen::ArrayXd& grid,
				Eigen::MatrixXd& generator)
		{
			int loc = 0;
			double weight = 0.0;
			double xa = 0.0;
#pragma omp parallel for private(loc, weight, xa)
			for (int i = 0; i < rows; ++i)
			{
				if (grid(i) != 0.0)
				{
					xa = (1.0 - beta) * grid(i);
					loc = 0;
					locate_lower_bound(grid, xa, rows - 2, loc);
					if (grid(i) < 0.0)
					{
						weight = (grid(loc + 1) - xa) / (grid(loc + 1) - grid(i));

						generator(i, i) = weight;
						generator(i, loc + 1) = 1.0 - weight;
					}
					else
					{
						weight = (grid(i) - xa) / (grid(i) - grid(loc));

						generator(i, loc) = weight;
						generator(i, i) = 1.0 - weight;
					}
				}
				else
				{
					generator(i, i) = 1.0;
				}
			}
		}

		void
			discretization(const Inputs& params, Workspace& ws)
		{
			symmetric_grid(params.nz1, -params.width1 / 2, params.width1 / 2, params.curv1, ws.z1_grid);
			symmetric_grid(params.nz2, -params.width2 / 2, params.width2 / 2, params.curv2, ws.z2_grid);
			ws.dz1_grid = ws.z1_grid.segment(1, params.nz1 - 1) - ws.z1_grid.segment(0, params.nz1 - 1);
			ws.dz2_grid = ws.z2_grid.segment(1, params.nz2 - 1) - ws.z2_grid.segment(0, params.nz2 - 1);
			generator_jump(params.nz1,  0.0, params.sigma1, ws.z1_grid, ws.dz1_grid, ws.generator1_jump);
			generator_jump(params.nz2, 0.0, params.sigma2, ws.z2_grid, ws.dz2_grid, ws.generator2_jump);
			ws.generator1_drift.setZero();
			ws.generator2_drift.setZero();
			generator_drift(params.nz1, params.beta1, ws.z1_grid, ws.generator1_drift);
			generator_drift(params.nz2, params.beta2, ws.z2_grid, ws.generator2_drift);

			ws.generator1 = (ws.generator1_jump - params.id1) * params.lambda1 + ws.generator1_drift - params.id1;
			ws.generator2 = (ws.generator2_jump - params.id2) * params.lambda2 + ws.generator2_drift - params.id2;

			ws.markov1_dt = params.id1 + params.dt * ws.generator1;
			ws.markov2_dt = params.id2 + params.dt * ws.generator2;

		}

		void
			stationary_dist(const Inputs& params, Workspace& ws)
		{
			ws.dist1.setZero();
			ws.dist1(params.nz1 / 2) = 1.0;
			ws.dist2.setZero();
			ws.dist2(params.nz2 / 2) = 1.0;
			find_stationary_dist((params.id1 - params.dtkfe * ws.generator1.transpose()).inverse(), params.tols_dist, params.maxiter_dist, ws.dist1, ws.dist1_next);
			find_stationary_dist((params.id2 - params.dtkfe * ws.generator2.transpose()).inverse(), params.tols_dist, params.maxiter_dist, ws.dist2, ws.dist2_next);

			//find_stationary_dist((params.id1 - 1e-5 * ws.generator1).inverse().transpose(), params.tols_dist, params.maxiter_dist, ws.dist1, ws.dist1_next);
			//find_stationary_dist((params.id2 - 1e-5 * ws.generator2).inverse().transpose(), params.tols_dist, params.maxiter_dist, ws.dist2, ws.dist2_next);
			//find_stationary_dist(ws.markov1_dt.transpose(), params.tols_dist, params.maxiter_dist, ws.dist1, ws.dist1_next);
			//find_stationary_dist(ws.markov2_dt.transpose(), params.tols_dist, params.maxiter_dist, ws.dist2, ws.dist2_next);
		}


		void
			simulate(const Inputs& params, Workspace& ws)
		{
			int t = 0;
			ws.z1_sim_index.setZero();
			ws.z2_sim_index.setZero();
			partial_sum(ws.dist1.cbegin(), ws.dist1.cend(), ws.cdf1.begin(), plus<double>());
			partial_sum(ws.dist2.cbegin(), ws.dist2.cend(), ws.cdf2.begin(), plus<double>());
#pragma omp parallel for
			for (t = 0; t < params.nz1; ++t)
			{
				partial_sum(ws.markov1_dt.row(t).cbegin(), ws.markov1_dt.row(t).cend(),
					ws.markov1_dt_cdf.row(t).begin(), plus<double>());
			}
#pragma omp parallel for
			for (t = 0; t < params.nz2; ++t)
			{
				partial_sum(ws.markov2_dt.row(t).cbegin(), ws.markov2_dt.row(t).cend(),
					ws.markov2_dt_cdf.row(t).begin(), plus<double>());
			}

#pragma omp parallel for private(t)
			for (int n = 0; n < params.nsim; ++n)
			{
				locate_dist(ws.cdf1, params.z1_rand(0, n), params.nz1 - 1, ws.z1_sim_index(0, n));
				locate_dist(ws.cdf2, params.z2_rand(0, n), params.nz2 - 1, ws.z2_sim_index(0, n));
				ws.z_sim(0, n) = ws.z1_grid(ws.z1_sim_index(0, n)) + ws.z2_grid(ws.z2_sim_index(0, n));
				ws.z_lev_sim(0, n) = std::exp(ws.z_sim(0, n));
				for (t = 1; t < params.Tsim; ++t)
				{
					locate_dist(ws.markov1_dt_cdf.row(ws.z1_sim_index(t - 1, n)), params.z1_rand(t, n), params.nz1 - 1, ws.z1_sim_index(t, n));
					locate_dist(ws.markov2_dt_cdf.row(ws.z2_sim_index(t - 1, n)), params.z2_rand(t, n), params.nz2 - 1, ws.z2_sim_index(t, n));
					ws.z_sim(t, n) = ws.z1_grid(ws.z1_sim_index(t, n)) + ws.z2_grid(ws.z2_sim_index(t, n));
					ws.z_lev_sim(t, n) = std::exp(ws.z_sim(t, n));
				}

				for (t = 0; t < params.years; ++t)
				{
					ws.z_ann_lev_sim(t, n) = ws.z_lev_sim.col(n).segment(t * params.dann, params.dann).sum();
					ws.z_ann_sim(t, n) = std::log(ws.z_ann_lev_sim(t, n));
				}
			}
		}

		void
			combine_process(Workspace& ws)
		{
			int i = 0;
			int j = 0;
			int minloc = 0;
			int n = ws.inputs.nz1 * ws.inputs.nz2;
			int n1 = ws.inputs.nz1;
			int n2 = ws.inputs.nz2;
			Eigen::ArrayXi loc(n);
			ws.markov1_quarter = ws.generator1.exp();
			ws.markov2_quarter = ws.generator2.exp();

			Eigen::ArrayXd z_grid(n);
			Eigen::ArrayXd z_max(n);
			Eigen::ArrayXd z_select(n);
			Eigen::VectorXd dist = Eigen::kroneckerProduct(ws.dist1, ws.dist2);
			Eigen::MatrixXd markov_quarter = Eigen::kroneckerProduct(ws.markov1_quarter, ws.markov2_quarter);
			Eigen::MatrixXd generator = Eigen::kroneckerProduct(ws.generator1, ws.inputs.id2)
				+ Eigen::kroneckerProduct(ws.inputs.id1, ws.generator2);
			z_grid = Eigen::kroneckerProduct(ws.z1_grid, Eigen::ArrayXd::Ones(n2))
				+ Eigen::kroneckerProduct(Eigen::ArrayXd::Ones(n1), ws.z2_grid);

			z_max.setConstant(z_grid.maxCoeff() + 1.0);
			z_grid.minCoeff(&minloc);
			loc(0) = minloc;

			for (i = 1; i < n; ++i)
			{
				z_select = (z_grid > z_grid(loc(i - 1))).select(z_grid, z_max);
				z_select.minCoeff(&minloc);
				loc(i) = minloc;
			}
#pragma omp parallel for private(j)
			for (i = 0; i < n; ++i)
			{
				ws.z_grid(i) = z_grid(loc(i));
				ws.dist(i) = dist(loc(i));
				for (j = 0; j < n; ++j)
				{
					ws.generator(i, j) = generator(loc(i), loc(j));
					ws.markov_quarter(i, j) = markov_quarter(loc(i), loc(j));
				}
			}
		}

		void
			write_grids(const Calibration& cal, const Workspace& ws, std::ofstream& of)
		{

			write_matrix_to_file(ws.z_grid, of, "skill_grid.csv");
			write_matrix_to_file(ws.dist, of, "skill_dist.csv");
			write_matrix_to_file(ws.dist1, of, "dist1.csv");
			write_matrix_to_file(ws.dist2, of, "dist2.csv");
			write_matrix_to_file(ws.markov_quarter, of, "markov.csv");
			write_matrix_to_file(ws.markov1_quarter, of, "markov1.csv");
			write_matrix_to_file(ws.markov2_quarter, of, "markov2.csv");
			write_matrix_to_file(ws.z1_grid, of, "z1_grid.csv");
			write_matrix_to_file(ws.z2_grid, of, "z2_grid.csv");
			write_matrix_to_file(ws.generator, of, "generator.csv");
			write_matrix_to_file(ws.generator1, of, "generator1.csv");
			write_matrix_to_file(ws.generator2, of, "generator2.csv");

		}

		void
			estimation(const Calibration& cal, Earning_results& results, std::ofstream& of)
		{
			Inputs earning_inputs = { cal };
			earning_inputs.init();
			Workspace ws = { earning_inputs };
			ws.init();
			Earning_moments moments;
			Discrete_calibration discrete_cal = { cal, earning_inputs, ws, moments };
			discrete_cal.init();
			iteration(discrete_cal);
			combine_process(ws);
			write_grids(cal, ws, of);
			construct_result_map(discrete_cal.init_map, results, moments);

			

		}

		namespace DFNLS
		{
			void 
				assign_parameters(const double* y, Discrete_calibration* params)
			{
				++params->iter;
				auto p = params->init_map.begin();
				if (params->discrete_opt_params.width1)
				{
					p = params->init_map.find("width1");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.width1 = p->second.xcur;
				}

				if (params->discrete_opt_params.width2)
				{
					p = params->init_map.find("width2");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.width2 = p->second.xcur;
				}

				if (params->discrete_opt_params.curv1)
				{
					p = params->init_map.find("curv1");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.curv1 = p->second.xcur;
				}

				if (params->discrete_opt_params.curv2)
				{
					p = params->init_map.find("curv2");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.curv2= p->second.xcur;
				}
			}

			void
				target_estimation_moments(const long n, const long mv, const double* y,
					double* v_err, void* data)
			{
				Discrete_calibration* params = (Discrete_calibration*)(data);

				assign_parameters(y, params);
				discretization(params->earning_inputs, params->ws);
				stationary_dist(params->earning_inputs, params->ws);
				simulate(params->earning_inputs, params->ws);
				compute_earning_moments(params->ws.z_ann_sim, params->earning_inputs.nsim, params->earning_moments);

				dfls_objective(params->target_map, params->targets, params->earning_moments, v_err);

				print_iteration(std::cout, params->init_map, params->target_map, params->iter)
					<< std::setw(16) << std::left << norm(v_err, params->target_map.size()) << endl;
			}
		}
	}



	namespace ESTIMATION
	{
		void
			Inputs::init()
		{
			Tburn = Qburn / dt;
			Tsim = Tburn + Quaters / dt;
			Tann = Quaters / 4;
			dann = 4 / dt;
			Eigen::Rand::P8_mt19937_64 urng1{ 10142 };
			Eigen::Rand::P8_mt19937_64 urng2{ 11445 };
			Eigen::Rand::NormalGen<double> norm_gen;
			Eigen::Rand::UniformRealGen<double> uniform_gen;
			z1_rand.resize(Tsim, nsim);
			z2_rand.resizeLike(z1_rand);
			z1_jump.resizeLike(z1_rand);
			z2_jump.resizeLike(z1_rand);

			z1_rand = norm_gen.generateLike(z1_rand, urng1);
			z2_rand = norm_gen.generateLike(z2_rand, urng2);

			z1_jump = uniform_gen.generateLike(z1_jump, urng1);
			z2_jump = uniform_gen.generateLike(z2_jump, urng2);
		}

		void
			Workspace::stationary_params()
		{
			mean1 = mean2 = 0.0;
			var1 = std::pow(params.sigma1, 2) * params.lambda1 / (2.0 * params.beta1 + params.lambda1);
			var2 = std::pow(params.sigma2, 2) * params.lambda2 / (2.0 * params.beta2 + params.lambda2);
			stay_prob1 = 1.0 - params.lambda1 * params.dt;
			stay_prob2 = 1.0 - params.lambda2 * params.dt;
			drift1 = -params.beta1 * params.dt;
			drift2 = -params.beta2 * params.dt;
		}

		void
			Workspace::init()
		{
			z1.resizeLike(params.z1_rand);
			z2.resizeLike(params.z1_rand);
			z.resizeLike(params.z1_rand);
			z_lev.resizeLike(params.z1_rand);

			z_ann.resize(params.Tann, params.nsim);
			z_ann_lev.resizeLike(z_ann);
		}

		void
			simulate_one_step(const double& jump_rand, const double& z_rand, const double& sigma,
				const double& drift, const double& stay_prob, double& pre, double& next)
		{
			if (jump_rand <= stay_prob)
			{
				next = (1.0 + drift) * pre;
			}
			else
			{
				next = z_rand * sigma;
			}
		}

		void
			simulate(const Inputs& params, Workspace& ws)
		{
			ws.stationary_params();
			ws.z1.row(0) = std::sqrt(ws.var1) * params.z1_rand.row(0);
			ws.z2.row(0) = std::sqrt(ws.var2) * params.z2_rand.row(0);
			ws.z.row(0) = ws.z1.row(0) + ws.z2.row(0);
			ws.z_lev.row(0) = ws.z.row(0).exp();
			int t = 0;
#pragma omp parallel for private(t)
			for (int n = 0; n < params.nsim; ++n)
			{
				for (t = 1; t < params.Tsim; ++t)
				{
					simulate_one_step(params.z1_jump(t, n), params.z1_rand(t, n), params.sigma1, ws.drift1, ws.stay_prob1,
						ws.z1(t - 1, n), ws.z1(t, n));
					simulate_one_step(params.z2_jump(t, n), params.z2_rand(t, n), params.sigma2, ws.drift2, ws.stay_prob2,
						ws.z2(t - 1, n), ws.z2(t, n));

					ws.z(t, n) = ws.z1(t, n) + ws.z2(t, n);
					ws.z_lev(t, n) = std::exp(ws.z(t, n));
				}
				for (t = 0; t < params.Tann; ++t)
				{
					ws.z_ann_lev(t, n) = ws.z_lev.col(n).segment(params.Tburn + t * params.dann, params.dann).sum();
					ws.z_ann(t, n) = std::log(ws.z_ann_lev(t, n));
				}
			}
		}



		void
			Estimate_calibration::init()
		{
			int wedth = 16;
			display_init_params(cal.earning_parameter_map(), wedth,
				options.estimate_opt_params_set(), init_map);

			display_dist_targets(cal.earning_parameter_map(), wedth, options.dist_targets_set(), target_map);

			name = "EARNING_ESTIMATION";
			tols_roots = cal.earning_params("tols_roots");
			maxiter_roots = cal.earning_size("maxiter_roots");
			step_size = cal.earning_params("step_size");
			maxiter_mins = cal.earning_size("maxiter_mins");
			tols_mins = cal.earning_params("tols_mins");

			maxfun_multip = 200;
			rhobeg = cal.earning_params("rhobeg");
			rhoend = cal.earning_params("rhoend");
			ndfls = cal.earning_size("ndfls");
			num_thread = cal.size("num_thread");


			dfls_target = DFNLS::target_estimation_moments;
		}

		void estimation(const Calibration& cal, Earning_results& results)
		{
			Inputs earning_inputs = { cal };
			earning_inputs.init();
			Workspace ws = { earning_inputs };
			ws.init();
			Earning_moments moments;
			Estimate_calibration estimate_cal = { cal, earning_inputs, ws, moments };
			estimate_cal.init();
			iteration(estimate_cal);
			construct_result_map(estimate_cal.init_map, results, moments);

		}

		namespace DFNLS
		{
			void assign_parameters(const double* y, Estimate_calibration* params)
			{
				++params->iter;
				auto p = params->init_map.begin();
				if (params->estimate_opt_params.beta1)
				{
					p = params->init_map.find("beta1");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.beta1 = p->second.xcur;
				}

				if (params->estimate_opt_params.beta2)
				{
					p = params->init_map.find("beta2");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.beta2 = p->second.xcur;
				}

				if (params->estimate_opt_params.lambda1)
				{
					p = params->init_map.find("lambda1");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.lambda1 = p->second.xcur;
				}

				if (params->estimate_opt_params.lambda2)
				{
					p = params->init_map.find("lambda2");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.lambda2 = p->second.xcur;
				}

				if (params->estimate_opt_params.sigma1)
				{
					p = params->init_map.find("sigma1");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.sigma1 = p->second.xcur;
				}

				if (params->estimate_opt_params.sigma2)
				{
					p = params->init_map.find("sigma2");
					p->second.xcur = logistic(y[distance(params->init_map.begin(), p)],
						p->second.high, p->second.low, p->second.init);
					params->earning_inputs.sigma2 = p->second.xcur;
				}
			}
			void
				target_estimation_moments(const long n, const long mv, const double* y,
					double* v_err, void* data)
			{
				Estimate_calibration* params = (Estimate_calibration*)(data);

				assign_parameters(y, params);
				simulate(params->earning_inputs, params->ws);
				compute_earning_moments(params->ws.z_ann, params->earning_inputs.nsim, params->earning_moments); 

				dfls_objective(params->target_map, params->targets, params->earning_moments, v_err);

				print_iteration(std::cout, params->init_map, params->target_map, params->iter)
					<< std::setw(16) << std::left << norm(v_err, params->target_map.size()) << endl;
			}
		}
	}


}
