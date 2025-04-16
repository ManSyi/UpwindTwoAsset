
#include <map>
#include <string>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <filesystem>
#include <vector>
#include <omp.h>
#include <numeric>
#include <algorithm>

#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>
#include<Eigen/SparseLU>
#include <unsupported/Eigen/KroneckerProduct> 

#include "Steady_state.h"
#include "Two_asset.h"
#include "Het_block.h"
#include "Support.h"



using namespace std;
typedef Eigen::SparseMatrix<double> SpMat;

namespace TA
{
	namespace Adj_fun_auclert
	{
		double
			adj(const Het_Inputs& het_inputs, const double& d_abs, int k, int m)
		{
			return het_inputs.chi1 / het_inputs.chi2
				* (d_abs / std::pow(het_inputs.agrid(k, m) + het_inputs.a_kink, het_inputs.chi2))
				* (het_inputs.agrid(k, m) + het_inputs.a_kink);
		}

		double
			adj1(const Het_Inputs& het_inputs, const double& d, int k, int m)
		{
			return het_inputs.chi1
				* std::pow(std::abs(d) / het_inputs.agrid(k, m) + het_inputs.a_kink, het_inputs.chi2 - 1)
				* sign(d);
		}
	}

	namespace Adj_fun_kaplan
	{
		double
			adj(const Het_Inputs& het_inputs, const double& d_abs, int k, int m)
		{
			return het_inputs.chi0 * d_abs * het_inputs.innaction
				+ het_inputs.chi1 * std::pow(d_abs / het_inputs.akinkgrid(k, m), het_inputs.chi2)
				* het_inputs.akinkgrid(k, m);
		}

		double
			adj1(const Het_Inputs& het_inputs, const double& d, int k, int m)
		{
			return (het_inputs.chi1 * het_inputs.chi2
				* std::pow(std::abs(d) / het_inputs.akinkgrid(k, m), het_inputs.chi2 - 1)
				+ het_inputs.chi0 * het_inputs.innaction) * sign(d);
		}
	}


	void
		swap_het_vecs(Het_workspace& ws, std::unordered_map<std::string, Eigen::MatrixXd>& map)
	{
		std::swap(map["V"], ws.V);
		std::swap(map["ccum1"], ws.ccum1);
		std::swap(map["ccum2"], ws.ccum2);
		std::swap(map["ccum4"], ws.ccum4);
		std::swap(map["subeff1ass"], ws.subeff1ass);
		std::swap(map["wealtheff1ass"], ws.wealtheff1ass);
		std::swap(map["subeff2ass"], ws.subeff2ass);
		std::swap(map["wealtheff2ass"], ws.wealtheff2ass);
	}

	void
		swap_het_dist(Het_workspace& ws, SteadyState& ss)
	{
		swap(ws.dist, ss.get_joint_dist("HA"));
		swap(ws.adist, ss.get_marg_dist("HA_A"));
		swap(ws.bdist, ss.get_marg_dist("HA_B"));
		swap(ws.acdf, ss.get_marg_dist("HA_A_CDF"));
		swap(ws.bcdf, ss.get_marg_dist("HA_B_CDF"));
		swap(ws.abdist, ss.get_joint_dist("HA_AB"));
		swap(ws.assetHJB, ss.get_trans_mat("HA"));
	}


	void
		convert_pols_map(const Het_Inputs& het_inputs, Het_workspace& ws, std::unordered_map<std::string, Tensor>& pols_map)
	{
		if (!het_inputs.isTransition)
		{
#pragma omp parallel for
			for (int r = 0; r < het_inputs.nesk; ++r)
			{
				pols_map["a"][r] = het_inputs.agrid;
				pols_map["b"][r] = het_inputs.bgrid;
			}
		}
		swap(pols_map["hour"], ws.hour);
		swap(pols_map["labor"], ws.labor);
		swap(pols_map["sb"], ws.sb);
		swap(pols_map["c"], ws.c);
		swap(pols_map["d"], ws.d);
		swap(pols_map["adj"], ws.adj);
		swap(pols_map["sa"], ws.sa);
		swap(pols_map["sd"], ws.sd);
	}

	// Common moments for two asset model. 
	void
		solve_targeting_moments(const Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp)
	{
		htp.Eb = expectation(het_inputs.bgrid, ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);
		htp.Ea = expectation(het_inputs.agrid, ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);
		switch (het_inputs.hour_supply)
		{
		case Hour_supply::Seq:
			htp.Elabor = expectation(ws.hour, ws.dist.reshaped(), het_inputs.nsk, het_inputs.nab, het_inputs.sgrid);
			htp.Ehour = expectation(ws.hour, ws.dist.reshaped(), het_inputs.nsk, het_inputs.nab);
			break;
		case Hour_supply::GHH:
			htp.Elabor = 0.0;
			htp.Ehour = 0.0;
			for (int r = 0; r < het_inputs.nsk; ++r)
			{
				htp.Elabor += het_inputs.hour_ghh(r) * het_inputs.sgrid(r) * ws.dist.col(r).sum();
				htp.Ehour += het_inputs.hour_ghh(r) * ws.dist.col(r).sum();
			}
			break;
		case Hour_supply::NoSupply:
			htp.Ehour = 1.0;
			htp.Elabor = 0.0;
			for (int r = 0; r < het_inputs.nsk; ++r)
			{
				htp.Elabor += het_inputs.sgrid(r) * ws.dist.col(r).sum();
			}
			break;
		default:
			break;
		}
		htp.Egrosslabinc = htp.Elabor * het_inputs.wage;
		ws.abdist = ws.dist.rowwise().sum().reshaped(het_inputs.na, het_inputs.nb);
		ws.adist  = ws.abdist.rowwise().sum();
		ws.bdist  = ws.abdist.colwise().sum().transpose();
		htp.b0_close_model = htp.b0_close_frac * htp.Egrosslabinc;
		htp.a0_close_model = htp.a0_close_frac * htp.Egrosslabinc;
		htp.FracLiq0Ill0 = probability(htp.a_dist_grid, htp.b_dist_grid, ws.abdist, htp.a0_close_model, htp.b0_close_model);
		htp.FracIll0 = probability(htp.a_dist_grid, ws.adist, htp.a0_close_model);
		htp.FracLiq0 = probability(htp.b_dist_grid, ws.bdist, htp.b0_close_model);
		if (htp.targets.targetFracLiqNeg)
		{
			htp.FracLiq0Ill0 -= probability(htp.a_dist_grid, htp.b_dist_grid, ws.abdist, htp.a0_close_model, htp.b_dist_grid(het_inputs.nb_neg - 1));
			htp.FracLiqNeg = ws.bdist.array().segment(0, het_inputs.nb_neg).sum();
			htp.FracLiq0 -= htp.FracLiqNeg;
		}
		htp.FracLiq0Illpos = htp.FracLiq0 - htp.FracLiq0Ill0;
	}

	// Back out those policies outside the het block loop
	void
		solve_res_pols(const Het_Inputs& het_inputs, Het_workspace& ws)
	{
		int r = 0;
		switch (het_inputs.adj_fun)
		{
		case Adj_fun::KAPLAN_ADJ:
#pragma omp parallel for
			for (r = 0; r < het_inputs.nesk; ++r)
			{
				ws.adj[r] = Adj_fun_kaplan::adj(het_inputs, ws.d[r].abs());
			}
			break;
		case Adj_fun::AUCLERT_ADJ:
#pragma omp parallel for
			for (r = 0; r < het_inputs.nesk; ++r)
			{
				ws.adj[r] = Adj_fun_auclert::adj(het_inputs, ws.d[r].abs());
			}
			break;
		default:
			break;
		}
		ws.sd[r] = -ws.d[r] - ws.adj[r];
		switch (het_inputs.hour_supply)
		{
		case Hour_supply::Seq:
#pragma omp parallel for
			for (r = 0; r < het_inputs.nsk; ++r)
			{
				ws.labor[r] = ws.hour[r] * het_inputs.sgrid(r);
			}
			break;
		case Hour_supply::GHH:
#pragma omp parallel for
			for (r = 0; r < het_inputs.nsk; ++r)
			{
				ws.hour[r].setConstant(het_inputs.hour_ghh(r));
				ws.labor[r].setConstant(het_inputs.hour_ghh(r) * het_inputs.sgrid(r));
			}
			break;
		case Hour_supply::NoSupply:
#pragma omp parallel for
			for (r = 0; r < het_inputs.nsk; ++r)
			{
				ws.hour[r].setConstant(1.0);
				ws.labor[r].setConstant(het_inputs.sgrid(r));
			}
		default:
			break;
		}
	}



	void
		solve_labor(const Het_Inputs& het_inputs, Het_workspace& ws)
	{
		int r = 0;
		switch (het_inputs.hour_supply)
		{
		case Hour_supply::Seq:
#pragma omp parallel for
			for (r = 0; r < het_inputs.nsk; ++r)
			{
				ws.labor[r] = ws.hour[r] * het_inputs.sgrid(r);
			}
			break;
		case Hour_supply::GHH:
#pragma omp parallel for
			for (r = 0; r < het_inputs.nsk; ++r)
			{
				ws.labor[r].setConstant(het_inputs.hour_ghh(r) * het_inputs.sgrid(r));
			}
			break;
		case Hour_supply::NoSupply:
#pragma omp parallel for
			for (r = 0; r < het_inputs.nsk; ++r)
			{
				ws.labor[r].setConstant(het_inputs.sgrid(r));
			}
			break;
		default:
			break;
		}
	}




	void
		solve_cdf(const Het_Inputs& het_inputs, Het_workspace& ws)
	{
		partial_sum(ws.adist.cbegin(), ws.adist.cend(), ws.acdf.begin(), plus<double>());
		partial_sum(ws.bdist.cbegin(), ws.bdist.cend(), ws.bcdf.begin(), plus<double>());
	}

	void
		solve_dist_stat(const Het_Inputs& het_inputs, Het_Outputs& htp, Het_workspace& ws, std::unordered_map<std::string, aggregates>& map)
	{
		map["Ea"] = htp.Ea;
		map["Eb"] = htp.Eb;
		map["Ehour"] = htp.Ehour;
		map["Elabor"] = htp.Elabor;
		map["Esa"] = expectation(ws.sa, ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);
		map["Esb"] = expectation(ws.sb, ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);
		map["Ec"] = expectation(ws.c, ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);
		
		map["Eadj"] = expectation(ws.adj, ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);
		map["Ed"] = expectation(ws.d, ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);

		map["Eb_neg"] = expectation(het_inputs.bgrid.min(0), ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);;
		map["FracLiq0Ill0"] = htp.FracLiq0Ill0;
		map["FracIll0"] = htp.FracIll0;
		map["FracLiq0"] = htp.FracLiq0;
		map["FracLiq0Illpos"] = htp.FracLiq0Illpos;
		map["FracLiqNeg"] = htp.FracLiqNeg;

#pragma omp parallel sections
		{
#pragma omp section
			{
				htp.A_GINI = (ws.acdf.array() * (1 - ws.acdf.array()) * het_inputs.tda).sum() / htp.Ea;
				htp.B_GINI = (ws.bcdf.array() * (1 - ws.bcdf.array()) * het_inputs.tdb).sum() / htp.Eb;

				ws.illiquid_wealth_dist = htp.a_dist_grid * ws.adist.array();
				htp.A_TOP_01 = 1.0 - quantile_share(ws.acdf, ws.illiquid_wealth_dist, htp.Ea, 0.999);
				htp.A_TOP_1 = 1.0 - quantile_share(ws.acdf, ws.illiquid_wealth_dist, htp.Ea, 0.99);
				htp.A_TOP_10 = 1.0 - quantile_share(ws.acdf, ws.illiquid_wealth_dist, htp.Ea, 0.9);
				htp.A_BOTTOM_50 = quantile_share(ws.acdf, ws.illiquid_wealth_dist, htp.Ea, 0.5);
				htp.A_BOTTOM_25 = quantile_share(ws.acdf, ws.illiquid_wealth_dist, htp.Ea, 0.25);
			}

#pragma omp section
			{
				ws.liquid_wealth_dist = htp.b_dist_grid * ws.bdist.array();
				htp.B_TOP_01 = 1.0 - quantile_share(ws.bcdf, ws.liquid_wealth_dist, htp.Eb, 0.999);
				htp.B_TOP_1 = 1.0 - quantile_share(ws.bcdf, ws.liquid_wealth_dist, htp.Eb, 0.99);
				htp.B_TOP_10 = 1.0 - quantile_share(ws.bcdf, ws.liquid_wealth_dist, htp.Eb, 0.9);
				htp.B_BOTTOM_50 = quantile_share(ws.bcdf, ws.liquid_wealth_dist, htp.Eb, 0.5);
				htp.B_BOTTOM_25 = quantile_share(ws.bcdf, ws.liquid_wealth_dist, htp.Eb, 0.25);
			}

#pragma omp section
			{
				htp.FracLiq0Ill0_close = probability(htp.b_dist_grid, ws.abdist.row(0), htp.b0_ub)
					- probability(htp.b_dist_grid, ws.abdist.row(0), htp.b0_lb);
				htp.FracLiq0_close = probability(htp.b_dist_grid, ws.bdist, htp.b0_ub)
					- probability(htp.b_dist_grid, ws.bdist, htp.b0_lb);
				htp.FracLiq0Illpos_close = htp.FracLiq0_close - htp.FracLiq0Ill0_close;
				htp.FracLiqNeg_close = probability(htp.b_dist_grid, ws.bdist, htp.b0_lb);
			}
		}

		map["A_GINI"] = htp.A_GINI;
		map["B_GINI"] = htp.B_GINI;

		map["A_TOP_0.1%"] = htp.A_TOP_01;
		map["A_TOP_1%"] = htp.A_TOP_1;
		map["A_TOP_10%"] = htp.A_TOP_10;
		map["A_BOTTOM_50%"] = htp.A_BOTTOM_50;
		map["A_BOTTOM_25%"] = htp.A_BOTTOM_25;

		map["B_TOP_0.1%"] = htp.B_TOP_01;
		map["B_TOP_1%"] = htp.B_TOP_1;
		map["B_TOP_10%"] = htp.B_TOP_10;
		map["B_BOTTOM_50%"] = htp.B_BOTTOM_50;
		map["B_BOTTOM_25%"] = htp.B_BOTTOM_25;


		map["b0_lb"] = htp.b0_lb;
		map["b0_ub"] = htp.b0_ub;
		map["FracLiqNeg_close"] = htp.FracLiqNeg_close;
		map["FracLiq0Ill0_close"] = htp.FracLiq0Ill0_close;
		map["FracLiq0_close"] = htp.FracLiq0_close;
		map["FracLiq0Illpos_close"] = htp.FracLiq0Illpos_close;
	}

	void
		target_HtM(const Calibration& cal, Het_Outputs& htp, const Het_Inputs& het_inputs, Het_workspace& ws)
	{
		double targetFracLiqNeg = cal.params("targetFracLiqNeg");
		double targetFracLiq0 = cal.params("targetFracLiq0");
		double targetFracLiq0Ill0 = cal.params("targetFracLiq0Ill0");
		double targetFracLiq0Illpos = cal.params("targetFracLiq0Illpos");


		int i = 0;
		double weight = 0.0;
		locate_lower_bound(ws.bcdf, targetFracLiqNeg, het_inputs.nb - 2, i, weight);
		htp.b0_lb = weight * htp.b_dist_grid(i) + (1.0 - weight) * htp.b_dist_grid(i + 1);
		if (htp.b0_lb > -1e-8) htp.b0_lb = htp.b_dist_grid(het_inputs.nb_neg - 1);
		Eigen::VectorXd a0bcdf(het_inputs.nb);
		partial_sum(ws.abdist.row(0).cbegin(), ws.abdist.row(0).cend(), a0bcdf.begin(), plus<double>());
		htp.FracLiqNeg_close = probability(htp.b_dist_grid, ws.bdist, htp.b0_lb);

		double fraclba0 = probability(htp.b_dist_grid, ws.bdist, htp.b0_lb);
		double fracuba0 = targetFracLiq0 + htp.FracLiqNeg_close;
		i = 0;
		locate_lower_bound(ws.bcdf, fracuba0, het_inputs.nb - 2, i, weight);
		htp.b0_ub = weight * htp.b_dist_grid(i) + (1.0 - weight) * htp.b_dist_grid(i + 1);

		

	}
	void
		solve_het_one_step(const Het_Inputs& het_inputs, Het_workspace& ws)
	{
		// need initial V and initial dist
		solve_HJB(het_inputs, ws);
		std::swap(ws.V, ws.next);
		construct_final_assetHJB(het_inputs, ws, ws.sa, ws.sb);
		construct_distKFE(het_inputs, ws, ws.assetHJB);
		solve_KFE(het_inputs, ws);
		std::swap(ws.dist, ws.next);
	}


	void
		construct_transpose_distKFE(const Het_Inputs& het_inputs, Het_workspace& ws, const std::vector<SpMat>& assetHJB)
	{
#pragma omp parallel for
		for (int r = 0; r < het_inputs.nesk; ++r)
		{
			ws.distKFE[r] = het_inputs.diagKFE[r]
				- het_inputs.DeltaKFE
				* assetHJB[r];
			ws.solvers[r]->compute(ws.distKFE[r]);
			if (ws.solvers[r]->info() != Eigen::Success) {
				std::cerr << "KFE decomposition failed" << std::endl;
				print_parameter(het_inputs, ws);
				exit(0);
			}
		}
	}


	void
		solve_convergent_pols(const Het_Outputs& target_params, const Het_Inputs& het_inputs, Het_workspace& ws)
	{
		ws.init_bellman();
		std::swap(ws.V, ws.V_init);
		ws.eps = 1;

		for (ws.iter = 0; ws.iter != target_params.maxiter_policy; ++ws.iter)
		{
			solve_HJB(het_inputs, ws);
			if (ws.iter % 10 == 0)
			{
				if (check_within_tols(ws.V, ws.next, target_params.tols_policy, ws.eps))
				{
					goto end;
				}
			}
			std::swap(ws.V, ws.next);
		}
		std::cout << "Couldn't find stationary policies after " << target_params.maxiter_policy << " iterations!!" << std::endl;
		print_parameter(het_inputs, ws);
		exit(0);
	end:
		construct_final_assetHJB(het_inputs, ws, ws.sa, ws.sb);
		construct_distKFE(het_inputs, ws, ws.assetHJB);
		solve_convergent_dist(target_params, het_inputs, ws);
	}

	void
		print_parameter(const Het_Inputs& het_inputs, const Het_workspace& ws)
	{
		std::cout << "\nCurrent parameter:"
			<< " rho: " << het_inputs.rho
			<< " chi0: " << het_inputs.chi0
			<< " chi1: " << het_inputs.chi1
			<< " chi2: " << het_inputs.chi2
			<< " rb_wedge: " << het_inputs.rb_wedge
			<< " vphi: " << het_inputs.vphi
			<<"\neps = " << ws.eps
			<< std::endl;
	}

	void
		solve_HJB(const Het_Inputs& het_inputs, Het_workspace& ws)
	{
#pragma omp parallel for
		for (int r = 0; r < het_inputs.nesk; ++r)
		{
			derivative(het_inputs, ws.V, ws.VaF[r], ws.VaB[r], ws.VbF[r], ws.VbB[r], r);
			// b forward
			het_inputs.solve_unbinding_cons(het_inputs, ws, ws.VbF[r], r,
				ws.cF[r], ws.scF[r], ws.hF[r], ws.utilityF[r]);
			// b backward
			het_inputs.solve_unbinding_cons(het_inputs, ws, ws.VbB[r], r,
				ws.cB[r], ws.scB[r], ws.hB[r], ws.utilityB[r]);

			// a binding, b forward
			ws.sb0F[r] = ws.scF[r] + ws.sd0[r];
			ws.H0F[r] = ws.utilityF[r] + ws.VbF[r] * ws.sb0F[r];

			// a binding, b backward
			ws.sb0B[r] = ws.scB[r] + ws.sd0[r];
			ws.H0B[r] = ws.utilityB[r] + ws.VbB[r] * ws.sb0B[r];

			// a forward, b forward
			het_inputs.solve_unbinding_res(het_inputs, ws,
				ws.VbF[r], ws.VaF[r], ws.scF[r], ws.utilityF[r], r,
				ws.dFF[r], ws.saFF[r], ws.sbFF[r], ws.HFF[r]);

			// a backward, b forward
			het_inputs.solve_unbinding_res(het_inputs, ws,
				ws.VbF[r], ws.VaB[r], ws.scF[r], ws.utilityF[r], r,
				ws.dBF[r], ws.saBF[r], ws.sbBF[r], ws.HBF[r]);

			// a forward, b backward
			het_inputs.solve_unbinding_res(het_inputs, ws,
				ws.VbB[r], ws.VaF[r], ws.scB[r], ws.utilityB[r], r,
				ws.dFB[r], ws.saFB[r], ws.sbFB[r], ws.HFB[r]);

			// a backward, b backward
			het_inputs.solve_unbinding_res(het_inputs, ws,
				ws.VbB[r], ws.VaB[r], ws.scB[r], ws.utilityB[r], r,
				ws.dBB[r], ws.saBB[r], ws.sbBB[r], ws.HBB[r]);

			het_inputs.solve_binding_pols(het_inputs, ws, r);

			ws.validFB[r] = ws.sbFB[r] < 0 && ws.saFB[r] > 0;
			ws.validFF[r] = ws.sbFF[r] > 0 && ws.saFF[r] > 0;
			ws.validBB[r] = ws.sbBB[r] < 0 && ws.saBB[r] < 0;
			ws.validBF[r] = ws.sbBF[r] > 0 && ws.saBF[r] < 0;
			ws.valid0F[r] = ws.sb0F[r] > 0;
			ws.valid0B[r] = ws.sb0B[r] < 0;

			ws.HBF[r].col(het_inputs.nb - 1).setConstant(het_inputs.H_min);
			ws.HBF[r].row(0).setConstant(het_inputs.H_min);

			ws.HBB[r].col(0).setConstant(het_inputs.H_min);
			ws.HBB[r].row(0).setConstant(het_inputs.H_min);

			ws.HFF[r].col(het_inputs.nb - 1).setConstant(het_inputs.H_min);
			ws.HFF[r].row(het_inputs.na - 1).setConstant(het_inputs.H_min);

			ws.HFB[r].col(0).setConstant(het_inputs.H_min);
			ws.HFB[r].row(het_inputs.na - 1).setConstant(het_inputs.H_min);

			ws.H0B[r].col(0).setConstant(het_inputs.H_min);
			ws.H0F[r].col(het_inputs.nb - 1).setConstant(het_inputs.H_min);


			ws.indFB[r] = ws.validFB[r] 
				&& (!ws.validFF[r] || ws.HFB[r] >= ws.HFF[r])
				&& (!ws.validBB[r] || ws.HFB[r] >= ws.HBB[r])
				&& (!ws.validBF[r] || ws.HFB[r] >= ws.HBF[r])
				&& (!ws.valid0B[r] || ws.HFB[r] >= ws.H0B[r])
				&& (!ws.valid0F[r] || ws.HFB[r] >= ws.H0F[r])
				&& (ws.HFB[r] >= ws.H0[r]);

			ws.indFF[r] = ws.validFF[r]
				&& (!ws.validFB[r] || ws.HFF[r] > ws.HFB[r])
				&& (!ws.validBB[r] || ws.HFF[r] >= ws.HBB[r])
				&& (!ws.validBF[r] || ws.HFF[r] >= ws.HBF[r])
				&& (!ws.valid0B[r] || ws.HFF[r] >= ws.H0B[r])
				&& (!ws.valid0F[r] || ws.HFF[r] >= ws.H0F[r])
				&& (ws.HFF[r] >= ws.H0[r]);

			ws.indBB[r] = ws.validBB[r]
				&& (!ws.validFF[r] || ws.HBB[r] > ws.HFF[r])
				&& (!ws.validFB[r] || ws.HBB[r] > ws.HFB[r])
				&& (!ws.validBF[r] || ws.HBB[r] >= ws.HBF[r])
				&& (!ws.valid0B[r] || ws.HBB[r] >= ws.H0B[r])
				&& (!ws.valid0F[r] || ws.HBB[r] >= ws.H0F[r])
				&& (ws.HBB[r] >= ws.H0[r]);

			ws.indBF[r] = ws.validBF[r]
				&& (!ws.validFF[r] || ws.HBF[r] > ws.HFF[r])
				&& (!ws.validBB[r] || ws.HBF[r] > ws.HBB[r])
				&& (!ws.validFB[r] || ws.HBF[r] > ws.HFB[r])
				&& (!ws.valid0B[r] || ws.HBF[r] >= ws.H0B[r])
				&& (!ws.valid0F[r] || ws.HBF[r] >= ws.H0F[r])
				&& (ws.HBF[r] >= ws.H0[r]);

			ws.ind0F[r] = ws.valid0F[r]
				&& (!ws.validFF[r] || ws.H0F[r] > ws.HFF[r])
				&& (!ws.validBB[r] || ws.H0F[r] > ws.HBB[r])
				&& (!ws.validBF[r] || ws.H0F[r] > ws.HBF[r])
				&& (!ws.validFB[r] || ws.H0F[r] > ws.HFB[r])
				&& (!ws.valid0B[r] || ws.H0F[r] >= ws.H0B[r])
				&& (ws.H0F[r] >= ws.H0[r]);

			ws.ind0B[r] = ws.valid0B[r]
				&& (!ws.validFF[r] || ws.H0B[r] > ws.HFF[r])
				&& (!ws.validBB[r] || ws.H0B[r] > ws.HBB[r])
				&& (!ws.validBF[r] || ws.H0B[r] > ws.HBF[r])
				&& (!ws.validFB[r] || ws.H0B[r] > ws.HFB[r])
				&& (!ws.valid0F[r] || ws.H0B[r] > ws.H0F[r])
				&& (ws.H0B[r] >= ws.H0[r]);



			ws.ind0[r] = !ws.indFF[r] && !ws.indFB[r] && !ws.indBB[r]
				&& !ws.indBF[r] && !ws.ind0F[r] && !ws.ind0B[r];

			ws.sb[r] = ws.indFF[r].cast<double>() * ws.sbFF[r] + ws.indFB[r].cast<double>() * ws.sbFB[r]
				+ ws.indBB[r].cast<double>() * ws.sbBB[r] + ws.indBF[r].cast<double>() * ws.sbBF[r]
				+ ws.ind0F[r].cast<double>() * ws.sb0F[r] + ws.ind0B[r].cast<double>() * ws.sb0B[r];

			ws.c[r] = (ws.indFB[r].cast<double>() + ws.indBB[r].cast<double>() + ws.ind0B[r].cast<double>()) * ws.cB[r]
				+ (ws.indFF[r].cast<double>() + ws.indBF[r].cast<double>() + ws.ind0F[r].cast<double>()) * ws.cF[r]
				+ ws.ind0[r].cast<double>() * ws.c0[r];

			ws.d[r] = ws.indFF[r].cast<double>() * ws.dFF[r] + ws.indFB[r].cast<double>() * ws.dFB[r]
				+ ws.indBB[r].cast<double>() * ws.dBB[r] + ws.indBF[r].cast<double>() * ws.dBF[r]
				+ (ws.ind0F[r].cast<double>() + ws.ind0B[r].cast<double>()) * (-het_inputs.a_drift)
				+ ws.ind0[r].cast<double>() * ws.d0[r];
			
			ws.sa[r] = ws.d[r] + het_inputs.a_drift;


			het_inputs.solve_rhs(het_inputs, ws, r);

			ws.adriftB[r] = ws.sa[r].min(0);
			ws.adriftF[r] = ws.sa[r].max(0);
			ws.bdriftB[r] = ws.sb[r].min(0);
			ws.bdriftF[r] = ws.sb[r].max(0);

			construct_assetHJB(het_inputs, ws, r);
			
			ws.assetHJB[r] -= het_inputs.diagHJB[r];

			ws.solvers[r]->compute(ws.assetHJB[r]);
			if (ws.solvers[r]->info() != Eigen::Success) {
				std::cerr << "HJB decomposition failed" << std::endl;
				print_parameter(het_inputs, ws);
				exit(0);
			}
			ws.next.col(r) = ws.solvers[r]->solve(ws.rhs.col(r));
		}
	}



	void
		solve_convergent_dist(const Het_Outputs& target_params, const Het_Inputs& het_inputs, Het_workspace& ws)
	{
		ws.dist = ws.dist_init;
		ws.eps = 1;
		for (ws.iter = 0; ws.iter != target_params.maxiter_dist; ++ws.iter)
		{
			solve_KFE(het_inputs, ws);
			if (ws.iter % 10 == 0)
			{
				if (check_within_tols(ws.dist, ws.next, target_params.tols_dist, ws.eps))
				{
					goto end;
				}
			}
			std::swap(ws.dist, ws.next);
		}
		std::cout << "Couldn't find stationary dist after " << target_params.maxiter_dist << " iterations!!" << endl;
		print_parameter(het_inputs, ws);
		exit(0);
	end:
		std::swap(ws.dist, ws.next);
	}


	void
		construct_final_assetHJB(const Het_Inputs& het_inputs, Het_workspace& ws,
			const Tensor& sa, const Tensor& sb)
	{
#pragma omp parallel for
		for (int r = 0; r < het_inputs.nesk; ++r)
		{
			ws.assetHJB[r] += het_inputs.diagHJB[r];
		}
	}

	void
		construct_distKFE(const Het_Inputs& het_inputs, Het_workspace& ws, const std::vector<SpMat>& assetHJB)
	{
#pragma omp parallel for
		for (int r = 0; r < het_inputs.nesk; ++r)
		{
			ws.distKFE[r] = het_inputs.diagKFE[r]
				- het_inputs.DeltaKFE
				* SpMat(assetHJB[r].transpose());
			ws.solvers[r]->compute(ws.distKFE[r]);
			if (ws.solvers[r]->info() != Eigen::Success) {
				std::cerr << "KFE decomposition failed" << std::endl;
				print_parameter(het_inputs, ws);
				exit(0);
			}
		}
	}


	void
		solve_KFE(const Het_Inputs& het_inputs, Het_workspace& ws)
	{
#pragma omp parallel for
		for (int r = 0; r < het_inputs.nesk; ++r)
		{
			ws.rhs.col(r) = ws.dist * het_inputs.offdiagKFE.col(r);
			ws.rhs(het_inputs.na * het_inputs.nb_neg, r) += het_inputs.deathrate * het_inputs.DeltaKFE
				* ws.dist.col(r).sum();
			ws.next.col(r) = ws.solvers[r]->solve(ws.rhs.col(r));
		}
		
	}

	void
		construct_assetHJB(const Het_Inputs& het_inputs, Het_workspace& ws, int r)
	{
		int i = 0;
		int ii = 0;
		int j = 0;
		double v = 0;
		ws.assetHJB[r].setZero();
		ws.assetHJB[r].reserve(Eigen::VectorXi::Constant(het_inputs.nab, 5));
		for (j = 0; j < het_inputs.nb; ++j)
		{
			for (i = 0; i < het_inputs.na; ++i)
			{
				ii = het_inputs.na * j + i; 

				// fill diag element
				v = -ws.adriftF[r](i, j) * het_inputs.dainv_diag(i + 1) + ws.adriftB[r](i, j) * het_inputs.dainv_diag(i)
					-ws.bdriftF[r](i, j) * het_inputs.dbinv_diag(j + 1) + ws.bdriftB[r](i, j) * het_inputs.dbinv_diag(j);
				ws.assetHJB[r].insert(ii, ii) = v;
				
				// fill a lower diag
				if (i > 0)
				{
					v = -ws.adriftB[r](i, j) * het_inputs.dainv(i - 1);
					if (v != 0) ws.assetHJB[r].insert(ii, ii - 1) = v;
				}
				// fill a up diag
				if (i < het_inputs.na - 1)
				{
					v = ws.adriftF[r](i, j) * het_inputs.dainv(i);
					if (v != 0) ws.assetHJB[r].insert(ii, ii + 1) = v;
				}

				// fill b lower diag
				if (j > 0)
				{
					v = -ws.bdriftB[r](i, j) * het_inputs.dbinv(j - 1);
					if (v != 0) ws.assetHJB[r].insert(ii, ii - het_inputs.na) = v;
				}
				// fill b up diag
				if (j < het_inputs.nb - 1)
				{
					v = ws.bdriftF[r](i, j) * het_inputs.dbinv(j);
					if (v != 0) ws.assetHJB[r].insert(ii, ii + het_inputs.na) = v;
				}
			}
		}
		ws.assetHJB[r].makeCompressed();
	}


	void
		derivative(const Het_Inputs& het_inputs, const Eigen::MatrixXd& V, het2& VaF, het2& VaB, het2& VbF, het2& VbB, int r)
	{
		VaF.block(0, 0, het_inputs.na - 1, het_inputs.nb)
			= ((V.col(r).reshaped(het_inputs.na, het_inputs.nb).block(1, 0, het_inputs.na - 1, het_inputs.nb)
				- V.col(r).reshaped(het_inputs.na, het_inputs.nb).block(0, 0, het_inputs.na - 1, het_inputs.nb)).array()
				* het_inputs.dagridinv).max(het_inputs.dVamin);

		VaF.row(het_inputs.na - 1) = VaF.row(het_inputs.na - 2);

		VaB.block(1, 0, het_inputs.na - 1, het_inputs.nb)
			= VaF.block(0, 0, het_inputs.na - 1, het_inputs.nb);

		VaF.row(0) = VaF.row(1);

		VbF.block(0, 0, het_inputs.na, het_inputs.nb - 1)
			= ((V.col(r).reshaped(het_inputs.na, het_inputs.nb).block(0, 1, het_inputs.na, het_inputs.nb - 1)
				- V.col(r).reshaped(het_inputs.na, het_inputs.nb).block(0, 0, het_inputs.na, het_inputs.nb - 1)).array()
				* het_inputs.dbgridinv).max(het_inputs.dVbmin);
		VbF.col(het_inputs.nb - 1) = VaF.col(het_inputs.nb - 2);

		VbB.block(0, 1, het_inputs.na, het_inputs.nb - 1)
			= VbF.block(0, 0, het_inputs.na, het_inputs.nb - 1);
		VbB.col(0) = VaF.col(1);
	}




	bool
		hour_secant(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, Het_workspace& ws)
	{
		ws.h0[r](k, m) = het_inputs.hour_high;
		ws.fhour[r](k, m) = ws.fhour_max[r](k, m);

		if (std::abs(ws.fhour[r](k, m)) < het_inputs.tols_hour)
		{
			return true;
		}
		ws.hour_low[r](k, m) = ws.hour_min[r](k, m);
		ws.fhour_low[r](k, m) = ws.fhour_min[r](k, m);

		if (std::abs(ws.fhour_low[r](k, m)) < het_inputs.tols_hour)
		{
			ws.h0[r](k, m) = ws.hour_low[r](k, m);
			return true;
		}

		if (ws.fhour[r](k, m) * ws.fhour_low[r](k, m) > 0)
		{
			std::cerr << "Hour_secant: Root must be bracketed! " << std::endl;
			return false;
		}

		for (int n = 0; n < het_inputs.maxiter_hour; ++n)
		{
			ws.df[r](k, m) = (ws.fhour[r](k, m) - ws.fhour_low[r](k, m)) / (ws.h0[r](k, m) - ws.hour_low[r](k, m));
			ws.hour_low[r](k, m) = ws.h0[r](k, m);
			ws.fhour_low[r](k, m) = ws.fhour[r](k, m);
			ws.h0[r](k, m) -= ws.fhour[r](k, m) / ws.df[r](k, m);
			fhour_supply(het_inputs, r, k, m, sd, ws.h0[r](k, m), ws, ws.fhour[r](k, m));
			if (std::abs(ws.fhour[r](k, m)) < het_inputs.tols_hour)
			{
				return true;
			}
		}

		return false;
	}


	bool
		hour_bisec(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, Het_workspace& ws)
	{
		ws.hour_low[r](k, m) = ws.hour_min[r](k, m);
		ws.fhour_low[r](k, m) = ws.fhour_min[r](k, m);

		if (std::abs(ws.fhour_low[r](k, m)) < het_inputs.tols_hour)
		{
			ws.h0[r](k, m) = ws.hour_low[r](k, m);
			return true;
		}

		ws.hour_high[r](k, m) = het_inputs.hour_high;
		ws.fhour_high[r](k, m) = ws.fhour_max[r](k, m);
		if (std::abs(ws.fhour_high[r](k, m)) < het_inputs.tols_hour)
		{
			ws.h0[r](k, m) = het_inputs.hour_high;
			return true;
		}

		if (ws.fhour_high[r](k, m) * ws.fhour_low[r](k, m) > 0)
		{

			std::cerr << "Hour_bisec: Root must be bracketed! " << std::endl;
			return false;
		}

		for (int n = 0; n < het_inputs.maxiter_hour; ++n)
		{
			ws.h0[r](k, m) = (ws.hour_high[r](k, m) + ws.hour_low[r](k, m)) * 0.5;
			fhour_supply(het_inputs, r, k, m, sd, ws.h0[r](k, m), ws, ws.fhour[r](k, m));
			if (std::abs(ws.fhour[r](k, m)) < het_inputs.tols_hour)
			{
				return true;
			}
			if (ws.fhour[r](k, m) * ws.fhour_low[r](k, m) < 0)
			{
				ws.hour_high[r](k, m) = ws.h0[r](k, m);
			}
			else
			{
				ws.hour_low[r](k, m) = ws.h0[r](k, m);
				ws.fhour_low[r](k, m) = ws.fhour[r](k, m);
			}
		}
		return false;
	}

	void
		hour_min(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, Het_workspace& ws)
	{
		ws.hour_min[r](k,m) = std::max(-(het_inputs.inc[r](k, m) + sd)
			/ het_inputs.after_tax_wage[r] + 1e-8, 0.0);
	}

	void
		fhour_foc(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, const Het_workspace& ws, const double& hour, double& fhour)
	{
		fhour = hour - utility2inv(het_inputs, het_inputs.after_tax_wage[r]
			* utility1(het_inputs, std::max(het_inputs.inc[r](k, m) + sd + het_inputs.after_tax_wage[r] * hour, 1e-8)));
	}

	void
		fhour_supply(const Het_Inputs& het_inputs, const int& r, const int& k, const int& m, const double& sd, const double& hour, Het_workspace& ws, double& fhour)
	{
		fhour_foc(het_inputs, r, k, m, sd, ws, hour, fhour);
	}
}
