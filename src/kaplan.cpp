#include <map>
#include <string>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_roots.h>

#include "Kaplan.h"
#include "Two_asset.h"

#include "Support.h"
#include "Calibration.h"
#include "Steady_state.h"
#include "Het_block.h"
#include "Solver.h"


#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;

namespace TA
{
	namespace KAPLAN
	{

		void
			set_parameters(Calibration& cal, SteadyState& ss)
		{
			double rb = cal.params("rb");
			double pi = cal.params("pi");
			double i = rb + pi;
			double lifespan = cal.params("lifespan");
			double deathrate = 1 / (4 * lifespan);
			double rho_zeta = std::exp(-0.5);
			/*Solve illiquid r and KYratio by arbitrage condition*/
			double alpha = cal.params("alpha");
			double deprec = cal.params("deprec");
			double epsilon = cal.params("epsilon");
			double mc = (epsilon - 1.0) / epsilon;
			double targetIllYratio = cal.params("targetIllYratio");
			double profdistfrac = cal.params("profdistfrac");
			double tax = cal.params("tax");
			double data_output = cal.params("data_annual_output");
			double transfer = 40000 * tax / data_output;
			double tfp = cal.params("tfp");

			// No-arbitrage condition
			// (1- mc) / (targetIllYratio - KYratio) = alpha * mc / KYratio - deprec
			double a = deprec;
			double b = -deprec * targetIllYratio - alpha * mc - profdistfrac * (1 - mc);
			double c = targetIllYratio * alpha * mc;
			double KYratio = (-b - std::sqrt(b * b - 4 * a * c)) / (2 * a);
			double equityYratio = targetIllYratio - KYratio;
			double r = profdistfrac * (1 - mc) / equityYratio;
			
			/************************************************************/


			double KNratio = std::pow(tfp * KYratio, 1.0 / (1.0 - alpha));
			double wage = mc * (1 - alpha) * tfp * std::pow(KNratio, alpha);

			double sigma = cal.params("sigma");
			double frisch = cal.params("frisch");
			double akink = 0.03 * targetIllYratio / 16;
			// It will change in het block iterations, this is a guess.
			

			const Options& op = cal.options();
			Hour_supply hs = op.hour_supply();
			double meanlabeff = 0.0;
			double vphi = 0.0;
			double profit = 0.0;
			switch (hs)
			{
			case Hour_supply::Seq:
				meanlabeff = KYratio / KNratio / (1.0 / 3.0);
				vphi = meanlabeff / (std::pow(0.75, -sigma) * std::pow(1.0 / 3.0, 1.0 / frisch));
				profit = (1 - mc) * 16 / KYratio;
				break;
			case Hour_supply::GHH:
				meanlabeff = KYratio / KNratio / (1.0 / 3.0);
				vphi = meanlabeff / (std::pow(1.0 / 3.0, 1.0 / frisch));
				profit = (1 - mc) * 14 / KYratio;
				break;
			case Hour_supply::NoSupply:
				meanlabeff = std::pow(KYratio, alpha / (alpha - 1)) * std::pow(tfp, 1 / (alpha - 1));
				vphi = 0.0;
				profit = (1 - mc) * 12 / KYratio;
				break;
			default:
				break;
			}
			if (op.run().solve_ss && !op.run().calibration && !op.run().solve_equm)
			{
				profit = cal.params("targetprofit");
				wage = cal.params("targetwage");
				r = cal.params("targetra");
			}
			double rk = r + deprec;
			construct_unordered_map_from_list(ss.get_aggregators(),
				{ {"alpha", alpha}, {"epsilon", epsilon},
				{"wage", wage},{"ra", r}, {"rd", rb}, {"tfp", tfp}, {"zeta", 0.0},{"dG", 0.0},
				{"r", r},  {"tax", tax}, {"rb", rb},  {"i", i}, {"pi", 0.0}, {"adj_price", 0.0},
				{"rk", rk}, {"mc", mc}, {"transfer", transfer}
				});
			append_parameters(
				{ {"transfer", transfer}, {"deathrate", deathrate}, {"targetKYratio", KYratio},
				{"mc", mc}, {"targetKNratio", KNratio}, {"targetprofit", profit}, {"targetwage", wage},
				{"rb", rb}, {"targetra", r}, {"rho_zeta", rho_zeta},
				{"vphi", vphi},  {"meanlabeff", meanlabeff}, {"akink", akink}}, cal
			);
			cal.make_grid();
		}

		void
			solve_res_aggregators(const Het_Inputs& het_inputs, const Het_Outputs& htp,
				SteadyState& ss)
		{
			if (het_inputs.options.run().solve_equm)
			{
				ss.get_aggregators()["wage"] = het_inputs.wage;
				ss.get_aggregators()["profitI"] = het_inputs.profit;
				ss.get_aggregators()["r"] =
					ss.get_aggregators()["ra"] = het_inputs.ra;
			}
			ss.get_aggregators()["LiqYratio"] = htp.LiqYratio;
			ss.get_aggregators()["IllYratio"] = htp.IllYratio;
			ss.get_aggregators()["KYratio"] = htp.KYratio;
			ss.get_aggregators()["KNratio"] = htp.KNratio;
			ss.get_aggregators()["eqI"] = htp.Ea - htp.K;
			ss.get_aggregators()["eqK"] = htp.K;
			ss.get_aggregators()["TobinQ"] = 1;
			ss.get_aggregators()["Inv"] = htp.deprec * htp.K;
			ss.get_aggregators()["K"] = htp.K;
			ss.get_aggregators()["rk"] = het_inputs.ra + htp.deprec;
			ss.get_aggregators()["profitI"] = (1 - htp.mc)* htp.K / htp.KYratio;
			ss.get_aggregators()["profitK"] = ss.get_aggregators()["rk"] * htp.K - ss.get_aggregators()["Inv"];
			ss.get_aggregators()["Y"] = htp.Y;
			ss.get_aggregators()["els_bond"] = het_inputs.cal.params("els_bond") * htp.Y;
			ss.get_aggregators()["res_bond"] = 0.0;
			ss.get_aggregators()["L"] = htp.Elabor;
			ss.get_aggregators()["Gov_debt"] = htp.Eb;



			ss.get_aggregators()["Gov_spend"] = het_inputs.tax * het_inputs.wage * htp.Elabor
				+ het_inputs.tax * (1 - het_inputs.profdistfrac) * het_inputs.profit
				- het_inputs.rb * htp.Eb - het_inputs.transfer;
			ss.get_aggregators()["Gov_surplus"] = het_inputs.tax * het_inputs.wage * htp.Elabor
				- het_inputs.transfer - ss.get_aggregators()["Gov_spend"]
				+ het_inputs.tax * (1 - het_inputs.profdistfrac) * het_inputs.profit;
			ss.get_aggregators()["GDP"] = ss.get_aggregators()["Ec"] + ss.get_aggregators()["Inv"] + ss.get_aggregators()["Gov_spend"];
		}

		void
			solve_targeting_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp)
		{
			TA::solve_targeting_moments(het_inputs, ws, htp);
			if (htp.Ea < 0.1) htp.K = htp.Ea;
			else
			{
				Capital_params cp = { htp.Y, htp.Elabor, htp.Ea, htp.mc, htp.alpha,
					htp.tfp, het_inputs.profdistfrac, het_inputs.ra, 0.0, 0.0, htp.tols_policy, htp.maxiter_policy };
				if (capital_targets(0.00001 * htp.Ea, &cp) > 0) htp.Kub = 0.00001 * htp.Ea;
				else if (capital_targets(0.0001 * htp.Ea, &cp) > 0) htp.Kub = 0.0001 * htp.Ea;
				else if (capital_targets(0.001 * htp.Ea, &cp) > 0) htp.Kub = 0.001 * htp.Ea;
				else if (capital_targets(0.01 * htp.Ea, &cp) > 0) htp.Kub = 0.01 * htp.Ea;
				else if (capital_targets(0.1 * htp.Ea, &cp) > 0) htp.Kub = 0.1 * htp.Ea;
				else htp.Kub = htp.Ea;
				cp.high = htp.Kub;
				solve_root(cp, &capital_targets, htp.K);
			}
			
			if (het_inputs.profdistfrac < 0.99)
			{
				het_inputs.profit = (1 - htp.mc) * htp.K / htp.targetKYratio;
				het_inputs.set_labor_income();
			}

			htp.KNratio = htp.K / htp.Elabor;
			htp.KYratio = (htp.K < 1e-8 ? 0 : std::pow(htp.KNratio, 1 - htp.alpha)) / htp.tfp;
			htp.Y = std::max(htp.Y, 0.01);
			htp.IllYratio = htp.Ea / htp.Y;
			htp.LiqYratio = htp.Eb / htp.Y;
		}

		double
			capital_targets(double K, void* params_)
		{
			Capital_params* params = (Capital_params*)params_;
			params->Y = params->tfp * std::pow(K, params->alpha) * std::pow(params->Elabor, 1 - params->alpha);
			return params->profdistfrac * (1 - params->mc) * params->Y - (params->Ea - K) * params->ra;
		}

		void
			solve_ss_het_one_step_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp)
		{
			TA::solve_targeting_moments(het_inputs, ws, htp);
			htp.KYratio = htp.alpha * htp.mc / (het_inputs.ra + htp.deprec);
			htp.K = htp.Ea / (1 + (1 - htp.mc) * het_inputs.profdistfrac / het_inputs.ra / htp.KYratio);
			htp.KNratio = htp.K / htp.Elabor;
			htp.Y = htp.tfp * std::pow(htp.K, htp.alpha) * std::pow(htp.Elabor, 1 - htp.alpha);
			htp.IllYratio = htp.Ea / htp.Y;
			htp.LiqYratio = htp.Eb / htp.Y;
		}


		void
			print_equm_head()
		{
			std::cout << std::setw(16) << std::left << "Iterations"
				<< std::setw(16) << std::left << "ra"
				<< std::setw(16) << std::left << "wage"
				<< std::setw(16) << std::left << "profit"
				<< std::setw(16) << std::left << "KNratio"
				<< std::setw(16) << std::left << "Error(relative)" << std::endl;
		}

		void
			print_equm_iteration(const Het_equm& params, const Het_Inputs& het_inputs)
		{
			std::cout << std::setw(16) << std::left << params.iter
				<< std::setw(16) << std::left << het_inputs.ra
				<< std::setw(16) << std::left << het_inputs.wage
				<< std::setw(16) << std::left << het_inputs.profit
				<< std::setw(16) << std::left << params.x
				<< std::setw(16) << std::left << params.eps << std::endl;
		}
		void
			solve_ss_het_equm(Het_equm& params, Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp)
		{
			params.x = htp.targetKNratio;
			htp.KYratio = htp.targetKYratio;
			params.print_head();
			for (params.iter = 0; params.iter < params.maxiter_equm; ++params.iter)
			{
				solve_convergent_pols(htp, het_inputs, ws);
				params.moments_fun(het_inputs, ws, htp);
				params.x_next = htp.KNratio;
				params.eps = std::abs(params.x_next / params.x - 1.0);
				if (params.eps < params.tols_equm)
				{
					std::cout << "Converged:" << "\n";
					params.print_iter(params, het_inputs);
					goto end;
				}
				params.print_iter(params, het_inputs);
				params.x = htp.KNratio = (1.0 - params.step_equm) * params.x + params.step_equm * params.x_next;
				params.moments_update_fun(het_inputs, ws, htp);
			}
			cout << "Couldn't find steady state after " << params.maxiter_equm << " iterations!!" << endl;
		end:
			params.moments_res_fun(het_inputs, ws, htp);
		}

		void
			solve_equm_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp)
		{
			// Ea - K = (1 - mc) * profdistfrac / (ra * KYratio) * K
			htp.Ea = expectation(het_inputs.agrid, ws.dist.reshaped(), het_inputs.nesk, het_inputs.nab);
			switch (het_inputs.hour_supply)
			{
			case Hour_supply::Seq:
				htp.Elabor = expectation(ws.hour, ws.dist.reshaped(), het_inputs.nsk, het_inputs.nab, het_inputs.sgrid);
				break;
			case Hour_supply::GHH:
				htp.Elabor = 0.0;
				for (int r = 0; r < het_inputs.nsk; ++r)
				{
					htp.Elabor += het_inputs.hour_ghh(r) * het_inputs.sgrid(r) * ws.dist.col(r).sum();
				}
				break;
			case Hour_supply::NoSupply:
				htp.Elabor = het_inputs.meanlabeff;
				break;
			default:
				break;
			}

			htp.K = htp.Ea / (1 + (1 - htp.mc) * het_inputs.profdistfrac / het_inputs.ra / htp.KYratio);
			htp.KNratio = htp.K / htp.Elabor;
		}

		void
			solve_equm_res_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp)
		{
			TA::solve_targeting_moments(het_inputs, ws, htp);
			//switch (het_inputs.hour_supply)
			//{
			//case Hour_supply::Seq:
			//	htp.Ehour = expectation(ws.hour, ws.dist.reshaped(), het_inputs.nsk, het_inputs.nab);
			//	break;
			//case Hour_supply::GHH:
			//	htp.Ehour = het_inputs.Ehour_ghh;
			//	break;
			//case Hour_supply::NoSupply:
			//	htp.Ehour = 1.0;
			//	break;
			//default:
			//	break;
			//}
			htp.Y = htp.tfp * std::pow(htp.K, htp.alpha) * std::pow(htp.Elabor, 1 - htp.alpha);
			htp.IllYratio = htp.Ea / htp.Y;
			htp.LiqYratio = htp.Eb / htp.Y;

		}

		void
			update_equm_moments(Het_Inputs& het_inputs, Het_workspace& ws, Het_Outputs& htp)
		{
			htp.KYratio = std::pow(htp.KNratio, 1 - htp.alpha) / htp.tfp;
			if (het_inputs.profdistfrac < 0.99) het_inputs.profit = (1 - htp.mc) * htp.K / htp.KYratio;
			het_inputs.wage = htp.mc * (1 - htp.alpha) * htp.tfp * std::pow(htp.KNratio, htp.alpha);
			het_inputs.ra = htp.mc * htp.alpha / htp.KYratio - htp.deprec;
			het_inputs.set_labor_income();
			het_inputs.set_income();
			//het_inputs.solve_binding_pols(het_inputs, ws);
		}

		void
			check_all_market_clearing(const Calibration& cal, SteadyState& ss)
		{
			double Eadj = ss.get_aggregator("Eadj");
			double Ec = ss.get_aggregator("Ec");
			double Gov_spend = ss.get_aggregator("Gov_spend");
			double Y = ss.get_aggregator("Y");
			double Inv = ss.get_aggregator("Inv");
			double rb_wedge = cal.params("rb_wedge");
			double K = ss.get_aggregator("K");
			const het& bgrid = cal.b();
			const Eigen::VectorXd& bdist = ss.get_marg_dist("HA_B");

			double good_mkt_clear = Ec + Inv + Gov_spend + Eadj - rb_wedge * (bgrid.min(0.0) * bdist.array()).sum() - Y;
			construct_unordered_map_from_list(ss.get_aggregators(),
				{ {"good_mkt_clear", good_mkt_clear}});
		}

		void 
			solve_both_binding_pols(const Het_Inputs& het_inputs, Het_workspace& ws)
		{
			int m = 0;
			int k = 0;
			for (int r = 0; r < het_inputs.nesk; ++r)
			{
				for (m = 0; m < het_inputs.nb; ++m)
				{
					for (k = 0; k < het_inputs.na; ++k)
					{
						solve_binding_cons(het_inputs, -het_inputs.a_drift(k, m), ws, r, k, m);
						ws.H00[r](k, m) = ws.utility0[r](k, m);
						ws.c00[r](k, m) = ws.c0[r](k, m);
						ws.sd0[r](k, m) = ws.sdtemp[r](k, m);
						ws.hour00[r](k, m) = ws.hour[r](k, m);
					}
				}
				
			}
		}

		void
			solve_binding_pols(const Het_Inputs& het_inputs, Het_workspace& ws, int r)
		{
			int k = 0;
			int m = 0;
			for (m = 0; m < het_inputs.nb; ++m)
			{
				for (k = 0; k < het_inputs.na; ++k)
				{
					ws.FdminB[r](k, m) = FdVa(het_inputs, ws, ws.VaB[r](k, m), het_inputs.dmin(k, m), r, k, m);
					ws.FdminF[r](k, m) = FdVa(het_inputs, ws, ws.VaF[r](k, m), het_inputs.dmin(k, m), r, k, m);
					if (ws.FdminB[r](k, m) >= 0 && ws.FdminF[r](k, m) >= 0) ws.d0[r](k, m) = -het_inputs.a_drift(k,m);
					else if (ws.FdminB[r](k, m) < 0 || ws.FdminF[r](k, m) >= 0)
					{
						if (-het_inputs.a_drift(k, m) <= het_inputs.dmin(k, m) || FdVa(het_inputs, ws, ws.VaB[r](k, m), -het_inputs.a_drift(k, m), r, k, m) <= 0)
						{
							ws.d0[r](k, m) = -het_inputs.a_drift(k, m);
							ws.H0[r](k, m) = ws.H00[r](k, m);
						}
						else
						{
							Fd_bisec(het_inputs, ws, ws.VaB[r](k, m), het_inputs.dmin(k,m), -het_inputs.a_drift(k, m), ws.FdminB[r](k, m), ws.d0[r](k, m), r, k, m);
							ws.H0[r](k, m) = ws.utility0[r](k, m) + (het_inputs.a_drift(k, m) + ws.d0[r](k, m)) * ws.VaB[r](k, m);
							if (k == 0) ws.H0[r](k, m) = het_inputs.H_min;
						}
					}
					else if (ws.FdminB[r](k, m) > 0 || ws.FdminF[r](k, m) < 0)
					{
						solve_FdVaF(het_inputs, ws, ws.d0[r](k, m), ws.H0[r](k, m), r, k, m);
					}
					else
					{
						if (-het_inputs.a_drift(k, m) <= het_inputs.dmin(k, m) || FdVa(het_inputs, ws, ws.VaB[r](k, m), -het_inputs.a_drift(k, m), r, k, m) <= 0)
						{
							solve_FdVaF(het_inputs, ws, ws.d0[r](k, m), ws.H0[r](k, m), r, k, m);
						}
						else
						{
							solve_FdVaF(het_inputs, ws, ws.dF0[r](k, m), ws.HF0[r](k, m), r, k, m);
							ws.cF0[r](k, m) = ws.c0[r](k, m);
							ws.hourF0[r](k, m) = ws.h0[r](k, m);
							Fd_bisec(het_inputs, ws, ws.VaB[r](k, m), het_inputs.dmin(k, m), -het_inputs.a_drift(k, m), ws.FdminB[r](k, m), ws.d0[r](k, m), r, k, m);
							ws.H0[r](k, m) = ws.utility0[r](k, m) + (het_inputs.a_drift(k, m) + ws.d0[r](k, m)) * ws.VaB[r](k, m);
							if (k == 0) ws.H0[r](k, m) = het_inputs.H_min;

							if (ws.HF0[r](k, m) > ws.H0[r](k, m))
							{
								ws.d0[r](k, m) = ws.dF0[r](k, m);
								ws.H0[r](k, m) = ws.HF0[r](k, m);
								ws.c0[r](k, m) = ws.cF0[r](k, m);
								ws.h0[r](k, m) = ws.hourF0[r](k, m);
							}
						}
					}

					if (ws.H0[r](k, m) <= ws.H00[r](k, m))
					{
						ws.d0[r](k, m) = -het_inputs.a_drift(k,m);
						ws.c0[r](k, m) = ws.c00[r](k, m);
						ws.h0[r](k, m) = ws.hour00[r](k, m);
						ws.H0[r](k, m) = ws.H00[r](k, m);
					}
					ws.sa0[r](k, m) = het_inputs.a_drift(k, m) + ws.d0[r](k, m);
				}
			}

		}

		void
			solve_FdVaF(const Het_Inputs& het_inputs, Het_workspace& ws, double& droot, double& H, int r, int k, int m)
		{
			if (FdVa(het_inputs, ws, ws.VaF[r](k, m), -het_inputs.a_drift(k, m), r, k, m) >= 0)
			{
				droot = -het_inputs.a_drift(k, m);
				H = ws.H00[r](k, m);
			}
			else
			{
				if (FdVa(het_inputs, ws, ws.VaF[r](k, m), 1e-9, ws.Fdtemp[r](k, m), r, k, m) < 0)
				{
					ws.d_high[r](k, m) = het_inputs.dmax;
					ws.d_low[r](k, m) = 1e-9;
				}
				else if (FdVa(het_inputs, ws, ws.VaF[r](k, m), -1e-9, ws.Fdtemp[r](k, m), r, k, m) < 0)
				{
					ws.d_high[r](k, m) = 0;
					ws.d_low[r](k, m) = 0;
				}
				else
				{
					ws.d_high[r](k, m) = -1e-9;
					ws.d_low[r](k, m) = het_inputs.dmin(k, m);
					//FdVa(het_inputs, ws, ws.VaF[r](k, m), ws.d_low[r](k, m), ws.Fdtemp[r](k, m), r, k, m);
				}
				Fd_bisec(het_inputs, ws, ws.VaF[r](k, m), ws.d_low[r](k, m), ws.d_high[r](k, m), ws.FdminF[r](k, m), droot, r, k, m);
				H = ws.utility0[r](k, m) + (het_inputs.a_drift(k, m) + droot) * ws.VaF[r](k, m);
			}
			if (k == het_inputs.na - 1) H = het_inputs.H_min;
		}

		bool 
			Fd_bisec(const Het_Inputs& het_inputs, Het_workspace& ws, const double& Va, 
				const double& dlow, const double& dhigh, const double& Flow, double& droot, int r, int k, int m)
		{
			if (dhigh <= dlow + het_inputs.tols_d)
			{
				droot = dlow;
				return true;
			}
			droot = (dhigh + dlow)* 0.5;
			ws.Fd0[r](k, m) = FdVa(het_inputs, ws, Va, droot, r, k, m);

			ws.Fd_low[r](k, m) = Flow;

			for (int n = 0; n < het_inputs.maxiter_d; ++n)
			{
				if (ws.Fd0[r](k, m) * ws.Fd_low[r](k, m) < 0)
				{
					ws.d_high[r](k,m) = droot;
				}
				else
				{
					ws.d_low[r](k, m) = droot;
					ws.Fd_low[r](k, m) = ws.Fd0[r](k, m);
				}
				droot = (ws.d_high[r](k, m) + ws.d_low[r](k, m)) * 0.5;

				ws.Fd0[r](k, m)=FdVa(het_inputs, ws, Va, droot, r, k, m);
				if (std::abs(ws.Fd0[r](k, m)) < het_inputs.tols_d
					|| std::abs(ws.d_high[r](k, m) - ws.d_low[r](k, m)) < het_inputs.tols_d)
				{
					return true;
				}
			}
			return false;
		}

		double
			FdVa(const Het_Inputs& het_inputs, Het_workspace& ws, const double& Va,const double& d,  int r, int k, int m)
		{
			solve_binding_cons(het_inputs, d, ws, r, k, m);
			switch (het_inputs.adj_fun)
			{
			case Adj_fun::KAPLAN_ADJ:
				return utility1(het_inputs, ws.c0[r](k, m)) * (1 + Adj_fun_kaplan::adj1(het_inputs, d, k, m)) - Va;
				break;
			case Adj_fun::AUCLERT_ADJ:
				return utility1(het_inputs, ws.c0[r](k, m)) * (1 + Adj_fun_auclert::adj1(het_inputs, d, k ,m)) - Va;
				break;
			default:
				break;
			}

		}

		double
			FdVa(const Het_Inputs& het_inputs, Het_workspace& ws, const double& Va, const double& d, double& Fd, int r, int k, int m)
		{
			solve_binding_cons(het_inputs, d, ws, r, k, m);
			switch (het_inputs.adj_fun)
			{
			case Adj_fun::KAPLAN_ADJ:
				Fd = utility1(het_inputs, ws.c0[r](k, m)) * (1 + Adj_fun_kaplan::adj1(het_inputs, d, k, m)) - Va;
				break;
			case Adj_fun::AUCLERT_ADJ:
				Fd = utility1(het_inputs, ws.c0[r](k, m)) * (1 + Adj_fun_auclert::adj1(het_inputs, d, k, m)) - Va;
				break;
			default:
				break;
			}
			return Fd;
		}

		void
			solve_binding_cons(const Het_Inputs& het_inputs, const double& d,
				Het_workspace& ws, int r, int k, int m)
		{
			switch (het_inputs.adj_fun)
			{
			case Adj_fun::KAPLAN_ADJ:
				ws.sdtemp[r](k,m) = -d - Adj_fun_kaplan::adj(het_inputs, std::abs(d), k, m);
				break;
			case Adj_fun::AUCLERT_ADJ:
				ws.sdtemp[r](k, m) = -d - Adj_fun_auclert::adj(het_inputs, std::abs(d), k, m);
				break;
			default:
				break;
			}
			switch (het_inputs.hour_supply)
			{
			case Hour_supply::Seq:
				hour_min(het_inputs, r, k, m, ws.sdtemp[r](k, m), ws);
				if (ws.hour_min[r](k, m) + het_inputs.tols_hour >= het_inputs.hour_high)
				{
					ws.h0[r](k, m) = het_inputs.hour_high;
				}
				else
				{
					fhour_supply(het_inputs, r, k, m, ws.sdtemp[r](k, m), ws.hour_min[r](k, m), ws, ws.fhour_min[r](k, m));
					fhour_supply(het_inputs, r, k, m, ws.sdtemp[r](k, m), het_inputs.hour_high, ws, ws.fhour_max[r](k, m));
					if (ws.fhour_min[r](k, m) >= 0) ws.h0[r](k, m) = ws.hour_min[r](k, m);
					else if (ws.fhour_max[r](k, m) <= 0) ws.h0[r](k, m) = het_inputs.hour_high;
					else
					{
						if (!hour_secant(het_inputs, r, k, m, ws.sdtemp[r](k, m), ws))
							if (!hour_bisec(het_inputs, r, k, m, ws.sdtemp[r](k, m), ws))
							{
								std::cerr << "Couldn't find consumption when binding!!" << std::endl;
								std::exit(0);
							}
					}
				}
				ws.c0[r](k, m) = std::max(het_inputs.inc[r](k, m) + het_inputs.after_tax_wage[r] * ws.h0[r](k, m) + ws.sdtemp[r](k, m), 1e-8);
				ws.utility0[r](k, m) = util_cons(het_inputs, ws.c0[r](k, m)) + disutil_hour(het_inputs, ws.h0[r](k, m));
				break;
			case Hour_supply::GHH:
				break;
			case Hour_supply::NoSupply:
				ws.c0[r](k, m) = std::max(het_inputs.inc[r](k, m) + het_inputs.after_tax_wage[r] + ws.sdtemp[r](k, m), 1e-8);
				ws.utility0[r](k, m) = util_cons(het_inputs, ws.c0[r](k, m));
				break;
			default:
				break;
			}
		}

//		void
//			solve_binding_cons(const Het_Inputs& het_inputs, const het2& sd, Het_workspace& ws)
//		{
//			int r = 0;
//			int k = 0;
//			int m = 0;
//
//			switch (het_inputs.hour_supply)
//			{
//			case Hour_supply::Seq:
//#pragma omp parallel for private(m, k)
//				for (r = 0; r < het_inputs.nsk; ++r)
//				{
//					ws.sc0[r].setZero();
//					for (m = 0; m < het_inputs.nb; ++m)
//					{
//						for (k = 0; k < het_inputs.na; ++k)
//						{
//							hour_min(het_inputs, r, k, m,  ws);
//							fhour_supply(het_inputs, r, k, m, ws.hour_min[r], ws, ws.fhour_min[r]);
//							fhour_supply(het_inputs, r, k, m, het_inputs.hour_high, ws, ws.fhour_max[r]);
//							if (ws.fhour_min[r] >= 0) ws.h0[r](k, m) = ws.hour_min[r];
//							else if (ws.fhour_max[r] <= 0) ws.h0[r](k, m) = het_inputs.hour_high;
//							else
//							{
//								if (!hour_secant(het_inputs, r, k, m, ws))
//									if (!hour_bisec(het_inputs, r, k, m, ws))
//									{
//										std::cerr << "Couldn't find consumption when binding!!" << std::endl;
//										std::exit(0);
//									}
//							}
//							ws.c0[r](k, m) = het_inputs.inc[r](k, m) + het_inputs.after_tax_wage[r] * ws.h0[r](k, m);
//							ws.Hc0[r](k, m) = util_cons(het_inputs, ws.c0[r](k, m)) + disutil_hour(het_inputs, ws.h0[r](k, m));
//						}
//					}
//				}
//				break;
//			case Hour_supply::GHH:
//#pragma omp parallel for
//				for (r = 0; r < het_inputs.nsk; ++r)
//				{
//					ws.sc0[r].setZero();
//					ws.c0[r] = het_inputs.inc[r] + het_inputs.after_tax_wage[r] * het_inputs.hour_ghh(r);
//					ws.Hc0[r] = (ws.c0[r] + het_inputs.disutility_hour_ghh(r) > 0).select(util_cons(het_inputs, ws.c0[r] + het_inputs.disutility_hour_ghh(r)), het_inputs.H_min);
//				}
//				break;
//			default:
//				break;
//			}
//
//		}

		void
			solve_unbinding_cons(const Het_Inputs& het_inputs, const Het_workspace& ws, const het2& Vb, int r,
				het2& c, het2& sc, het2& hour, het2& utility)
		{
			switch (het_inputs.hour_supply)
			{
			case Hour_supply::Seq:
				c = utility1inv(het_inputs, Vb);
				hour = utility2inv(het_inputs, het_inputs.after_tax_wage[r] * Vb).min(het_inputs.hour_high);
				sc = het_inputs.inc[r] + het_inputs.after_tax_wage[r] * hour - c;
				utility = util_cons(het_inputs, c) + disutil_hour(het_inputs, hour);
				break;
			case Hour_supply::GHH:
				c = utility1inv(het_inputs, Vb) - het_inputs.disutility_hour_ghh(r);
				sc = het_inputs.inc[r] + het_inputs.after_tax_wage[r] * het_inputs.hour_ghh(r) - c;
				utility = (c + het_inputs.disutility_hour_ghh(r) > 0).select(util_cons(het_inputs, c + het_inputs.disutility_hour_ghh(r)), het_inputs.H_min);
				break;
			case Hour_supply::NoSupply:
				c = utility1inv(het_inputs, Vb);
				sc = het_inputs.inc[r] + het_inputs.after_tax_wage[r] - c;
				utility = util_cons(het_inputs, c);
			default:
				break;
			}

		}

		void
			solve_unbinding_res(const Het_Inputs& het_inputs, const Het_workspace& ws, 
				const het2& Vb, const het2& Va, const het2& sc, const het2& utility, int r,
				het2& d, het2& sd, het2& sa, het2& sb, het2& H)
		{
			switch (het_inputs.adj_fun)
			{
			case Adj_fun::KAPLAN_ADJ:
				d = Adj_fun_kaplan::adj1inv(het_inputs, Va / Vb - 1);
				sb = sc -d - Adj_fun_kaplan::adj(het_inputs, d.abs());
				break;
			case Adj_fun::AUCLERT_ADJ:
				d = Adj_fun_auclert::adj1inv(het_inputs, Va / Vb - 1);
				sb = sc -d - Adj_fun_auclert::adj(het_inputs, d.abs());
				break;
			default:
				break;
			}
			sa = het_inputs.a_drift + d;
			H = utility + Vb * sb + Va * sa;
		}


		void
			solve_rhs(const Het_Inputs& het_inputs, Het_workspace& ws, int r)
		{
			if (r < het_inputs.nsk)
			{
				switch (het_inputs.hour_supply)
				{
				case Hour_supply::Seq:
					ws.hour[r] = (ws.indFB[r].cast<double>() + ws.indBB[r].cast<double>() + ws.ind0B[r].cast<double>()) * ws.hB[r]
						+ (ws.indFF[r].cast<double>() + ws.indBF[r].cast<double>() + ws.ind0F[r].cast<double>()) * ws.hF[r]
						+ ws.ind0[r].cast<double>() * ws.h0[r];
					ws.rhs.col(r) = -(util_cons(het_inputs, ws.c[r]) + disutil_hour(het_inputs, ws.hour[r])).reshaped().matrix()
						- ws.V * het_inputs.offdiagHJB.row(r).transpose();
					break;
				case Hour_supply::GHH:
					ws.rhs.col(r) = -util_cons(het_inputs, ws.c[r] + het_inputs.disutility_hour_ghh(r)).reshaped().matrix()
						- ws.V * het_inputs.offdiagHJB.row(r).transpose();
					break;
				case Hour_supply::NoSupply:
					ws.rhs.col(r) = -util_cons(het_inputs, ws.c[r]).reshaped().matrix()
						- ws.V * het_inputs.offdiagHJB.row(r).transpose();
					break;
				default:
					break;
				}

			}
			else
			{
				ws.rhs.col(r) = -(util_cons(het_inputs, ws.c[r]).reshaped() - het_inputs.unemployedloss).matrix()
					- ws.V * het_inputs.offdiagHJB.row(r).transpose();
			}
		}

	}
}