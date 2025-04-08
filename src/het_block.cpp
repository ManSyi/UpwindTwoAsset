#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <set>
#include <string>
#include <memory>
#include <math.h>
#include <unordered_map>
#include <chrono>
#include <omp.h>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/KroneckerProduct> 


#include "Support.h"
#include "Calibration.h"
#include "Steady_state.h"

#include "Het_block.h"
#include "Two_asset.h"
#include "Kaplan.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using namespace std;







void
Het_Inputs::init()
{
	rho = cal.params("rho");
	deathrate = cal.params("deathrate");
	// Monetary shock persistence
	rho_zeta = cal.params("rho_zeta");
	chi0 = cal.params("chi0");
	chi2 = cal.params("chi2");
	chi1 = (options.run().add_innaction ? chi1 = 1 - chi0 : cal.params("chi1"));
	sigma = cal.params("sigma");
	a_kink = cal.params("akink");
	meanlabeff = cal.params("meanlabeff");
	transfer = cal.params("transfer");
	tax = cal.params("tax");
	dVamin = cal.params("dVamin");
	dVbmin = cal.params("dVbmin");
	DeltaHJB = cal.params("DeltaHJB");
	DeltaKFE = cal.params("DeltaKFE");
	DeltaFeyman = cal.params("DeltaFeyman");
	DeltaDecomp = 1.0;
	
	maxiter_discmpc = cal.size("maxiter_discmpc");
	tols_discmpc = cal.params("tols_discmpc");

	profdistfrac = cal.params("profdistfrac");

	after_tax_wage.resize(nsk);
	inc_labor.resize(nesk);
	init_empty_tensor(inc, na, nb, nesk);
	
	akinkgrid = agrid.max(a_kink);

	skill_dist_extend.setZero(nesk);
	switch (hour_supply)
	{
	case Hour_supply::GHH:
		hour_ghh.setZero(nsk);
		disutility_hour_ghh.setZero(nsk);
		break;
	default:
		break;
	}

	switch (model)
	{
	case Model::KAPLAN:
		frisch = cal.params("frisch");
		vphi = cal.params("vphi");
		maxiter_hour = cal.size("maxiter_hour");
		tols_hour = cal.params("tols_hour");
		hour_high = cal.params("hour_high");
		maxiter_d = cal.size("maxiter_d");
		tols_d = cal.params("tols_d");
		dmax = cal.params("dmax");
		solve_both_binding_pols = TA::KAPLAN::solve_both_binding_pols;
		solve_unbinding_cons = TA::KAPLAN::solve_unbinding_cons;
		solve_unbinding_res = TA::KAPLAN::solve_unbinding_res;
		solve_binding_pols = TA::KAPLAN::solve_binding_pols;
		solve_rhs = TA::KAPLAN::solve_rhs;

		swap_vecs = TA::swap_het_vecs;
		solve_res_pols = TA::solve_res_pols;
		solve_cdf = TA::solve_cdf;
		target_htm = TA::target_HtM;
		solve_dist_stat = TA::solve_dist_stat;
		solve_res_aggregators = TA::KAPLAN::solve_res_aggregators;
		swap_dist = TA::swap_het_dist;
		map_pols = TA::convert_pols_map;

		dmin = -akinkgrid * pow((1 - chi0) / chi1, 1 / (chi2 - 1));
		ra_max = pow((1 - chi0) / chi1, 1 / (chi2 - 1));

		skill_dist_extend = skill_dist;
		break;
	default:
		break;
	}
	dainv_diag.resize(na + 1);
	dainv = dainv_diag.segment(1, na - 1) = da.inverse();
	dainv_diag(0) = dainv_diag(na) = 0;

	dbinv_diag.resize(nb + 1);
	dbinv = dbinv_diag.segment(1, nb - 1) = db.inverse();
	dbinv_diag(0) = dbinv_diag(nb) = 0;

	dagridinv = dagrid.inverse();
	dbgridinv = dbgrid.inverse();

	idnesk.setIdentity(nesk, nesk);
	idnab.resize(nab, nab);
	idnab.setIdentity();
	markov_diag.setZero(nesk, nesk);
	diagHJB.resize(nesk);
	diagKFE.resize(nesk);
	diagFeyman.resize(nesk);
	markov_diag_spmat.resize(nesk);
	death_alloc.setZero(nab, nab);
	death_alloc.row(na * nb_neg).setConstant(deathrate * DeltaKFE);

	H_min = -1e12;

	if (FRACbNeg)
	{
		rb_wedge = cal.params("rb_wedge");
	}
}
void
Het_Inputs::set_extensive_mat()
{


	markov_diag.diagonal() = markov.diagonal();
	markov_offdiag = markov - markov_diag;
#pragma omp parallel for
	for (int r = 0; r < nesk; ++r)
	{
		markov_diag_spmat[r] = idnab * markov_diag(r, r);
	}
}
void
Het_Inputs::set_diags()
{
	offdiagKFE = idnesk + DeltaKFE * markov_offdiag;
	offdiagHJB = idnesk / DeltaHJB + markov_offdiag;

#pragma omp parallel for
	for (int r = 0; r < nesk; ++r)
	{
		diagHJB[r] = idnab / DeltaHJB + (rho + deathrate - markov_diag(r, r)) * idnab;
		diagKFE[r] = idnab + (deathrate - markov_diag(r, r)) * idnab * DeltaKFE;
		diagFeyman[r] = idnab - markov_diag(r, r) * idnab * DeltaFeyman;
	}
}

void
Het_Inputs::set_KFEdiags()
{
	offdiagKFE = idnesk + DeltaKFE * markov_offdiag;
#pragma omp parallel for
	for (int r = 0; r < nesk; ++r) diagKFE[r] = idnab + (deathrate - markov_diag(r, r)) * idnab * DeltaKFE;
}

void
Het_Inputs::set_Feymandiags()
{
#pragma omp parallel for
	for (int r = 0; r < nesk; ++r) diagFeyman[r] = idnab - markov_diag(r, r) * idnab * DeltaFeyman;
}

void
Het_Inputs::set_HJBdiags()
{
	offdiagHJB = idnesk / DeltaHJB + markov_offdiag;
#pragma omp parallel for
	for (int r = 0; r < nesk; ++r)
	{
		diagHJB[r] = idnab / DeltaHJB + (rho + deathrate - markov_diag(r, r)) * idnab;
	}
}



void
Het_Inputs::set_labor_income()
{
	after_tax_wage = (1.0 - tax) * wage * sgrid;
	switch (model)
	{
	case Model::KAPLAN:
		inc_labor = (1 - tax) * (1.0 - profdistfrac) * profit * sgrid / meanlabeff;
		break;
	default:
		break;
	}

	inc_labor += transfer;

	switch (hour_supply)
	{
	case Hour_supply::GHH:
		hour_ghh = (after_tax_wage / vphi).pow(frisch).min(hour_high);
		disutility_hour_ghh = disutil_hour(*this, hour_ghh);
		break;
	default:
		break;
	}
}

void
Het_Inputs::set_interest()
{
	if (FRACbNeg)
	{
		rb_borrow = rb + rb_wedge;
		rb_structure = rb + (bgrid < 0).cast<double>() * rb_wedge;

		if (bgrid(0, 0) < -transfer / (rb_borrow + deathrate))
		{
			std::cerr << "Natural borrowing limit violated!" << std::endl;
			std::exit(0);
		}
	}
	else
	{
		rb_structure = het2::Constant(na, nb, rb);
	}
}

void
Het_Inputs::set_income()
{
	switch (model)
	{
	case Model::KAPLAN:
		//a_drift = (ra + deathrate) * agrid * (1 - agrid.pow(14) * std::pow(agrid(na - 1, 0) * 0.999, -14));
		a_drift = (ra + deathrate) * agrid;
		b_drift = (rb_structure + deathrate) * bgrid;
		break;
	default:
		break;
	}
#pragma omp parallel for
	for (int r = 0; r < nesk; ++r)
		inc[r] = inc_labor[r] + b_drift;
}

void
Het_Inputs::set_dag_inputs()
{
	// set het inputs by target value, used for steady state. 
	rb = cal.params("rb");
	wage = cal.params("targetwage");
	rb_borrow = rb + rb_wedge;
	tax = cal.params("tax");
	transfer = cal.params("transfer");
	switch (model)
	{
	case Model::KAPLAN:
		ra = cal.params("targetra");
		break;
	default:
		break;
	}

}
void
Het_Inputs::set_hh()
{
	set_dag_inputs();
	set_interest();
	set_labor_income();
	set_income();
	set_extensive_mat();
	set_diags();
}


void
Het_policies::init()
{
	init_empty_tensor(c, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(cem, het_inputs.na, het_inputs.nb, het_inputs.nsk);
	init_empty_tensor(cb, 1, het_inputs.nb, 1);
	init_empty_tensor(cb_em, 1, het_inputs.nb, 1);
	init_empty_tensor(d, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(hour, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(labor, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(adj, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sa, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sb, het_inputs.na, het_inputs.nb, het_inputs.nesk);
}

void
Het_workspace::init()
{
	switch (model)
	{
	case Model::KAPLAN:
		init_empty_tensor(df, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		init_empty_tensor(fhour_high, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		init_empty_tensor(fhour_low, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		init_empty_tensor(hour_low, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		init_empty_tensor(hour_high, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		init_empty_tensor(hour_min, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		init_empty_tensor(fhour_min, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		init_empty_tensor(fhour_max, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		init_empty_tensor(fhour, het_inputs.na, het_inputs.nb, het_inputs.nesk);
		break;
	default:
		break;
	}

	next.setZero(het_inputs.nab, het_inputs.nesk);
	rhs.setZero(het_inputs.nab, het_inputs.nesk);

	dist.setZero(het_inputs.nab, het_inputs.nesk);
	dist_init.setZero(het_inputs.nab, het_inputs.nesk);


	subeff1ass.setZero(het_inputs.nab, het_inputs.nesk);
	subeff2ass.setZero(het_inputs.nab, het_inputs.nesk);
	wealtheff1ass.setZero(het_inputs.nab, het_inputs.nesk);
	wealtheff2ass.setZero(het_inputs.nab, het_inputs.nesk);

	init_empty_tensor(adj1, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(effdisc, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	muc.setZero(het_inputs.nab, het_inputs.nesk);
	badj.setZero(het_inputs.nab, het_inputs.nesk);
	init_empty_tensor(mpc, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(mpd_a, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(mpd_b, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(VaB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(VaB, het_inputs.na, het_inputs.nb, het_inputs.nesk);


	ccum1.setZero(het_inputs.nab, het_inputs.nesk);
	ccum2.setZero(het_inputs.nab, het_inputs.nesk);
	ccum4.setZero(het_inputs.nab, het_inputs.nesk);

	


	bdist.setZero(het_inputs.nb);
	bcdf.setZero(het_inputs.nb);
	adist.setZero(het_inputs.na);
	acdf.setZero(het_inputs.na);
	abdist.setZero(het_inputs.na, het_inputs.nb);
	
	illiquid_wealth_dist.setZero(het_inputs.na);
	liquid_wealth_dist.setZero(het_inputs.nb);

	//transition mat
	init_empty_tensor(assetHJB, het_inputs.nab, het_inputs.nab, het_inputs.nesk);
	init_empty_tensor(zeros, het_inputs.nab, het_inputs.nab, het_inputs.nesk);
	init_empty_tensor(distKFE, het_inputs.nab, het_inputs.nab, het_inputs.nesk);
	init_empty_tensor(FeynmanKac, het_inputs.nab, het_inputs.nab, het_inputs.nesk);
	init_empty_tensor(mpcHJB, het_inputs.nab, het_inputs.nab, het_inputs.nesk);
	init_empty_tensor(diagdiscmpc, het_inputs.nab, het_inputs.nab, het_inputs.nesk);

	solvers.resize(het_inputs.nesk);
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		solvers[r] = new Eigen::PardisoLU<SpMat>();
		solvers[r]->pardisoParameterArray()[59] = 1;
	}
	//value fun
	V_init.setZero(het_inputs.nab, het_inputs.nesk);
	V.setZero(het_inputs.nab, het_inputs.nesk);



	/*Upwinds*/
	init_empty_tensor(VaB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(VaF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(VbB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(VbF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(cF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(cB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(c0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(hF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(hB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(h0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(scF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(scB, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	init_empty_tensor(sdFB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sdFF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sdBB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sdBF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sd0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sdtemp, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	init_empty_tensor(FdB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(FdF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(FdminB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(FdminF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(Fd0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(Fdtemp, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	init_empty_tensor(sbFB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sbFF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sbBB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sbBF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sb0F, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sb0B, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	init_empty_tensor(saFB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(saFF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(saBB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(saBF, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	init_empty_tensor(dBB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(dBF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(dFB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(dFF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(dF0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(dB0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(cF0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(cB0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(hourF0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(hourB0, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	init_empty_tensor(utilityF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(utilityB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(utility0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	//Halmilton
	init_empty_tensor(HFB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(HFF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(HBB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(HBF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(H0F, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(H0B, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(HF0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(HB0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(H00, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(H0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(c00, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(hour00, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	//Policies
	init_empty_tensor(c, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(d, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sb, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sa, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sd, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(adriftF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(adriftB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(bdriftF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(bdriftB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(hour, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(labor, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	init_empty_tensor(adj, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	
	init_empty_tensor(c0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(sa0, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(d0, het_inputs.na, het_inputs.nb, het_inputs.nesk);


	init_empty_tensor(d_high, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(d_low, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(Fd_low, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	//Ind
	init_empty_tensor(indFB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(indFF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(indBB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(indBF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(ind0F, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(ind0B, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(ind0, het_inputs.na, het_inputs.nb, het_inputs.nesk);

	init_empty_tensor(validFB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(validFF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(validBB, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(validBF, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(valid0F, het_inputs.na, het_inputs.nb, het_inputs.nesk);
	init_empty_tensor(valid0B, het_inputs.na, het_inputs.nb, het_inputs.nesk);
}

void
Het_workspace::delete_solvers()
{
	for (int r = 0; r < het_inputs.nesk; ++r) delete solvers[r];
}


void
Het_workspace::init_dist()
{
	int r = 0;
	int ib0_borr = 0, ib0_noborr = 0;
	if (het_inputs.deathrate < 1e-8)
	{
		ib0_borr = het_inputs.nb_neg + 1;
		ib0_noborr = 1;
	}
	else
	{
		ib0_borr = het_inputs.nb_neg;
		ib0_noborr = 0;
	}
#pragma omp parallel for
	for (r = 0; r < het_inputs.nesk; ++r)
	{
		if (!het_inputs.FRACbNeg)
		{
			dist_init(ib0_noborr, r) = het_inputs.skill_dist_extend(r);
		}
		else
		{
			dist_init(het_inputs.na * ib0_borr, r) = het_inputs.skill_dist_extend(r);
		}
		dist_init.col(r) /= dist_init.col(r).sum() / het_inputs.skill_dist_extend(r);
	}
}

void
Het_workspace::init_bellman()
{

	int r = 0;
	int j = 0;
	switch (model)
	{
	case Model::KAPLAN:
#pragma omp parallel for
		for (r = 0; r < het_inputs.nesk; ++r)
		{
			switch (het_inputs.hour_supply)
			{
			case Hour_supply::Seq:
				V_init.col(r) = (util_cons(het_inputs, het_inputs.inc[r] + het_inputs.after_tax_wage[r] / 3)
					+ disutil_hour(het_inputs, 1.0 / 3.0)).reshaped().matrix()
					/ (het_inputs.deathrate + het_inputs.rho);
				break;
			case Hour_supply::GHH:
				V_init.col(r) = util_cons(het_inputs, het_inputs.inc[r] + het_inputs.after_tax_wage[r] * het_inputs.hour_ghh(r))
					.reshaped().matrix()
					/ (het_inputs.deathrate + het_inputs.rho);
				break;
			case Hour_supply::NoSupply:
				V_init.col(r) = util_cons(het_inputs, het_inputs.inc[r] + het_inputs.after_tax_wage[r])
					.reshaped().matrix()
					/ (het_inputs.deathrate + het_inputs.rho);
				break;
			default:
				break;
			}
		}
		break;
	default:
		break;
	}

}





void
init_map_tensor_like(std::unordered_map<std::string, Tensor>& tens, const std::unordered_map<std::string, Tensor>& tar)
{
	int dim2 = tar.begin()->second.size();
	int dim0 = tar.begin()->second[0].rows();
	int dim1 = tar.begin()->second[0].cols();
	for (auto& elem : tar)
	{
		init_empty_tensor(tens[elem.first], dim0, dim1, dim2);
	}
}

double
util_cons(const Het_Inputs& het_inputs, double c)
{
	return ((std::abs(het_inputs.sigma - 1.0) < 1e-12) ? std::log(c) : std::pow(c, 1 - het_inputs.sigma) / (1 - het_inputs.sigma));
}



double
disutil_hour(const Het_Inputs& het_inputs, double hour)
{
	return -het_inputs.vphi * std::pow(hour, 1 + 1 / het_inputs.frisch) / (1 + 1 / het_inputs.frisch);
}

double
utility1(const Het_Inputs& het_inputs, double c)
{
	return std::pow(c, -het_inputs.sigma);
}

double
utility1inv(const Het_Inputs& het_inputs, double Vb)
{
	return std::pow(Vb, -1 / het_inputs.sigma);
}

double
utility2inv(const Het_Inputs& het_inputs, double Vh)
{
	return std::pow(Vh / het_inputs.vphi, het_inputs.frisch);
}

double
utility2(const Het_Inputs& het_inputs, double hour)
{
	return -het_inputs.vphi * std::pow(hour, het_inputs.frisch);
}

void
construct_cumHJB(const Het_Inputs& het_inputs, Het_workspace& ws)
{
#pragma omp parallel for
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		ws.FeynmanKac[r] = het_inputs.diagFeyman[r] - het_inputs.DeltaFeyman * ws.assetHJB[r];
		ws.solvers[r]->compute(ws.FeynmanKac[r]);
		if (ws.solvers[r]->info() != Eigen::Success) {
			std::cerr << "FeynmanKac decomposition failed!" << std::endl;
			exit(0);
		}
	}
}

void
solve_cum(const Het_Inputs& het_inputs, Het_workspace& ws)
{
	construct_cumHJB(het_inputs, ws);
	solve_cum(het_inputs, ws.c, ws, 1 / het_inputs.DeltaFeyman, ws.ccum1);
	solve_cum(het_inputs, ws.c, ws, 2 / het_inputs.DeltaFeyman, ws.ccum2);
	solve_cum(het_inputs, ws.c, ws, 4 / het_inputs.DeltaFeyman, ws.ccum4);

}

void
solve_cum(const Het_Inputs& het_inputs, const Tensor& pols, Het_workspace& ws, int period, Eigen::MatrixXd& cum)
{
	cum.setZero();
	int r = 0;
	for (int t = 0; t < period; ++t)
	{
		solve_cumHJB(het_inputs, pols, ws, cum);
		std::swap(ws.next, cum);
	}
}

void
solve_cumHJB(const Het_Inputs& het_inputs, const Tensor& pols, Het_workspace& ws, Eigen::MatrixXd& cum)
{
#pragma omp parallel for
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		ws.rhs.col(r) = cum.col(r) + het_inputs.DeltaFeyman * pols[r].reshaped().matrix()
			+ het_inputs.DeltaFeyman * cum * het_inputs.markov_offdiag.row(r).transpose();
		ws.next.col(r) = ws.solvers[r]->solve(ws.rhs.col(r));
	}
}

void
solve_mpcs(const Het_Inputs& het_inputs, Het_workspace& ws)
{
#pragma omp parallel for
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		ws.mpc[r].block(0, 0, het_inputs.na, het_inputs.nb - 1)
			= (ws.c[r].block(0, 1, het_inputs.na, het_inputs.nb - 1) - ws.c[r].block(0, 0, het_inputs.na, het_inputs.nb - 1))
			* het_inputs.dbgridinv;

		ws.mpc[r].col(het_inputs.nb - 1)
			= ws.mpc[r].col(het_inputs.nb - 2);

		ws.mpd_b[r].block(0, 0, het_inputs.na, het_inputs.nb - 1)
			= (ws.d[r].block(0, 1, het_inputs.na, het_inputs.nb - 1) - ws.d[r].block(0, 0, het_inputs.na, het_inputs.nb - 1))
			* het_inputs.dbgridinv;

		ws.mpd_b[r].col(het_inputs.nb - 1)
			= ws.mpd_b[r].col(het_inputs.nb - 2);

		ws.mpd_a[r].block(0, 0, het_inputs.na - 1, het_inputs.nb)
			= (ws.d[r].block(1, 0, het_inputs.na - 1, het_inputs.nb) - ws.d[r].block(0, 0, het_inputs.na - 1, het_inputs.nb))
			* het_inputs.dagridinv;

		ws.mpd_a[r].row(het_inputs.na - 1)
			= ws.mpd_a[r].row(het_inputs.na - 2);
	}
}

void
prepare_decomposition(const Het_Inputs& het_inputs, Het_workspace& ws)
{
#pragma omp parallel for
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		switch (het_inputs.adj_fun)
		{
		case Adj_fun::KAPLAN_ADJ:
			ws.adj1[r] = TA::Adj_fun_kaplan::adj1(het_inputs, ws.d[r]);
			break;
		case Adj_fun::AUCLERT_ADJ:
			ws.adj1[r] = TA::Adj_fun_auclert::adj1(het_inputs, ws.d[r]);
			break;
		default:
			break;
		}
		ws.effdisc[r] = ws.mpc[r] + (1.0 + ws.adj1[r]) * ws.mpd_b[r] - ws.mpd_a[r];

		switch (het_inputs.hour_supply)
		{
		case Hour_supply::GHH:
			if (r < het_inputs.nsk) ws.muc.col(r) = utility1(het_inputs, ws.c[r] + het_inputs.disutility_hour_ghh(r)).reshaped();
			else ws.muc.col(r) = utility1(het_inputs, ws.c[r]).reshaped();
			break;
		default:
			ws.muc.col(r) = utility1(het_inputs, ws.c[r]).reshaped();
			break;
		}

		ws.badj.col(r).array() = (het_inputs.bgrid.reshaped() * (ws.mpc[r].reshaped() / ws.c[r].reshaped())) * ws.muc.col(r).array();
	}
}

void
construct_diag_dismpc(const Het_Inputs& het_inputs, const Tensor& pols, double dig,
	Het_workspace& ws)
{
#pragma omp parallel for
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		ws.diagdiscmpc[r].setZero();
		ws.diagdiscmpc[r].reserve(Eigen::VectorXi::Constant(het_inputs.nab, 1));
		int k = 0, m = 0, i = 0;
		for (m = 0; m < het_inputs.nb; ++m)
		{
			for (k = 0; k < het_inputs.na; ++k)
			{
				i = het_inputs.na * m + k;
				ws.diagdiscmpc[r].insert(i, i) 
					= het_inputs.DeltaDecomp * dig + het_inputs.DeltaDecomp * pols[r](k, m);

			}
		}
		ws.diagdiscmpc[r].makeCompressed();
	}

}

void
solve_ss_decomp(const Het_Inputs& het_inputs, Het_workspace& ws)
{
	solve_mpcs(het_inputs, ws);
	prepare_decomposition(het_inputs, ws);

	ws.decomp_diag = -std::log(het_inputs.rho_zeta) + het_inputs.rho - het_inputs.rb;
	// DECOMP EFFECT: ONE ASSET
	construct_diag_dismpc(het_inputs, ws.mpc, ws.decomp_diag, ws);

	init_decomposition(het_inputs, ws.muc, ws, ws.subeff1ass);
	init_decomposition(het_inputs, ws.badj, ws, ws.wealtheff1ass);

	construct_decompHJB(het_inputs, ws);
	solve_convergent_decomp(het_inputs, ws.muc, ws, ws.subeff1ass);
	solve_convergent_decomp(het_inputs,  ws.badj, ws, ws.wealtheff1ass);

	// DECOMP EFFECT: TWO ASSET
	construct_diag_dismpc(het_inputs, ws.effdisc, ws.decomp_diag, ws);

	init_decomposition(het_inputs, ws.muc, ws, ws.subeff2ass);
	init_decomposition(het_inputs, ws.badj, ws, ws.wealtheff2ass);

	construct_decompHJB(het_inputs, ws);
	solve_convergent_decomp(het_inputs, ws.muc, ws, ws.subeff2ass);
	solve_convergent_decomp(het_inputs, ws.badj, ws, ws.wealtheff2ass);


}

void
construct_decompHJB(const Het_Inputs& het_inputs, Het_workspace& ws)
{
#pragma omp parallel for
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		ws.diagdiscmpc[r] -= het_inputs.DeltaDecomp * het_inputs.markov_diag_spmat[r];
		ws.mpcHJB[r] = ws.diagdiscmpc[r] - het_inputs.DeltaDecomp * ws.assetHJB[r];
		ws.solvers[r]->compute(ws.mpcHJB[r]);
		if (ws.solvers[r]->info() != Eigen::Success) {
			std::cerr << "FeynmanKac decomposition failed!" << std::endl;
			exit(0);
		}
	}
}

void
init_decomposition(const Het_Inputs& het_inputs, const Eigen::MatrixXd& rhs, Het_workspace& ws, Eigen::MatrixXd& decomp)
{
#pragma omp parallel for
	for (int r = 0; r < het_inputs.nesk; ++r)
	{
		ws.mpcHJB[r] = ws.diagdiscmpc[r] - het_inputs.DeltaDecomp * ws.assetHJB[r];
		ws.solvers[r]->compute(ws.mpcHJB[r]);
		if (ws.solvers[r]->info() != Eigen::Success) {
			std::cerr << "FeynmanKac decomposition failed!" << std::endl;
			exit(0);
		}
		decomp.col(r) = ws.solvers[r]->solve(rhs.col(r));
	}
}

void
solve_convergent_decomp(const Het_Inputs& het_inputs, const Eigen::MatrixXd& rhs, Het_workspace& ws, Eigen::MatrixXd& decomp)
{
	// iterate
	for (ws.iter = 0; ws.iter < het_inputs.maxiter_discmpc; ++ws.iter)
	{
		solve_decompHJB(het_inputs, rhs, decomp, ws);
		if (ws.iter % 10 == 0)
		{
			if (check_within_tols(decomp, ws.next, het_inputs.tols_discmpc, ws.eps))
			{
				goto end;
			}
		}
		std::swap(decomp, ws.next);
	}
	cout << "Couldn't find stationary mpcs after " << het_inputs.maxiter_discmpc << " iterations!!" << endl;
	exit(0);
end:
	ws.iter = 0;
}

