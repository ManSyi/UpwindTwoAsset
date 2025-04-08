/* These algorithms are modifications and based on the software NEWUOA, authored by M. J. D. Powell,
   to minimize sum of squares by taking advantage of the problem structure.
	  min_{x \in R^n }  F(x) := Sum_{i=1}^{mv}  v_err_i(x)^2
   where v_err(x) : R^n \to R^{mv} is a vector function.
   This subroutine seeks the least value of sum of the squares of the components of v_err(x)
   by combing trust region method and Levenberg-Marquardt method

   References:

   1.  M. J. D. Powell, The NEWUOA software for unconstrained optimization without derivatives,
	   DAMTP 2004/ NA 05
   2.  H. Zhang, A. R. CONN, AND K. SCHEINBERG, A DERIVATIVE-FREE ALGORITHM FOR THE LEAST-SQUARE
	   MINIMIZATION, technical report, 2009

	  ------------------------------------------------------------------
	  | This program is free software; you can redistribute it and/or  |
	  |modify it under the terms of the GNU General Public License as  |
	  |published by the Free Software Foundation; either version 2 of  |
	  |the License, or (at your option) any later version.             |
	  |This program is distributed in the hope that it will be useful, |
	  |but WITHOUT ANY WARRANTY; without even the implied warranty of  |
	  |MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the   |
	  |GNU General Public License for more details.                    |
	  |                                                                |
	  |You should have received a copy of the GNU General Public       |
	  |License along with this program; if not, write to the Free      |
	  |Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, |
	  |MA  02110-1301  USA                                             |
	  -----------------------------------------------------------------| */

//#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
//#include "stdafx.h"
//#endif
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "newuoa_h.h"
#include "Support.h"
	  /* Macros to deal with single/double precision. */
#undef REAL
#ifdef SINGLE_PRECISION
# define REAL      float
# define ABS(x)    fabsf(x)
# define SQRT(x)   sqrtf(x)
# define SIN(x)    sinf(x)
# define COS(x)    cosf(x)
#else
# define REAL      double
# define ABS(x)    fabs(x)
# define SQRT(x)   sqrt(x)
# define SIN(x)    sin(x)
# define COS(x)    cos(x)
#endif

/* Macros yielding the min/max of two values. */
#define MAX(a,b) ((a) >= (b) ? (a) : (b))
#define MIN(a,b) ((a) <= (b) ? (a) : (b))

/* Constants */
static const REAL one = 1.0, half = 0.5, tenth = 0.1, zero = 0.0, twopi = 2.0 * M_PI;
static const LOGICAL debug = 0;

/*---------------------------------------------------------------------------*/
/* NEWUOA_H PRIVATE FUNCTIONS */

static REAL f_value(const INTEGER mv, const REAL* v_err)
{
	INTEGER m1;
	REAL f = 0.0;
	for (m1 = 0; m1 < mv; m1++)
		f += v_err[m1] * v_err[m1];
	return f;
} /* f_value */

static void f_grad(const INTEGER mv, const REAL* v_base, REAL* v_gtemp)
{
	INTEGER m1;
	for (m1 = 0; m1 < mv; m1++)
		v_gtemp[m1] = 2 * v_base[m1];
} /* f_grad */

/* SYMV computes a matrix-vector product defined as y = A * x,
   where x and y are n-element vectors and A is an n-by-n symmetric matrix. */
static void symv(const INTEGER n, const REAL alpha, const REAL* a,
	const INTEGER lda, const REAL* x, const INTEGER incx,
	const REAL beta, REAL* y, const INTEGER incy) {
	INTEGER i, ia, j;
	for (i = 0; i < n; ++i)
		y[i * incy] *= beta;
	ia = 0;
	for (j = 0; j < n; ++j)
		for (i = 0; i <= j; ++i) {
			if (i < j) y[j * incy] += alpha * a[ia * lda] * x[i * incx];
			y[i * incy] += alpha * a[ia * lda] * x[j * incx];
			++ia;
		}
}

static int
trsapp_h(const INTEGER n, const INTEGER npt, REAL* xopt,
	REAL* xpt, REAL* gq, REAL* hq, REAL* pq,
	const REAL delta, REAL* step, REAL* d, REAL* g,
	REAL* hd, REAL* hs, REAL* gbeg, REAL* v_gtemp,
	REAL* crvmin, REAL* gqv, REAL* hqv, REAL* pqv,
	REAL* xbase, REAL* vquad, REAL* gqv_opt,
	REAL* v_opt, REAL* v_base,
	const REAL xoptsq, const INTEGER mv,
	LOGICAL* model_update, LOGICAL* opt_update)
{
	/*	N is the number of variables of a quadratic objective function, Q say.
		The arguments NPT, XOPT, XPT, GQ, HQ and PQ have their usual meanings, in
		order to define the current quadratic model Q.

		DELTA is the trust region radius, and has to be positive.

		STEP will be set to the calculated trial step.  The arrays D, G, HD and HS
		will be used for working space.

		CRVMIN will be set to the least curvature of H along the conjugate
		directions that occur, except that it is set to zero if STEP goes all the
		way to the trust region boundary.

		The calculation of STEP begins with the truncated conjugate gradient
		method. If the boundary of the trust region is reached, then further
		changes to STEP may be made, each one being in the 2D space spanned by the
		current STEP and the corresponding gradient of Q. Thus STEP should provide
		a substantial reduction to Q within the trust region. */

		/* Constants */
	const REAL delsq = delta * delta;

	/* Local variables */
	REAL alpha, angle, angtest, bstep, cf, cth, dd, dg, dhd, dhs, ds,
		f_base, f_opt, gg, ggbeg, ggsav, gnorm2, qadd, qbeg, qmin,
		qnew, qred, qsav, ratio, reduc, sg, sgk, shs, ss, sth, t1, t2,
		temp, tempa, tempb;
	INTEGER i, ih, isave, iterc, itermax, itersw, iu, j, k, m1;
	LOGICAL zero_res;

	/* Parameter adjustments */
	xpt -= npt + 1;
	--xopt;
	--gq;
	--hq;
	--pq;
	--step;
	--d;
	--g;
	--hd;
	--hs;
	gqv -= mv + 1;
	hqv -= mv + 1;
	pqv -= mv + 1;
	--xbase;
	gqv_opt -= mv + 1;
	--v_opt;
	--v_base;

	/* Set uninitialized variables */
	tempa = zero;
	tempb = zero;
	shs = zero;
	sg = zero;

	if (!(*model_update) && !(*opt_update)) goto L8;
	*model_update = 0;
	*opt_update = 0;
	if (SQRT(xoptsq) > delta * .25) {

		/* Use the gradient at xopt to formulate J^t J */
		for (m1 = 1; m1 <= mv; ++m1) {
			for (i = 1; i <= n; ++i)
				gqv_opt[m1 + i * mv] = gqv[m1 + i * mv];
			for (k = 1; k <= npt; ++k) {
				temp = zero;
				for (j = 1; j <= n; ++j)
					temp += xpt[k + j * npt] * xopt[j];
				temp *= pqv[m1 + k * mv];
				for (i = 1; i <= n; ++i)
					gqv_opt[m1 + i * mv] += temp * xpt[k + i * npt];
			}
			symv(n, one, &hqv[m1 + mv], mv, &xopt[1], 1, one, &gqv_opt[m1 + mv], mv);
		}
		f_grad(mv, &v_opt[1], &v_gtemp[1]);
		gnorm2 = zero;
		for (i = 1; i <= n; ++i) {
			gq[i] = zero;
			for (m1 = 1; m1 <= mv; ++m1)
				gq[i] += v_gtemp[m1] * gqv_opt[m1 + i * mv];
			gnorm2 += gq[i] * gq[i];
		}

		/* Calculate the explicite Hessian. */
		f_opt = f_value(mv, &v_opt[1]);
		if (gnorm2 >= 1. || f_opt <= SQRT(gnorm2)) zero_res = 1;
		else zero_res = 0;
		ih = 0;
		for (j = 1; j <= n; ++j) {
			for (i = 1; i <= j; ++i) {
				++ih;
				if (zero_res) {
					t1 = zero;
					for (m1 = 1; m1 <= mv; ++m1)
						t1 += gqv_opt[m1 + i * mv] * gqv_opt[m1 + j * mv];
					hq[ih] = t1 * 2.;
				}
				else {
					t1 = zero;
					for (m1 = 1; m1 <= mv; ++m1) {
						t2 = zero;
						for (k = 1; k <= npt; ++k)
							t2 += xpt[k + i * npt] * pqv[m1 + k * mv] * xpt[k + j * npt];
						t2 += hqv[m1 + ih * mv];
						t1 += gqv_opt[m1 + i * mv] * gqv_opt[m1 + j * mv] + v_opt[m1] * t2;
					}
					hq[ih] = t1 * 2.;
				}
			}
		}
	}
	else {

		/* Use the gradient at xbase to formulate J^t J */
		f_grad(mv, &v_base[1], &v_gtemp[1]);
		gnorm2 = zero;
		for (i = 1; i <= n; ++i) {
			gq[i] = zero;
			for (m1 = 1; m1 <= mv; ++m1)
				gq[i] += v_gtemp[m1] * gqv[m1 + i * mv];
			gnorm2 += gq[i] * gq[i];
		}

		/* Calculate the explicite Hessian. */
		f_base = f_value(mv, &v_base[1]);
		if (gnorm2 >= 1. || f_base <= SQRT(gnorm2)) zero_res = 1;
		else zero_res = 0;
		ih = 0;
		for (j = 1; j <= n; ++j)
			for (i = 1; i <= j; ++i) {
				++ih;
				if (zero_res) {
					t1 = zero;
					for (m1 = 1; m1 <= mv; ++m1)
						t1 += gqv[m1 + i * mv] * gqv[m1 + j * mv];
					hq[ih] = t1 * 2.;
				}
				else {
					t1 = zero;
					for (m1 = 1; m1 <= mv; ++m1) {
						t2 = zero;
						for (k = 1; k <= npt; ++k)
							t2 += xpt[k + i * npt] * pqv[m1 + k * mv] * xpt[k + j * npt];
						t2 += hqv[m1 + ih * mv];
						t1 += gqv[m1 + i * mv] * gqv[m1 + j * mv] + v_base[m1] * t2;
					}
					hq[ih] = t1 * 2.;
				}
			}

		/* calculte the gradient at xopt */
		symv(n, one, &hq[1], 1, &xopt[1], 1, one, &gq[1], 1);
	}
L8:
	iterc = 0;
	itermax = n;
	itersw = itermax;
	if (debug) {
		REAL t = zero;
		for (i = 1; i <= n; ++i)
			t += xopt[i] * xopt[i];
		fprintf(stdout, " ||xopt||=%25.15E\n", (double)SQRT(t));
	}
	gnorm2 = zero;
	for (i = 1; i <= n; ++i) {
		gnorm2 += gq[i] * gq[i];
		d[i] = zero;
		hd[i] = zero;
	}
	gnorm2 = SQRT(gnorm2);
	if (debug) fprintf(stdout, " gnorm2=%25.15E\n", (double)gnorm2);

	/* Prepare for the first line search. */
	qred = zero;
	dd = zero;
	for (i = 1; i <= n; ++i) {
		step[i] = zero;
		hs[i] = zero;
		g[i] = gq[i];
		d[i] = -g[i];
		gbeg[i - 1] = g[i];
		dd += d[i] * d[i];
	}
	*crvmin = zero;
	if (dd == zero) goto L160;
	ds = zero;
	ss = zero;
	gg = dd;
	ggbeg = gg;
	if (debug) fprintf(stdout, " GGBEG=%25.15E\n", ggbeg);

	/* Calculate the step to the trust region boundary and the product HD.
	   Set the vector HD to the vector D multiplied by the second derivative matrix of Q. */
L40:
	++iterc;
	temp = delsq - ss;
	bstep = temp / (ds + SQRT(ds * ds + dd * temp));
	// bstep = (-ds + SQRT(ds * ds + dd * temp))/dd;
	if (debug) fprintf(stdout, " BSTEP=%25.15E\n", bstep);
	symv(n, one, &hq[1], 1, &d[1], 1, zero, &hd[1], 1);
	if (iterc > itersw) goto L120;
	dhd = zero;
	for (j = 1; j <= n; ++j)
		dhd += d[j] * hd[j];

	/* Update CRVMIN and set the step-length ALPHA. */
	alpha = bstep;
	if (debug) fprintf(stdout, " ITERC=%6ld\n DHD/DD=%25.15E\n", (long)iterc, (double)(dhd / dd));
	if (dhd > zero) {
		temp = dhd / dd;
		if (iterc == 1) *crvmin = temp;
		*crvmin = MIN(*crvmin, temp);
		alpha = MIN(alpha, gg / dhd);
	}
	qadd = alpha * (gg - half * alpha * dhd);
	qred += qadd;

	/* Update STEP and HS. */
	ggsav = gg;
	gg = zero;
	for (i = 1; i <= n; ++i) {
		step[i] += alpha * d[i];
		hs[i] += alpha * hd[i];
		temp = g[i] + hs[i];
		gg += temp * temp;
	}
	if (debug) fprintf(stdout, " GG=%25.15E\n", (double)gg);
	if (gg <= MIN(1e-4 * ggbeg, 1e-16)) goto L160;
	if (gg <= 1e-14 * gnorm2) goto L160;
	if (iterc == itermax) goto L160;

	/* Begin another conjugate direction iteration if required. */
	if (alpha < bstep) {
		if (qadd <= qred * 1e-6) goto L160;
		temp = gg / ggsav;
		dd = zero;
		ds = zero;
		ss = zero;
		for (i = 1; i <= n; ++i) {
			d[i] = temp * d[i] - g[i] - hs[i];
			dd += d[i] * d[i];
			ds += d[i] * step[i];
			ss += step[i] * step[i];
		}
		if (ss < delsq) goto L40;
	}
	*crvmin = zero;
	itersw = iterc;

	/* Test whether an alternative iteration is required. */
L90:
	if (gg <= ggbeg * 1e-4) goto L160;
	if (debug) fprintf(stdout, "curve search performed\n");
	sg = zero;
	shs = zero;
	for (i = 1; i <= n; ++i) {
		sg += step[i] * g[i];
		shs += step[i] * hs[i];
	}
	sgk = sg + shs;
	angtest = sgk / SQRT(gg * delsq);
	if (angtest <= -.99) goto L160;

	/* Begin the alternative iteration by calculating D and HD and some
	   scalar products. */
	++iterc;
	temp = SQRT(delsq * gg - sgk * sgk);
	tempa = delsq / temp;
	tempb = sgk / temp;
	for (i = 1; i <= n; ++i)
		d[i] = tempa * (g[i] + hs[i]) - tempb * step[i];
	symv(n, one, &hq[1], 1, &d[1], 1, zero, &hd[1], 1);
L120:
	dg = zero;
	dhd = zero;
	dhs = zero;
	for (i = 1; i <= n; ++i) {
		dg += d[i] * g[i];
		dhd += hd[i] * d[i];
		dhs += hd[i] * step[i];
	}

	/* Seek the value of the angle that minimizes Q. */
	cf = half * (shs - dhd);
	qbeg = sg + cf;
	qsav = qbeg;
	qmin = qbeg;
	isave = 0;
	iu = 49;
	temp = twopi / (REAL)(iu + 1);
	for (i = 1; i <= iu; ++i) {
		angle = (REAL)i * temp;
		cth = COS(angle);
		sth = SIN(angle);
		qnew = (sg + cf * cth) * cth + (dg + dhs * cth) * sth;
		if (qnew < qmin) {
			qmin = qnew;
			isave = i;
			tempa = qsav;
		}
		else if (i == isave + 1)
			tempb = qnew;
		qsav = qnew;
	}
	if (isave == 0) tempa = qnew;
	if (isave == iu) tempb = qbeg;
	angle = zero;
	if (tempa != tempb) {
		tempa -= qmin;
		tempb -= qmin;
		angle = half * (tempa - tempb) / (tempa + tempb);
	}
	angle = temp * ((REAL)isave + angle);

	/* Calculate the new STEP and HS. Then test for convergence. */

	cth = COS(angle);
	sth = SIN(angle);
	reduc = qbeg - (sg + cf * cth) * cth - (dg + dhs * cth) * sth;
	gg = zero;
	for (i = 1; i <= n; ++i) {
		step[i] = cth * step[i] + sth * d[i];
		hs[i] = cth * hs[i] + sth * hd[i];
		temp = g[i] + hs[i];
		gg += temp * temp;
	}
	qred += reduc;
	ratio = reduc / qred;
	if (iterc < itermax && ratio > .01) goto L90;
L160:
	symv(n, one, &hq[1], 1, &step[1], 1, zero, &hd[1], 1);
	*vquad = zero;
	for (i = 1; i <= n; ++i)
		*vquad += step[i] * (gbeg[i - 1] + half * hd[i]);
	// *vquad += step[i]*(gq[i] + half * hd[i]) + xopt[i]*hd[i]
	if (*vquad > zero) {
		fprintf(stdout, " Warning: the TR subproblem was not well solved!\n");
		REAL t = zero;
		for (i = 1; i <= n; ++i)
			t += step[i] * step[i];
		fprintf(stdout, " vquad=%25.15E Stepsize=%25.15E\n", (double)*vquad, (double)SQRT(t));
		if (SQRT(t) >= half * delta) return -100;
	}
	return 0;
} /* trsapp_h */

static void
biglag(const INTEGER n, const INTEGER npt, REAL* xopt,
	REAL* xpt, REAL* bmat, REAL* zmat, INTEGER* idz,
	const INTEGER ndim, const INTEGER kopt, const INTEGER knew,
	const REAL delta, REAL* d, REAL* alpha, REAL* gw, REAL* hcol, REAL* w)
{
	/*	N is the number of variables.
		NPT is the number of interpolation equations.
		XOPT is the best interpolation point so far.
		XPT contains the coordinates of the current interpolation points.
		BMAT provides the last N columns of H.
		ZMAT and IDZ give a factorization of the first NPT by NPT submatrix of H.
		NDIM is the first dimension of BMAT and has the value NPT+N.
		KOPT is the index of the optimal interpolation point.
		KNEW is the index of the interpolation point that is going to be moved.
		DELTA is the current trust region bound.
		D will be set to the step from XOPT to the new point.
		ALPHA will be set to the KNEW-th diagonal element of the H matrix.
		GW, HCOL and W will be used for working space.

		The step D is calculated in a way that attempts to maximize the modulus
		of LFUNC(XOPT+D), subject to the bound ||D|| <= DELTA, where LFUNC is
		the KNEW-th Lagrange function. */

		/* Constants */
	const REAL dsq = delta * delta;
	const INTEGER nptm = npt - n - 1;

	/* Local variables */
	REAL angle, cf1, cf2, cf3, cf4, cf5, cth, dd, denom, dgd, dgw, gwsq,
		scale, step, sth, sum, tau, taubeg, taumax, tauold,
		temp, tempa, tempb, tmpa, tmpb, vlnew;
	INTEGER i, j, k, iu, iterc, isave;

	/* Parameter adjustments */
	zmat -= npt + 1;
	xpt -= npt + 1;
	--xopt;
	bmat -= ndim + 1;
	--d;
	--gw;
	--hcol;
	--w;

	/* Set uninitialized variables */
	tempa = zero;
	tempb = zero;

	/* Set the first D and GW, where GW is the gradient of LFUNC at XOPT. The
	   first NPT components of HCOL and W will be set to the leading elements
	   of the KNEW-th column of H and the scalar products (XPT(K,.),XOPT),
	   K=1,2,...,NPT, respectively. DGD will be set to the curvature of LFUNC
	   in the direction D. */
	iterc = 0;
	for (i = 1; i <= n; ++i) {
		d[i] = xpt[knew + i * npt] - xopt[i];
		gw[i] = bmat[knew + i * ndim];
	}
	for (k = 1; k <= npt; ++k)
		hcol[k] = zero;
	for (j = 1; j <= nptm; ++j) {
		temp = zmat[knew + j * npt];
		if (j < *idz) temp = -temp;
		for (k = 1; k <= npt; ++k)
			hcol[k] += temp * zmat[k + j * npt];
	}
	dgd = zero;
	for (k = 1; k <= npt; ++k) {
		w[k] = zero;
		sum = zero;
		for (j = 1; j <= n; ++j) {
			w[k] += xopt[j] * xpt[k + j * npt];
			sum += d[j] * xpt[k + j * npt];
		}
		temp = hcol[k] * w[k];
		dgd += hcol[k] * sum * sum;
		for (i = 1; i <= n; ++i)
			gw[i] += temp * xpt[k + i * npt];
	}
	*alpha = hcol[knew];

	/* Step along the direction D or -D if the usefulness of GW is doubtful,
	   where the tests depend on the angle between GW and D and on ||GW||. */
	dd = zero;
	gwsq = zero;
	dgw = zero;
	for (i = 1; i <= n; ++i) {
		dd += d[i] * d[i];
		dgw += d[i] * gw[i];
		gwsq += gw[i] * gw[i];
	}
	scale = delta / SQRT(dd);
	if (dgw * dgd < zero) scale = -scale;
	for (i = 1; i <= n; ++i)
		d[i] = scale * d[i];
	dgw = scale * dgw;
	denom = dsq * gwsq - dgw * dgw;
	if (denom <= dsq * .01 * gwsq) goto L150;
	vlnew = dgw + half * scale * scale * dgd;
	if (dsq * gwsq < vlnew * .01 * vlnew) goto L150;

	/* Begin the iteration by making GW orthogonal to D and of length DELTA. */
L80:
	++iterc;
	denom = SQRT(denom);
	for (i = 1; i <= n; ++i)
		gw[i] = (dsq * gw[i] - dgw * d[i]) / denom;

	/* Find the elements of W_check, and accumulate their contributions to
	   the coefficients of TAU, which is the restriction of LFUNC to a two
	   dimensional part of the boundary of the trust region. */
	cf1 = zero;
	cf2 = zero;
	cf3 = zero;
	cf4 = zero;
	cf5 = zero;
	for (k = 1; k <= npt; ++k) {
		tempa = zero;
		tempb = zero;
		for (i = 1; i <= n; ++i) {
			tempa += xpt[k + i * npt] * d[i];
			tempb += xpt[k + i * npt] * gw[i];
		}
		tmpa = tempa * hcol[k];
		tmpb = tempb * hcol[k];
		cf1 += half * tmpb * tempb;
		cf2 += tmpa * w[k];
		cf3 += tmpb * w[k];
		cf4 += half * (tmpa * tempa - tmpb * tempb);
		cf5 += tmpa * tempb;
	}
	for (i = 1; i <= n; ++i) {
		temp = bmat[knew + i * ndim];
		cf2 += temp * d[i];
		cf3 += temp * gw[i];
	}

	/* Seek the value of the angle that maximizes the modulus of TAU. */
	taubeg = cf1 + cf2 + cf4;
	taumax = taubeg;
	tauold = taubeg;
	isave = 0;
	iu = 49;
	temp = twopi / (REAL)(iu + 1);
	for (i = 1; i <= iu; ++i) {
		angle = (REAL)i * temp;
		cth = COS(angle);
		sth = SIN(angle);
		tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
		if (ABS(tau) > ABS(taumax)) {
			taumax = tau;
			isave = i;
			tempa = tauold;
		}
		else if (i == isave + 1)
			tempb = tau;
		tauold = tau;
	}
	if (isave == 0) tempa = tau;
	if (isave == iu) tempb = taubeg;
	step = zero;
	if (tempa != tempb) {
		tempa -= taumax;
		tempb -= taumax;
		step = half * (tempa - tempb) / (tempa + tempb);
	}
	angle = temp * ((REAL)isave + step);

	/* Calculate the new D and test for convergence. */
	cth = COS(angle);
	sth = SIN(angle);
	tau = cf1 + (cf2 + cf4 * cth) * cth + (cf3 + cf5 * cth) * sth;
	for (i = 1; i <= n; ++i)
		d[i] = cth * d[i] + sth * gw[i];
	if (iterc >= n) goto L200;
	if (iterc == 1) taubeg = zero;
	if (ABS(tau) <= ABS(taubeg) * 1.1) goto L200;

	/* Set GW to the gradient of LFUNC at the new displacement D from XOPT.
	   Then branch for the next iteration unless GW and D are nearly parallel. */
L150:
	for (i = 1; i <= n; ++i)
		gw[i] = bmat[knew + i * ndim];
	for (k = 1; k <= npt; ++k) {
		sum = w[k];
		for (j = 1; j <= n; ++j)
			sum += d[j] * xpt[k + j * npt];
		temp = hcol[k] * sum;
		for (i = 1; i <= n; ++i)
			gw[i] += temp * xpt[k + i * npt];
	}
	gwsq = zero;
	dgw = zero;
	for (i = 1; i <= n; ++i) {
		gwsq += gw[i] * gw[i];
		dgw += d[i] * gw[i];
	}
	denom = dsq * gwsq - dgw * dgw;
	if (denom >= dsq * 1e-8 * gwsq) goto L80;
L200:
	return;
} /* biglag */

static void
bigden(const INTEGER n, const INTEGER npt, REAL* xopt,
	REAL* xpt, REAL* bmat, REAL* zmat, const INTEGER idz,
	const INTEGER ndim, const INTEGER kopt, const INTEGER knew, const REAL delta,
	REAL* d, REAL* vlag, REAL* beta, REAL* gw,
	REAL* w, REAL* wvec, REAL* prod)
{
	/*	N is the number of variables.
		NPT is the number of interpolation equations.
		XOPT is the best interpolation point so far.
		XPT contains the coordinates of the current interpolation points.
		BMAT provides the last N columns of H.
		ZMAT and IDZ give a factorization of the first NPT by NPT submatrix of H.
		NDIM is the first dimension of BMAT and has the value NPT+N.
		KOPT is the index of the optimal interpolation point.
		KNEW is the index of the interpolation point that is going to be moved.
		DELTA is the current trust region bound.
		D will be set to the step from XOPT to the new point.
		VLAG will be set to Theta*Wcheck+e_b for the final choice of D.
		BETA will be set to the value that will occur in the updating formula
		  when the KNEW-th interpolation point is moved to its new position.
		GW, W, WVEC, PROD and the private arrays DEN, DENEX and WW will be
		  used for working space, but on return W will be set to W_check.
		  The space for W may be part of the space for PROD.

		D is calculated in a way that should provide a denominator with a large
		modulus in the updating formula when the KNEW-th interpolation point is
		shifted to the new position XOPT+D. */

		/* Constants */
	const REAL dsq = delta * delta;
	const INTEGER nptm = npt - n - 1;

	/* Local variables */
	REAL alpha, angle, dd, denmax, denold, dg, gg,
		prval, step, sum, sumold, tau, temp, tempa, tempb, tempc,
		tempd, tempg, xoptd, xoptg, xoptsq, den[9], denex[9], ww[9];
	INTEGER i, j, k, jc, ip, iu, ku, iterc, isave;

	/* Parameter adjustments */
	zmat -= npt + 1;
	xpt -= npt + 1;
	--xopt;
	prod -= ndim + 1;
	wvec -= ndim + 1;
	bmat -= ndim + 1;
	--d;
	--vlag;
	--gw;
	--w;

	/* Set uninitialized variables */
	xoptsq = zero;

	/* Set the initial D and G, where G is the gradient of the KNEW-th
	   Lagrange function at the trust region centre. The gradient of this
	   function at XPT(KNEW,.) is set in W. The array VLAG will hold the
	   second derivative coefficients of the Lagrange function. */

	for (i = 1; i <= n; ++i) {
		d[i] = xpt[knew + i * npt] - xopt[i];
		gw[i] = bmat[knew + i * ndim];
		w[i] = bmat[knew + i * ndim];
	}
	for (k = 1; k <= npt; ++k)
		vlag[k] = zero;
	for (j = 1; j <= nptm; ++j) {
		temp = zmat[knew + j * npt];
		if (j < idz) temp = -temp;
		for (k = 1; k <= npt; ++k)
			vlag[k] += temp * zmat[k + j * npt];
	}
	for (k = 1; k <= npt; ++k) {
		sum = zero;
		tempb = zero;
		for (j = 1; j <= n; ++j) {
			sum += xpt[k + j * npt] * xopt[j];
			tempb += xpt[k + j * npt] * xpt[knew + j * npt];
		}
		if (k == kopt) xoptsq = sum;
		tempa = vlag[k] * sum;
		tempb = vlag[k] * tempb;
		for (i = 1; i <= n; ++i) {
			gw[i] += tempa * xpt[k + i * npt];
			w[i] += tempb * xpt[k + i * npt];
		}
	}
	alpha = vlag[knew];

	/* Revise G if its modulus seems to be unusually small. */
	temp = zero;
	tempa = zero;
	tempb = zero;
	for (i = 1; i <= n; ++i) {
		temp += d[i] * d[i];
		tempa += gw[i] * gw[i];
		tempb += w[i] * w[i];
	}
	if (tempb * dsq > temp * 1e4 * tempa)
		for (i = 1; i <= n; ++i) gw[i] = w[i];

	/* Begin the iteration by making D and G orthogonal and of length DELTA. */
	iterc = 0;
L80:
	++iterc;
	dd = zero;
	dg = zero;
	for (i = 1; i <= n; ++i) {
		dd += d[i] * d[i];
		dg += d[i] * gw[i];
	}
	gg = zero;
	for (i = 1; i <= n; ++i) {
		gw[i] = dd * gw[i] - dg * d[i];
		gg += gw[i] * gw[i];
	}
	tempd = delta / SQRT(dd);
	tempg = delta / SQRT(gg);
	xoptd = zero;
	xoptg = zero;
	for (i = 1; i <= n; ++i) {
		d[i] = tempd * d[i];
		gw[i] = tempg * gw[i];
		xoptd += xopt[i] * d[i];
		xoptg += xopt[i] * gw[i];
	}

	/* Set the coefficients of the first two terms of BETA. */
	tempa = half * xoptd * xoptd;
	tempb = half * xoptg * xoptg;
	den[0] = dsq * (xoptsq + half * dsq) + tempa + tempb;
	den[1] = 2 * xoptd * dsq;
	den[2] = 2 * xoptg * dsq;
	den[3] = tempa - tempb;
	den[4] = xoptd * xoptg;
	for (i = 6; i <= 9; ++i)
		den[i - 1] = zero;

	/* Put the coefficients of Wcheck in WVEC. */
	for (k = 1; k <= npt; ++k) {
		tempa = zero;
		tempb = zero;
		tempc = zero;
		for (i = 1; i <= n; ++i) {
			tempa += xpt[k + i * npt] * d[i];
			tempb += xpt[k + i * npt] * gw[i];
			tempc += xpt[k + i * npt] * xopt[i];
		}
		wvec[k + ndim] = (tempa * tempa + tempb * tempb) / 4;
		wvec[k + ndim * 2] = tempa * tempc;
		wvec[k + ndim * 3] = tempb * tempc;
		wvec[k + ndim * 4] = (tempa * tempa - tempb * tempb) / 4;
		wvec[k + ndim * 5] = half * tempa * tempb;
	}
	for (i = 1; i <= n; ++i) {
		ip = i + npt;
		wvec[ip + ndim] = zero;
		wvec[ip + ndim * 2] = d[i];
		wvec[ip + ndim * 3] = gw[i];
		wvec[ip + ndim * 4] = zero;
		wvec[ip + ndim * 5] = zero;
	}

	/* Put the coefficents of THETA*Wcheck in PROD. */

	for (jc = 1; jc <= 5; ++jc) {
		ku = npt;
		if (jc == 2 || jc == 3) ku = ndim;
		for (i = 1; i <= npt; ++i)
			prod[i + jc * ndim] = zero;
		for (j = 1; j <= nptm; ++j) {
			sum = zero;
			for (i = 1; i <= npt; ++i)
				sum += zmat[i + j * npt] * wvec[i + jc * ndim];
			if (j < idz) sum = -sum;
			for (i = 1; i <= npt; ++i)
				prod[i + jc * ndim] += sum * zmat[i + j * npt];
		}
		if (ku == ndim) {
			for (i = 1; i <= npt; ++i) {
				sum = zero;
				for (j = 1; j <= n; ++j)
					sum += bmat[i + j * ndim] * wvec[npt + j + jc * ndim];
				prod[i + jc * ndim] += sum;
			}
		}
		for (i = 1; i <= n; ++i) {
			sum = zero;
			for (k = 1; k <= ku; ++k)
				sum += bmat[k + i * ndim] * wvec[k + jc * ndim];
			prod[npt + i + jc * ndim] = sum;
		}
	}

	/* Include in DEN the part of BETA that depends on THETA. */
	for (k = 1; k <= ndim; ++k) {
		sum = zero;
		for (i = 1; i <= 5; ++i) {
			ww[i - 1] = half * prod[k + i * ndim] * wvec[k + i * ndim];
			sum += ww[i - 1];
		}
		den[0] = den[0] - ww[0] - sum;
		tempa = prod[k + ndim] * wvec[k + ndim * 2] +
			prod[k + ndim * 2] * wvec[k + ndim];
		tempb = prod[k + ndim * 2] * wvec[k + ndim * 4] +
			prod[k + ndim * 4] * wvec[k + ndim * 2];
		tempc = prod[k + ndim * 3] * wvec[k + ndim * 5] +
			prod[k + ndim * 5] * wvec[k + ndim * 3];
		den[1] = den[1] - tempa - half * (tempb + tempc);
		den[5] -= half * (tempb - tempc);
		tempa = prod[k + ndim] * wvec[k + ndim * 3] +
			prod[k + ndim * 3] * wvec[k + ndim];
		tempb = prod[k + ndim * 2] * wvec[k + ndim * 5] +
			prod[k + ndim * 5] * wvec[k + ndim * 2];
		tempc = prod[k + ndim * 3] * wvec[k + ndim * 4] +
			prod[k + ndim * 4] * wvec[k + ndim * 3];
		den[2] = den[2] - tempa - half * (tempb - tempc);
		den[6] -= half * (tempb + tempc);
		tempa = prod[k + ndim] * wvec[k + ndim * 4] +
			prod[k + ndim * 4] * wvec[k + ndim];
		den[3] = den[3] - tempa - ww[1] + ww[2];
		tempa = prod[k + ndim] * wvec[k + ndim * 5] +
			prod[k + ndim * 5] * wvec[k + ndim];
		tempb = prod[k + ndim * 2] * wvec[k + ndim * 3] +
			prod[k + ndim * 3] * wvec[k + ndim * 2];
		den[4] = den[4] - tempa - half * tempb;
		den[7] = den[7] - ww[3] + ww[4];
		tempa = prod[k + ndim * 4] * wvec[k + ndim * 5] +
			prod[k + ndim * 5] * wvec[k + ndim * 4];
		den[8] -= half * tempa;
	}

	/* Extend DEN so that it holds all the coefficients of DENOM. */
	sum = zero;
	for (i = 1; i <= 5; ++i) {
		ww[i - 1] = half * (prod[knew + i * ndim] * prod[knew + i * ndim]);
		sum += ww[i - 1];
	}
	denex[0] = alpha * den[0] + ww[0] + sum;
	tempa = 2 * prod[knew + ndim] * prod[knew + ndim * 2];
	tempb = prod[knew + ndim * 2] * prod[knew + ndim * 4];
	tempc = prod[knew + ndim * 3] * prod[knew + ndim * 5];
	denex[1] = alpha * den[1] + tempa + tempb + tempc;
	denex[5] = alpha * den[5] + tempb - tempc;
	tempa = 2 * prod[knew + ndim] * prod[knew + ndim * 3];
	tempb = prod[knew + ndim * 2] * prod[knew + ndim * 5];
	tempc = prod[knew + ndim * 3] * prod[knew + ndim * 4];
	denex[2] = alpha * den[2] + tempa + tempb - tempc;
	denex[6] = alpha * den[6] + tempb + tempc;
	tempa = 2 * prod[knew + ndim] * prod[knew + ndim * 4];
	denex[3] = alpha * den[3] + tempa + ww[1] - ww[2];
	tempa = 2 * prod[knew + ndim] * prod[knew + ndim * 5];
	denex[4] = alpha * den[4] + tempa + prod[knew + ndim * 2] * prod[knew + ndim * 3];
	denex[7] = alpha * den[7] + ww[3] - ww[4];
	denex[8] = alpha * den[8] + prod[knew + ndim * 4] * prod[knew + ndim * 5];

	/* Seek the value of the angle that maximizes the modulus of DENOM. */
	sum = denex[0] + denex[1] + denex[3] + denex[5] + denex[7];
	denold = sum;
	denmax = sum;
	isave = 0;
	iu = 49;
	temp = twopi / (REAL)(iu + 1);
	ww[0] = one;
	for (i = 1; i <= iu; ++i) {
		angle = (REAL)i * temp;
		ww[1] = COS(angle);
		ww[2] = SIN(angle);
		for (j = 4; j <= 8; j += 2) {
			ww[j - 1] = ww[1] * ww[j - 3] - ww[2] * ww[j - 2];
			ww[j] = ww[1] * ww[j - 2] + ww[2] * ww[j - 3];
		}
		sumold = sum;
		sum = zero;
		for (j = 1; j <= 9; ++j)
			sum += denex[j - 1] * ww[j - 1];
		if (ABS(sum) > ABS(denmax)) {
			denmax = sum;
			isave = i;
			tempa = sumold;
		}
		else if (i == isave + 1) tempb = sum;
	}
	if (isave == 0) tempa = sum;
	if (isave == iu) tempb = denold;
	step = zero;
	if (tempa != tempb) {
		tempa -= denmax;
		tempb -= denmax;
		step = half * (tempa - tempb) / (tempa + tempb);
	}
	angle = temp * ((REAL)isave + step);

	/* Calculate the new D and test for convergence. */
	ww[1] = COS(angle);
	ww[2] = SIN(angle);
	for (i = 1; i <= n; ++i)
		d[i] = ww[1] * d[i] + ww[2] * gw[i];
	for (j = 4; j <= 8; j += 2) {
		ww[j - 1] = ww[1] * ww[j - 3] - ww[2] * ww[j - 2];
		ww[j] = ww[1] * ww[j - 2] + ww[2] * ww[j - 3];
	}
	*beta = zero;
	denmax = zero;
	tau = zero;
	for (j = 1; j <= 9; ++j) {
		*beta += den[j - 1] * ww[j - 1];
		denmax += denex[j - 1] * ww[j - 1];
		if (j <= 5) tau += prod[knew + j * ndim] * ww[j - 1];
	}
	if (iterc >= n) goto L390;
	if (iterc == 1) denold = zero;
	if (ABS(denmax) <= ABS(denold) * 1.2) goto L390;

	/* Set G to half the gradient of DENOM with respect to D. Then branch
	   for the next iteration. */

	for (i = 1; i <= n; ++i)
		gw[i] = tau * bmat[knew + i * ndim];
	for (k = 1; k <= ndim; ++k) {
		prval = zero;
		for (j = 1; j <= 5; ++j) {
			prval += prod[k + j * ndim] * ww[j - 1];
		}
		if (k <= npt) {
			sum = zero;
			for (i = 1; i <= n; ++i)
				sum += xpt[k + i * npt] * (xopt[i] + d[i]);
			if (k == kopt) {
				tempa = alpha * (sum - xoptsq + dsq);
				tempb = tempa + alpha * sum;
				for (i = 1; i <= n; ++i)
					gw[i] = gw[i] + tempa * xopt[i] + tempb * d[i];
			}
			temp = (tau * vlag[k] - alpha * prval) * sum;
			for (i = 1; i <= n; ++i)
				gw[i] += temp * xpt[k + i * npt];
		}
		else gw[k - npt] -= alpha * prval;
	}
	gg = zero;
	dg = zero;
	for (i = 1; i <= n; ++i) {
		gg += gw[i] * gw[i];
		dg += d[i] * gw[i];
	}
	temp = dg * dg / (dsq * gg);
	if (temp <= one - 1e-8) goto L80;

	/* Set the vector VLAG before the RETURN from the subroutine. */
L390:
	for (k = 1; k <= ndim; ++k) {
		vlag[k] = zero;
		sum = zero;
		for (j = 1; j <= 5; ++j) {
			vlag[k] += prod[k + j * ndim] * ww[j - 1];
			sum += wvec[k + j * ndim] * ww[j - 1];
		}
		w[k] = sum;
	}
	vlag[kopt] += one;
	return;
} /* bigden */

static void
update(const INTEGER n, const INTEGER npt, REAL* bmat,
	REAL* zmat, INTEGER* idz, const INTEGER ndim, REAL* vlag,
	const REAL beta, const INTEGER knew, REAL* w)
{
	/*	The arrays BMAT and ZMAT with IDZ are updated, in order to shift the
		interpolation point that has index KNEW. On entry, VLAG contains the
		components of the vector Theta*Wcheck+e_b of the updating formula
		(6.11), and BETA holds the value of the parameter that has this name.
		The vector W is used for working space. */

		/* Constants */
	const INTEGER nptm = npt - n - 1;

	/* Local variables */
	REAL alpha, denom, scala, scalb, tau, tausq, temp, tempa, tempb;
	INTEGER i, iflag, j, ja, jb, jl, jp;


	/* Parameter adjustments */
	zmat -= npt + 1;
	bmat -= ndim + 1;
	--vlag;
	--w;

	/* Apply the rotations that put zeros in the KNEW-th row of ZMAT. */
	jl = 1;
	for (j = 2; j <= nptm; ++j) {
		if (j == *idz) jl = *idz;
		else if (zmat[knew + j * npt] != zero) {
			tempa = zmat[knew + jl * npt];
			tempb = zmat[knew + j * npt];
			temp = SQRT(tempa * tempa + tempb * tempb);
			tempa /= temp;
			tempb /= temp;
			for (i = 1; i <= npt; ++i) {
				temp = tempa * zmat[i + jl * npt] + tempb * zmat[i + j * npt];
				zmat[i + j * npt] = tempa * zmat[i + j * npt] - tempb * zmat[i + jl * npt];
				zmat[i + jl * npt] = temp;
			}
			zmat[knew + j * npt] = zero;
		}
	}

	/* Put the first NPT components of the KNEW-th column of HLAG into W,
	   and calculate the parameters of the updating formula. */
	tempa = zmat[knew + npt];
	if (*idz >= 2) tempa = -tempa;
	if (jl > 1) tempb = zmat[knew + jl * npt];
	for (i = 1; i <= npt; ++i) {
		w[i] = tempa * zmat[i + npt];
		if (jl > 1) w[i] += tempb * zmat[i + jl * npt];
	}
	alpha = w[knew];
	tau = vlag[knew];
	tausq = tau * tau;
	denom = alpha * beta + tausq;
	vlag[knew] -= one;

	/* Complete the updating of ZMAT when there is only one nonzero element
	   in the KNEW-th row of the new matrix ZMAT, but, if IFLAG is set to one,
	   then the first column of ZMAT will be exchanged with another one later. */
	iflag = 0;
	if (jl == 1) {
		temp = alpha / denom;
		tempa = SQRT(ABS(temp));
		tempb = tempa * tau / alpha;
		for (i = 1; i <= npt; ++i)
			zmat[i + npt] = tempa * vlag[i] - tempb * w[i];
		if (*idz == 1 && temp < zero) *idz = 2;
		if (*idz >= 2 && temp >= zero) iflag = 1;
	}
	else {
		/* Complete the updating of ZMAT in the alternative case. */
		ja = 1;
		if (beta >= zero) ja = jl;
		jb = jl + 1 - ja;
		temp = zmat[knew + jb * npt] / denom;
		tempa = temp * beta;
		tempb = temp * tau;
		temp = zmat[knew + ja * npt];
		scala = one / SQRT(ABS(beta) * temp * temp + tausq);
		scalb = scala * SQRT(ABS(denom));
		for (i = 1; i <= npt; ++i) {
			zmat[i + ja * npt] = scala * (tau * zmat[i + ja * npt] - temp * vlag[i]);
			zmat[i + jb * npt] = scalb * (zmat[i + jb * npt] - tempa * w[i] - tempb * vlag[i]);
		}
		if (denom <= zero) {
			if (beta < zero) ++(*idz);
			else iflag = 1;
		}
	}

	/* IDZ is reduced in the following case, and usually the first column
	   of ZMAT is exchanged with a later one. */

	if (iflag == 1) {
		--(*idz);
		for (i = 1; i <= npt; ++i) {
			temp = zmat[i + npt];
			zmat[i + npt] = zmat[i + *idz * npt];
			zmat[i + *idz * npt] = temp;
		}
	}

	/* Finally, update the matrix BMAT. */
	for (j = 1; j <= n; ++j) {
		jp = npt + j;
		w[jp] = bmat[knew + j * ndim];
		tempa = (alpha * vlag[jp] - tau * w[jp]) / denom;
		tempb = (-beta * w[jp] - tau * vlag[jp]) / denom;
		for (i = 1; i <= jp; ++i) {
			bmat[i + j * ndim] = bmat[i + j * ndim] + tempa * vlag[i] + tempb * w[i];
			if (i > npt) bmat[jp + (i - npt) * ndim] = bmat[i + j * ndim];
		}
	}
} /* update */

static void
print_error(const char* reason)
{
	fprintf(stderr, "\n    Return from NEWUOA_H because %s.\n", reason);
}

static void
print_x(FILE* output, INTEGER n, const REAL* x, const REAL* dx)
{
	INTEGER i;
	for (i = 0; i < n; ++i) {
		fprintf(output, "%s%15.6E%s",
			((i % 5 == 0) ? "  " : ""),
			(double)(dx == NULL ? x[i] : (x[i] + dx[i])),
			((i == n - 1 || i % 5 == 4) ? "\n" : ""));
	}
}

static int newuob_h(const INTEGER n, const INTEGER npt, newuoa_dfovec* dfovec,
	void* data, REAL* x, const REAL rhobeg, const REAL rhoend,
	const INTEGER iprint, const INTEGER maxfun,
	REAL* xbase, REAL* xopt, REAL* xnew,
	REAL* xpt, REAL* gq, REAL* hq, REAL* pq,
	REAL* bmat, REAL* zmat, const INTEGER ndim, REAL* d,
	REAL* vlag, REAL* w, REAL* diffv, REAL* v_beg, REAL* v_err, REAL* v_opt,
	REAL* v_base, REAL* v_temp, REAL* v_vquad, REAL* wv, REAL* gqv, REAL* hqv,
	REAL* pqv, REAL* gqv_opt, REAL* hd1, const INTEGER mv)
{
	/*	The arguments N, NPT, DFOVEC, DATA, X, RHOBEG, RHOEND, IPRINT, MAXFUN and MV
		are identical to the corresponding arguments in SUBROUTINE NEWUOA_H.

		XBASE will hold a shift of origin that should reduce the contributions
		from rounding errors to values of the model and Lagrange functions.

		XOPT will be set to the displacement from XBASE of the vector of variables
		that provides the least calculated F so far.

		XNEW will be set to the displacement from XBASE of the vector of variables
		for the current calculation of F.

		XPT will contain the interpolation point coordinates relative to XBASE.

		GQ will hold the gradient of the quadratic model at XBASE.

		HQ will hold the explicit second derivatives of the quadratic model.

		PQ will contain the parameters of the implicit second derivatives of the
		quadratic model.

		BMAT will hold the last N columns of H.

		ZMAT will hold the factorization of the leading NPT by NPT submatrix of H,
		this factorization being ZMAT times Diag(DZ) times ZMAT^T, where the
		elements of DZ are plus or minus one, as specified by IDZ.

		NDIM is the first dimension of BMAT and has the value NPT+N.

		D is reserved for trial steps from XOPT.

		VLAG will contain the values of the Lagrange functions at a new point X.
		They are part of a product that requires VLAG to be of length NDIM.

		The array W will be used for working space. Its length must be at least
		10*NDIM = 10*(NPT+N). */

		/* Constants */
	const INTEGER np = n + 1, nptm = npt - np;
	const char* reason = NULL;

	/* Local variables */
	REAL alpha, beta, crvmin, delta, diff, diffa, diffb, diffc, dnorm, dsq,
		dstep, f, fbeg, fopt, ratio, reciq, rho, rhosq, vquad1, xoptsq;
	INTEGER idz, ih, ip, iteropt, knew, kopt, ksave, m1, nf, nfm, nfmm, nfsav;
	LOGICAL model_update = 1, opt_update = 1;
	int status;

	/* Temporary variables */
	REAL bsum, detrat, distsq, dx, fsave, temp;
	INTEGER i, j, k, ktemp;

	/* Parameter adjustments */
	zmat -= npt + 1;
	xpt -= npt + 1;
	--x;
	--xbase;
	--xopt;
	--xnew;
	--gq;
	--hq;
	--pq;
	bmat -= ndim + 1;
	--d;
	--vlag;
	--w;

	/* Set uninitialized variables */
	idz = 0;
	knew = 0;
	kopt = 0;
	nfsav = 0;
	iteropt = 0;
	alpha = zero;
	crvmin = zero;
	delta = zero;
	diffa = zero;
	diffb = zero;
	diffc = zero;
	dnorm = zero;
	dsq = zero;
	f = zero;
	fbeg = zero;
	fopt = zero;
	ratio = zero;
	rho = zero;
	vquad1 = zero;
	xoptsq = zero;
	dstep = zero;

	/* Set the initial elements of XPT, BMAT, HQ, PQ and ZMAT to zero. */
	for (j = 1; j <= n; ++j) {
		xbase[j] = x[j];
		for (k = 1; k <= npt; ++k)
			xpt[k + j * npt] = zero;
		for (i = 1; i <= ndim; ++i)
			bmat[i + j * ndim] = zero;
	}
	for (j = 1; j <= n * np / 2; ++j) {
		for (m1 = 1; m1 <= mv; ++m1)
			hqv[m1 + j * mv - mv - 1] = zero;
		hq[j] = zero;
	}
	for (k = 1; k <= npt; ++k) {
		for (m1 = 1; m1 <= mv; ++m1)
			pqv[m1 + k * mv - mv - 1] = zero;
		pq[k] = zero;
		for (j = 1; j <= nptm; ++j)
			zmat[k + j * npt] = zero;
	}

	/* Begin the initialization procedure. NF becomes one more than the number
	   of function values so far. The coordinates of the displacement of the
	   next initial interpolation point from XBASE are set in XPT(NF,.). */
	rhosq = rhobeg * rhobeg;
	//	recip = one / rhosq;
	reciq = SQRT(half) / rhosq;
	nf = 0;
L50:
	nfm = nf;
	nfmm = nf - n;
	++nf;
	if (nfm <= 2 * n) {
		if (nfm >= 1 && nfm <= n)
			xpt[nf + nfm * npt] = rhobeg;
		else if (nfm > n)
			xpt[nf + nfmm * npt] = -rhobeg;
	}

	/* Calculate the next value of F, label 70 being reached immediately
	   after this calculation. The least function value so far and its index
	   are required. */
	for (j = 1; j <= n; ++j)
		x[j] = xpt[nf + j * npt] + xbase[j];
	goto L310;
L70:
	w[nf] = f;
	if (nf == 1) {
		fbeg = f;
		fopt = f;
		for (m1 = 1; m1 <= mv; ++m1) {
			REAL t = v_err[m1 - 1];
			v_base[m1 - 1] = t;
			v_beg[m1 - 1] = t;
			v_opt[m1 - 1] = t;
		}
		kopt = 1;
	}
	else if (f < fopt) {
		fopt = f;
		for (m1 = 1; m1 <= mv; ++m1)
			v_opt[m1 - 1] = v_err[m1 - 1];
		kopt = nf;
	}

	/* Set the nonzero initial elements of BMAT and the quadratic model in
	   the cases when NF is at most 2*N+1. */
	if (nfm <= 2 * n) {
		if (nfm >= 1 && nfm <= n) {
			for (m1 = 1; m1 <= mv; ++m1)
				gqv[m1 + nfm * mv - mv - 1] = (v_err[m1 - 1] - v_beg[m1 - 1]) / rhobeg;
			if (npt < nf + n) {
				bmat[nfm * ndim + 1] = -one / rhobeg;
				bmat[nf + nfm * ndim] = one / rhobeg;
				bmat[npt + nfm + nfm * ndim] = -half * rhosq;
			}
		}
		else if (nfm > n) {
			bmat[nf - n + nfmm * ndim] = half / rhobeg;
			bmat[nf + nfmm * ndim] = -half / rhobeg;
			zmat[nfmm * npt + 1] = -reciq - reciq;
			zmat[nf - n + nfmm * npt] = reciq;
			zmat[nf + nfmm * npt] = reciq;
			ih = nfmm * (nfmm + 1) / 2;
			for (m1 = 1; m1 <= mv; ++m1) {
				temp = (v_beg[m1 - 1] - v_err[m1 - 1]) / rhobeg;
				hqv[m1 + ih * mv - mv - 1] = (gqv[m1 + nfmm * mv - mv - 1] - temp) / rhobeg;
				gqv[m1 + nfmm * mv - mv - 1] = half * (gqv[m1 + nfmm * mv - mv - 1] + temp);
			}
		}
	}
	if (nf < npt) goto L50;

	/* Begin the iterative procedure, because the initial model is complete. */
	rho = rhobeg;
	delta = rho;
	idz = 1;
	diffa = zero;
	diffb = zero;
	xoptsq = zero;
	for (i = 1; i <= n; ++i) {
		xopt[i] = xpt[kopt + i * npt];
		xoptsq += xopt[i] * xopt[i];
	}
L90:
	nfsav = nf;

	/* Generate the next trust region step and test its length. Set KNEW to -1 if
	   the purpose of the next F will be to improve the model. */
L100:
	knew = 0;
	if (debug) fprintf(stdout, " Before TRSAPP: delta=%25.15E  rho=%25.15E\n", (double)delta, (double)rho);
	status = trsapp_h(n, npt, &xopt[1], &xpt[npt + 1], &gq[1], &hq[1], &pq[1],
		delta, &d[1], &w[1], &w[np], &w[np + n], &w[np + 2 * n], &w[np + 3 * n], &w[np + 4 * n],
		&crvmin, gqv, hqv, pqv, &xbase[1], &vquad1, gqv_opt, v_opt,
		v_base, xoptsq, mv, &model_update, &opt_update);
	if (status) return status;
	dsq = zero;
	for (i = 1; i <= n; ++i)
		dsq += d[i] * d[i];
	dnorm = MIN(delta, SQRT(dsq));
	if (debug) fprintf(stdout, " After TRSAPP: ||d||=%25.15E  vquad1=%25.15E\n", (double)SQRT(dsq), (double)vquad1);

L111:
	if (dnorm < half * rho) {
		knew = -1;
		delta = tenth * delta;
		ratio = -1.0;
		if (delta <= rho * 1.5) delta = rho;
		if (nf <= nfsav + 2) goto L460;
		temp = crvmin * 0.125 * rho * rho;
		if (temp <= MAX(MAX(diffa, diffb), diffc)) goto L460;
		goto L490;
	}

	/* Shift XBASE if XOPT may be too far from XBASE. First make the changes
	   to BMAT that do not depend on ZMAT. */
L120:
	if (debug) fprintf(stdout, " DSQ=%25.15E XOPTSQ=%25.15E\n", (double)dsq, (double)xoptsq);
	if (dsq <= xoptsq * 0.1) {
		REAL sum, sumz, tempq;
		if (debug) fprintf(stdout, "  Xbase move\n");
		model_update = 1;
		tempq = xoptsq * 0.25;
		for (k = 1; k <= npt; ++k) {
			sum = zero;
			for (i = 1; i <= n; ++i)
				sum += xpt[k + i * npt] * xopt[i];
			for (m1 = 1; m1 <= mv; ++m1)
				v_temp[m1 - 1] = pqv[m1 + k * mv - mv - 1] * sum;
			sum -= half * xoptsq;
			w[npt + k] = sum;
			for (i = 1; i <= n; ++i) {
				for (m1 = 1; m1 <= mv; ++m1)
					gqv[m1 + i * mv - mv - 1] += v_temp[m1 - 1] * xpt[k + i * npt];
				xpt[k + i * npt] -= half * xopt[i];
				vlag[i] = bmat[k + i * ndim];
				w[i] = sum * xpt[k + i * npt] + tempq * xopt[i];
				ip = npt + i;
				for (j = 1; j <= i; ++j)
					bmat[ip + j * ndim] = bmat[ip + j * ndim] + vlag[i] * w[j] + w[i] * vlag[j];
			}
		}

		/* Then the revisions of BMAT that depend on ZMAT are calculated. */
		for (k = 1; k <= nptm; ++k) {
			sumz = zero;
			for (i = 1; i <= npt; ++i) {
				sumz += zmat[i + k * npt];
				w[i] = w[npt + i] * zmat[i + k * npt];
			}
			for (j = 1; j <= n; ++j) {
				sum = tempq * sumz * xopt[j];
				for (i = 1; i <= npt; ++i)
					sum += w[i] * xpt[i + j * npt];
				vlag[j] = sum;
				if (k < idz) sum = -sum;
				for (i = 1; i <= npt; ++i)
					bmat[i + j * ndim] += sum * zmat[i + k * npt];
			}
			for (i = 1; i <= n; ++i) {
				ip = i + npt;
				temp = vlag[i];
				if (k < idz) temp = -temp;
				for (j = 1; j <= i; ++j)
					bmat[ip + j * ndim] += temp * vlag[j];
			}
		}

		/* The following instructions complete the shift of XBASE, including
		   the changes to the parameters of the quadratic model. */
		ih = 0;
		for (j = 1; j <= n; ++j) {
			for (m1 = 1; m1 <= mv; ++m1)
				wv[m1 + j * mv - mv - 1] = zero;
			for (k = 1; k <= npt; ++k) {
				for (m1 = 1; m1 <= mv; ++m1)
					wv[m1 + j * mv - mv - 1] += pqv[m1 + k * mv - mv - 1] * xpt[k + j * npt];
				xpt[k + j * npt] -= half * xopt[j];
			}
			for (i = 1; i <= j; ++i) {
				++ih;
				if (i < j)
					for (m1 = 1; m1 <= mv; ++m1)
						gqv[m1 + j * mv - mv - 1] += hqv[m1 + ih * mv - mv - 1] * xopt[i];
				for (m1 = 1; m1 <= mv; ++m1) {
					gqv[m1 + i * mv - mv - 1] += hqv[m1 + ih * mv - mv - 1] * xopt[j];
					hqv[m1 + ih * mv - mv - 1] = hqv[m1 + ih * mv - mv - 1] +
						wv[m1 + i * mv - mv - 1] * xopt[j] +
						wv[m1 + j * mv - mv - 1] * xopt[i];
				}
				bmat[npt + i + j * ndim] = bmat[npt + j + i * ndim];
			}
		}
		for (j = 1; j <= n; ++j) {
			xbase[j] += xopt[j];
			xopt[j] = zero;
		}
		xoptsq = zero;
		for (m1 = 1; m1 <= mv; ++m1)
			v_base[m1 - 1] = v_opt[m1 - 1];
	}

	/* Pick the model step if KNEW is positive. A different choice of D
	   may be made later, if the choice of D by BIGLAG causes substantial
	   cancellation in DENOM. */
	if (knew > 0)
		biglag(n, npt, &xopt[1], &xpt[npt + 1], &bmat[ndim + 1], &zmat[npt + 1], &idz,
			ndim, kopt, knew, dstep, &d[1], &alpha, &w[1], &w[np], &w[ndim + 1]);

	/* Calculate VLAG and BETA for the current choice of D. The first NPT
	   components of W_check will be held in W. */
	for (k = 1; k <= npt; ++k) {
		REAL suma = zero;
		REAL sumb = zero;
		REAL sum = zero;
		for (j = 1; j <= n; ++j) {
			suma += xpt[k + j * npt] * d[j];
			sumb += xpt[k + j * npt] * xopt[j];
			sum += bmat[k + j * ndim] * d[j];
		}
		w[k] = suma * (half * suma + sumb);
		vlag[k] = sum;
	}
	beta = zero;
	for (k = 1; k <= nptm; ++k) {
		REAL sum = zero;
		for (i = 1; i <= npt; ++i)
			sum += zmat[i + k * npt] * w[i];
		if (k < idz) {
			beta += sum * sum;
			sum = -sum;
		}
		else beta -= sum * sum;
		for (i = 1; i <= npt; ++i)
			vlag[i] += sum * zmat[i + k * npt];
	}
	bsum = zero;
	dx = zero;
	for (j = 1; j <= n; ++j) {
		REAL sum = zero;
		for (i = 1; i <= npt; ++i)
			sum += w[i] * bmat[i + j * ndim];
		bsum += sum * d[j];
		INTEGER jp = npt + j;
		for (k = 1; k <= n; ++k)
			sum += bmat[jp + k * ndim] * d[k];
		vlag[jp] = sum;
		bsum += sum * d[j];
		dx += d[j] * xopt[j];
	}
	beta = dx * dx + dsq * (xoptsq + dx + dx + half * dsq) + beta - bsum;
	vlag[kopt] += one;

	/* If KNEW is positive and if the cancellation in DENOM is unacceptable,
	   then BIGDEN calculates an alternative model step, XNEW being used for
	   working space. */
	if (knew > 0) {
		temp = one + alpha * beta / (vlag[knew] * vlag[knew]);
		if (ABS(temp) <= 0.8)
			bigden(n, npt, &xopt[1], &xpt[npt + 1], &bmat[ndim + 1], &zmat[npt + 1], idz,
				ndim, kopt, knew, dstep, &d[1], &vlag[1], &beta, &xnew[1], &w[1],
				&w[ndim * 5 + 1], &w[1]);
	}

	/* Calculate the next value of the objective function. */
L290:
	for (i = 1; i <= n; ++i) {
		xnew[i] = xopt[i] + d[i];
		x[i] = xbase[i] + xnew[i];
	}
	++nf;
L310:
	if (nf > MAX(maxfun, 1)) {
		--nf;
		reason = "CALFUN has been called MAXFUN times";
		status = NEWUOA_TOO_MANY_EVALUATIONS;
		goto done;
	}

	/* dfovec(n, mv, x, v_err) provides the values of the vector function v_err(x): R^n \to R^{mv}.
	   Here: n, mv, x \in R^n are input, v_err \in R^{mv} are output. */
	dfovec(n, mv, &x[1], v_err, data);

	/* f_value(mv,v_err,F) provides the value of the sum of the squres of the components of v_err(x)
	   i.e. F = sum_{i=1}^{mv} v_err_i (x)^2 */
	f = f_value(mv, v_err);
	//if (iprint == 3) {
	//	fprintf(stdout, "\n"
	//		"    Function number%6ld    F =%18.10E"
	//		"    The corresponding X is:\n",
	//		(long)nf, (double)f);
	//	print_x(stdout, n, &x[1], NULL);
	//}
	if (iprint == 3) {
		fprintf(stdout, "\n"
			"    Function number%6ld    F =%18.10E\n",
			(long)nf, (double)f);
	}

	if (nf == 1) iteropt = 1;
	else if (f <= fopt) iteropt = nf;
	if (f <= MAX(1e-12, 1e-20 * fbeg)) {
		fprintf(stdout, "  F.le.dmax1(1.d-12,1.d-20*FBEG)\n");
		status = NEWUOA_SUCCESS;
		goto done;
	}
	if (nf <= npt) goto L70;
	if (knew == -1) {
		status = NEWUOA_SUCCESS;
		goto done;
	}

	/* Use the quadratic model to predict the change in F due to the step D,
	   and set DIFF to the error of this prediction. */
	for (m1 = 1; m1 <= mv; ++m1)
		v_vquad[m1 - 1] = zero;
	ih = 0;
	for (j = 1; j <= n; ++j) {
		for (m1 = 1; m1 <= mv; ++m1)
			v_vquad[m1 - 1] += d[j] * gqv[m1 + j * mv - mv - 1];
		for (i = 1; i <= j; ++i) {
			++ih;
			temp = d[i] * xnew[j] + d[j] * xopt[i];
			if (i == j) temp = half * temp;
			for (m1 = 1; m1 <= mv; ++m1)
				v_vquad[m1 - 1] += temp * hqv[m1 + ih * mv - mv - 1];
		}
	}
	for (k = 1; k <= npt; ++k)
		for (m1 = 1; m1 <= mv; ++m1)
			v_vquad[m1 - 1] += pqv[m1 + k * mv - mv - 1] * w[k];
	for (m1 = 1; m1 <= mv; ++m1)
		diffv[m1 - 1] = v_err[m1 - 1] - v_opt[m1 - 1] - v_vquad[m1 - 1];
	if (debug) fprintf(stdout, " Knew=%6ld vquad1 old=%25.15E\n", (long)knew, (double)vquad1);
	if (knew > 0) {
		for (i = 0; i < n; ++i) hd1[i] = zero;
		symv(n, one, &hq[1], 1, &d[1], 1, zero, hd1, 1);
		vquad1 = zero;
		for (i = 1; i <= n; ++i)
			vquad1 += d[i] * (gq[i] + half * hd1[i - 1]);
	}
	if (debug) fprintf(stdout, " vquad1 new=%25.15E\n", (double)vquad1);
	diff = f - fopt - vquad1;
	diffc = diffb;
	diffb = diffa;
	diffa = ABS(diff);
	if (dnorm > rho) nfsav = nf;

	/* Update FOPT and XOPT if the new F is the least value of the objective
	   function so far. The branch when KNEW is positive occurs if D is not
	   a trust region step. */
	fsave = fopt;
	if (f < fopt) {
		opt_update = 1;
		fopt = f;
		for (m1 = 1; m1 <= mv; ++m1)
			v_opt[m1 - 1] = v_err[m1 - 1];
		xoptsq = zero;
		for (i = 1; i <= n; ++i) {
			xopt[i] = xnew[i];
			xoptsq += xopt[i] * xopt[i];
		}
	}
	ksave = knew;
	if (knew > 0) goto L410;

	/* Pick the next value of DELTA after a trust region step. */
	if (vquad1 >= zero) {
		reason = "a trust region step has failed to reduce Q";
		status = NEWUOA_STEP_FAILED;
		goto done;
	}
	ratio = (f - fsave) / vquad1;
	if (debug) fprintf(stdout, " Ratio=%25.15E\n", (double)ratio);
	if (ratio <= tenth) delta = half * dnorm;
	else if (ratio <= 0.7) delta = MAX(half * delta, dnorm);
	else delta = MIN(MAX(2.0 * delta, 4.0 * dnorm), 1e10);
	if (delta <= rho * 1.5) delta = rho;

	/* Set KNEW to the index of the next interpolation point to be deleted. */
	temp = MAX(tenth * delta, rho);
	rhosq = temp * temp;
	ktemp = 0;
	detrat = zero;
	if (f >= fsave) {
		ktemp = kopt;
		detrat = one;
	}
	for (k = 1; k <= npt; ++k) {
		REAL hdiag = zero;
		for (j = 1; j <= nptm; ++j) {
			temp = one;
			if (j < idz) temp = -one;
			hdiag += temp * (zmat[k + j * npt] * zmat[k + j * npt]);
		}
		temp = ABS(beta * hdiag + vlag[k] * vlag[k]);
		distsq = zero;
		for (j = 1; j <= n; ++j) {
			REAL dist = xpt[k + j * npt] - xopt[j];
			distsq += dist * dist;
		}
		if (distsq > rhosq) {
			REAL dist = distsq / rhosq;
			temp *= dist * (dist * dist);
		}
		if (temp > detrat && k != ktemp) {
			detrat = temp;
			knew = k;
		}
	}
	if (knew == 0) goto L460;

	/* Update BMAT, ZMAT and IDZ, so that the KNEW-th interpolation point
	   can be moved. Then make this move, and update the quadratic model. */
L410:
	update(n, npt, &bmat[ndim + 1], &zmat[npt + 1], &idz, ndim, &vlag[1], beta, knew, &w[1]);
	model_update = 1;
	ih = 0;
	for (i = 1; i <= n; ++i) {
		for (m1 = 1; m1 <= mv; ++m1)
			v_temp[m1 - 1] = pqv[m1 + knew * mv - mv - 1] * xpt[knew + i * npt];
		for (j = 1; j <= i; ++j) {
			++ih;
			for (m1 = 1; m1 <= mv; ++m1)
				hqv[m1 + ih * mv - mv - 1] += v_temp[m1 - 1] * xpt[knew + j * npt];
		}
	}
	for (m1 = 1; m1 <= mv; ++m1)
		pqv[m1 + knew * mv - mv - 1] = zero;
	for (k = 1; k <= nptm; ++k) {
		if (zmat[knew + k * npt] != zero) {
			for (m1 = 1; m1 <= mv; ++m1)
				v_temp[m1 - 1] = diffv[m1 - 1] * zmat[knew + k * npt];
			if (k < idz) {
				temp = -temp;
				for (m1 = 1; m1 <= mv; ++m1)
					v_temp[m1 - 1] = -v_temp[m1 - 1];
			}
			for (j = 1; j <= npt; ++j)
				for (m1 = 1; m1 <= mv; ++m1)
					pqv[m1 + j * mv - mv - 1] += v_temp[m1 - 1] * zmat[j + k * npt];
		}
	}
	for (i = 1; i <= n; ++i) {
		xpt[knew + i * npt] = xnew[i];
		for (m1 = 1; m1 <= mv; ++m1)
			gqv[m1 + i * mv - mv - 1] += diffv[m1 - 1] * bmat[knew + i * ndim];
	}
	if (f < fsave) kopt = knew;

	/* If a trust region step has provided a sufficient decrease in F, then
	   branch for another trust region calculation. The case KSAVE>0 occurs
	   when the new function value was calculated by a model step. */
	if (f <= fsave + tenth * vquad1) goto L100;
	if (ksave > 0) goto L100;

	/* Alternatively, find out if the interpolation points are close enough
	   to the best point so far. */
	knew = 0;
L460:
	distsq = delta * 4. * delta;
	for (k = 1; k <= npt; ++k) {
		REAL sum = zero;
		for (j = 1; j <= n; ++j) {
			REAL dist = xpt[k + j * npt] - xopt[j];
			sum += dist * dist;
		}
		if (sum > distsq) {
			knew = k;
			distsq = sum;
		}
	}

	/* If KNEW is positive, then set DSTEP, and branch back for the next
	   iteration, which will generate a "model step". */
	if (knew > 0) {
		dstep = MAX(MIN(tenth * SQRT(distsq), half * delta), rho);
		dsq = dstep * dstep;
		goto L120;
	}
	if (ratio > zero) goto L100;

	/* Knew =-1, indicating \|d\| \le 1/2 rho */
	if (MAX(delta, dnorm) > rho) {
		if (knew == -1 && delta > dnorm) {
			knew = 0;
			goto L111;
		}
		goto L100;
	}

	/* The calculations with the current value of RHO are complete. Pick the
	   next values of RHO and DELTA. */
L490:
	if (rho > rhoend) {
		delta = half * rho;
		ratio = rho / rhoend;
		if (ratio <= 16.) rho = rhoend;
		else if (ratio <= 250.) rho = SQRT(ratio) * rhoend;
		else rho = tenth * rho;
		delta = MAX(delta, rho);
		//if (iprint >= 2) {
		//	if (iprint >= 3) fprintf(stdout, "\n");
		//	fprintf(stdout, "\n"
		//		"    New RHO =%11.4E "
		//		"    Number of function values =%6ld\n"
		//		"    Least value of F =%23.15E     "
		//		"    The corresponding X is:\n",
		//		(double)rho, (long)nf, (double)fopt);
		//	print_x(stdout, n, &xbase[1], &xopt[1]);
		//}
		if (iprint >= 2) {
			if (iprint >= 3) fprintf(stdout, "\n");
			fprintf(stdout, "\n"
				"    New RHO =%11.4E "
				"    Number of function values =%6ld\n"
				"    Least value of F =%23.15E\n",
				(double)rho, (long)nf, (double)fopt);
		}

		if (knew == -1 && delta > dnorm) {
			nfsav = nf;
			knew = 0;
			goto L111;
		}
		goto L90;
	}

	/* Return from the calculation, after another Newton-Raphson step, if
	   it is too short to have been tried before. */
	if (knew == -1) goto L290;
	status = NEWUOA_SUCCESS;
done:
	if (fopt <= f) {
		for (i = 1; i <= n; ++i)
			x[i] = xbase[i] + xopt[i];
		f = fopt;
	}
	if (iprint > 0) {
		//if (status == NEWUOA_SUCCESS) {
		//	fprintf(stdout, "\n"
		//		"    At the return from NEWUOA "
		//		"    Number of function values =%6ld\n"
		//		"    Least value of F =%23.15E     "
		//		"    The corresponding X is:\n",
		//		(long)nf, (double)f);
		//	print_x(stdout, n, &x[1], NULL);
		//	fprintf(stdout, " IterOpt=%12ld\n", (long)iteropt);
		//}
		if (status == NEWUOA_SUCCESS) {
			fprintf(stdout, "\n"
				"    At the return from NEWUOA "
				"    Number of function values =%6ld\n"
				"    Least value of F =%23.15E\n",
				(long)nf, (double)f);
			//fprintf(stdout, " IterOpt=%12ld\n", (long)iteropt);
		}
		else if (reason != NULL) print_error(reason);
	}

	/* Return current status. */
	return status;
} /* newuob_h */

/*---------------------------------------------------------------------------*/
/* NEWUOA_H PUBLIC FUNCTIONS */

int newuoa_h(const INTEGER n, const INTEGER npt, newuoa_dfovec* dfovec,
	void* data, REAL* x, const REAL rhobeg, const REAL rhoend,
	const INTEGER iprint, const INTEGER maxfun, REAL* w, const INTEGER mv)
{
	INTEGER ndim, np, nptm, ixb, ixo, ixn, ixp, igq, ihq, ipq, ibmat, izmat,
		id, ivl, iw, idiffv, iv_beg, iv_err, iv_opt, iv_base,
		iv_temp, iv_vquad, iwv, igqv, ihqv, ipqv, igqv_opt, ihd1;

	/* Partition the working space array, so that different parts of it can be
	   treated separately by the subroutine that performs the main calculation. */
	if (npt < n + 2 || npt > 2 * n + 1) {
		if (iprint > 0) print_error("NPT is not in the required interval [N+2, 2N+1]");
		return NEWUOA_BAD_NPT;
	}
	ndim = npt + n;
	np = n + 1;
	nptm = npt - np;
	ixb = 0; /* C-indices start at 0 */
	ixo = ixb + n;
	ixn = ixo + n;
	ixp = ixn + n;
	igq = ixp + n * npt;
	ihq = igq + n;
	ipq = ihq + n * np / 2;
	ibmat = ipq + npt;
	izmat = ibmat + ndim * n;
	id = izmat + npt * nptm;
	ivl = id + n;
	iw = ivl + ndim;
	idiffv = iw + 10 * ndim;
	iv_beg = idiffv + mv;
	iv_err = iv_beg + mv;
	iv_opt = iv_err + mv;
	iv_base = iv_opt + mv;
	iv_temp = iv_base + mv;
	iv_vquad = iv_temp + mv;
	iwv = iv_vquad + mv;
	igqv = iwv + mv * n;
	ihqv = igqv + mv * n;
	ipqv = ihqv + mv * n * (n + 1) / 2;
	igqv_opt = ipqv + mv * npt;
	ihd1 = igqv_opt + mv * n;

	/* The above settings provide a partition of W for subroutine NEWUOB_H. */

	return newuob_h(n, npt, dfovec, data, x, rhobeg, rhoend, iprint, maxfun,
		&w[ixb], &w[ixo], &w[ixn], &w[ixp], &w[igq],
		&w[ihq], &w[ipq], &w[ibmat], &w[izmat], ndim, &w[id],
		&w[ivl], &w[iw], &w[idiffv], &w[iv_beg], &w[iv_err], &w[iv_opt],
		&w[iv_base], &w[iv_temp], &w[iv_vquad], &w[iwv], &w[igqv],
		&w[ihqv], &w[ipqv], &w[igqv_opt], &w[ihd1], mv);
} /* newuoa_h */

const char* newuoa_reason(int status)
{
	switch (status) {
	case NEWUOA_ITERATE:
		return "caller is requested to evaluate the objective function";
	case NEWUOA_SUCCESS:
		return "algorithm converged";
	case NEWUOA_BAD_NPT:
		return "NPT is not in the required interval";
	case NEWUOA_ROUNDING_ERRORS:
		return "too much cancellation in a denominator";
	case NEWUOA_TOO_MANY_EVALUATIONS:
		return "maximum number of function evaluations exceeded";
	case NEWUOA_STEP_FAILED:
		return "trust region step has failed to reduce quadratic approximation";
	case NEWUOA_BAD_ADDRESS:
		return "illegal NULL address";
	case NEWUOA_CORRUPTED:
		return "corrupted or misused workspace";
	default:
		return "unknown status";
	}
}

/*---------------------------------------------------------------------------
   TESTING
   The Chebyquad test problem (Fletcher, 1965) for N = 2,4,6 and 8,
   with NPT = 2N+1.

   Test problem for NEWUOA,FIXME: the objective function being the sum of the
   reciprocals of all pairwise distances between the points P_I,
   I=1,2,...,M in two dimensions, where M=N/2 and where the components of
   P_I are X(2*I-1) and X(2*I). Thus each vector X of N variables defines
   the M points P_I. The initial X gives equally spaced points on a
   circle. Four different choices of the pairs (N,NPT) are tried, namely
   (10,16), (10,21), (20,26) and (20,41). Convergence to a local minimum
   that is not global occurs in both the N=10 cases. The details of the
   results are highly sensitive to computer rounding errors. The choice
   IPRINT=2 provides the current X and optimal F so far whenever RHO is
   reduced. The bound constraints of the problem require every component of
   X to be in the interval [-1,1]. */

static void dfovec_test(const INTEGER n, const INTEGER mv,
	const REAL* x, REAL* v_err, void* data)
{
	REAL y[10][10], sum;
	INTEGER i, j;

	for (j = 0; j < n; j++) {
		y[j][0] = 1.0;
		y[j][1] = 2.0 * x[j] - 1.0;
		for (i = 2; i <= mv; i++)
			y[j][i] = 2.0 * y[j][1] * y[j][i - 1] - y[j][i - 2];
	}
	for (i = 0; i < mv; i++) {
		sum = 0;
		for (j = 0; j < n; j++)
			sum += y[j][i];
		sum /= n;
		if (i % 2 == 0)
			sum += 1.0 / ((i + 1) * (i - 1));
		v_err[i] = sum;
	}
} /* dfovec_test */

void newuoa_h_test(void)
{
	const INTEGER nmax = 8, nptmax = 2 * nmax + 1, mmax = nmax + 1,
		nspace = (nptmax + 11) * (nptmax + nmax) + nmax * (5 * nmax + 11) / 2 + mmax * (nptmax + nmax * (nmax + 7) / 2 + 7);
	REAL w[nspace], x[nmax], v_err[mmax], rhobeg, rhoend, f;
	INTEGER i, n, npt, mv, iprint, maxfun;

	iprint = 2;
	maxfun = 5000;
	rhoend = 1e-6;
	for (n = 2; n <= nmax; n += 2) {
		npt = 2 * n + 1;
		mv = n + 1;
		for (i = 0; i < n; i++)
			x[i] = (REAL)(i + 1) / mv;
		rhobeg = x[0] * 0.2;
		fprintf(stdout, "\n\n    Results with N =%2d and NPT =%3d\n", (int)n, (int)npt);
		newuoa_h(n, npt, dfovec_test, NULL, x, rhobeg, rhoend, iprint, maxfun, w, mv);
		dfovec_test(n, mv, x, v_err, NULL);
		f = 0.0;
		for (i = 0; i < mv; i++)
			f += v_err[i] * v_err[i];
		fprintf(stdout, " Final function value, f_final=%25.16E\n", (double)f);
	}
} /* newuoa_h_test */

#ifdef TESTING

int
main(int argc, char* argv[])
{
	newuoa_h_test();
	return 0;
}

#endif /* TESTING */

/*---------------------------------------------------------------------------*/
