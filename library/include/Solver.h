#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_multimin.h>
#include "newuoa_h.h"
#include "Calibration.h"
#include "Steady_state.h"
typedef double (*Root_target)(double ra, void* params);

typedef int
(*Roots_target)(const gsl_vector* x, void* params, gsl_vector* f);

typedef double
(*Min_target)(const gsl_vector* x, void* params);

struct Root
{
	gsl_function* F;
	double low;
	double high;
	double eps;
	int maxiter;
};

struct Roots
{
	const gsl_multiroot_function& F;
	const gsl_vector* x_init;
	int size;
	double tols;
	int maxiter;
};

struct Mins
{
	const gsl_multimin_function& F;
	const gsl_vector* x_init;
	int size;
	double tols;
	int maxiter;
	double step_size;
};

int
solve_multi_dim_roots(Roots& inputs, bool display = 0);

int
solve_multi_dims_min(Mins& inputs, bool display = 0);

int
solve_one_dim_root(Root& inputs, double& root, bool display = 0);

int
print_state(int iter, gsl_multiroot_fsolver* s);

template <typename Derived>
void
solve_root(Derived& params, Root_target root_target, double& root, bool display = 0)
{
	gsl_function F = { root_target, &params };
	Root root_struct = { &F, params.low, params.high, params.tols, params.maxiter };
	solve_one_dim_root(root_struct, root, display);
}

template <typename Derived>
void
solve_roots(Derived& params, Roots_target roots_target, gsl_vector* x_init, bool display = 0)
{
	gsl_multiroot_function F = { roots_target, x_init->size, &params };
	Roots inputs = { F, x_init, x_init->size,params.tols_roots, params.maxiter_roots };
	solve_multi_dim_roots(inputs);
}

template <typename Derived>
void
solve_mins(Derived& params, Min_target min_target, gsl_vector* x_init, bool display = 0)
{
	gsl_multimin_function F = { min_target, x_init->size, &params };
	Mins inputs = { F, x_init, x_init->size, params.tols_mins, params.maxiter_mins, params.step_size };
	solve_multi_dims_min(inputs);
}

template <typename Derived>
void
iteration(Derived& data)
{
	std::chrono::high_resolution_clock::time_point tp1 = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::time_point tp2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<long long, std::nano> dur1 = tp2 - tp1;
	std::cout << data.name << " iterations begins..." << std::endl;
	std::string filename;
	std::streambuf* oldcout = nullptr;
	std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
	std::cout.precision(8);
	std::ofstream fout;
	switch (data.opt)
	{
	case Opt::Nelder_Mead:
	{
		std::cout << "Minimization using Nelder_Mead algorithm in GSL library." << std::endl;
		filename = data.name + "_iterations_Nelder_Mead.txt";
		fout.open(filename);
		oldcout = std::cout.rdbuf(fout.rdbuf());
		print_head(std::cout, data.init_map, data.target_map) << std::endl;
		gsl_vector* x_init = Nelder_Mead::init_parameters(data.init_map);
		solve_mins(data, data.min_target, x_init, data.display);

	}
	break;
	case Opt::ROOTS:
	{
		std::cout << "Root-finding using Hybrid algorithm in GSL library." << std::endl;
		filename = data.name + "_iterations_ROOTS.txt";
		fout.open(filename);
		oldcout = std::cout.rdbuf(fout.rdbuf());
		print_head(std::cout, data.init_map, data.target_map) << std::endl;
		gsl_vector* x_init = ROOTS::init_parameters(data.init_map);
		solve_roots(data, data.roots_target, x_init, data.display);

	}
	break;
	case Opt::DFNLS:
	{
		std::cout << "Minimization using derivative-free algorithms for the least-squares"
			<< " algorithm proposed by Powell (2006) and Zhang etc. (2010)." << std::endl;
		filename = data.name + "_iterations_DFNLS.txt";
		fout.open(filename);
		oldcout = std::cout.rdbuf(fout.rdbuf());
		double* y_init = DFNLS::init_parameters(data.init_map);
		long n = data.init_map.size();
		long npt = 2 * n + 1;
		long iprint = data.display;
		long maxfun = data.maxfun_multip * (n + 1);
		long mv = data.target_map.size();
		long nw = (npt + 11) * (npt + n) + n * (5 * n + 11) / 2
			+ mv * (npt + n * (n + 7) / 2 + 7);
		double* w = new double[nw];
		int status = 0;
		for (int i = 0; i < data.ndfls; ++i)
		{
			std::cout << "\nDFLS minimization attempt " << i + 1 << "\n";
			print_head(std::cout, data.init_map, data.target_map) << std::endl;
			status = newuoa_h(n, npt, data.dfls_target, &data, y_init, data.rhobeg, data.rhoend,
				iprint, maxfun, w, mv);
			std::printf("    Returned by %s\n", newuoa_reason(status));
		}
		delete[] y_init;
	}
	break;
	default:
		break;
	}

	tp2 = std::chrono::high_resolution_clock::now();
	dur1 = tp2 - tp1;
	std::cout << "\n\nIterations cost " << std::chrono::duration_cast<std::chrono::milliseconds>(dur1).count()
		<< " milliseconds" << std::endl;
	std::cout.rdbuf(oldcout);
}