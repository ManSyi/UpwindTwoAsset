
#include "Solver.h"
int
solve_one_dim_root(Root& inputs, double& root, bool display)
{
	int status = 0;
	int iter = 0;
	const gsl_root_fsolver_type* T;
	gsl_root_fsolver* s;

	T = gsl_root_fsolver_brent;
	s = gsl_root_fsolver_alloc(T);
	double eps = inputs.eps;
	int maxiter = inputs.maxiter;
	double low = inputs.low;
	double high = inputs.high;
	

	gsl_root_fsolver_set(s, inputs.F, low, high);

	if (display)
	{
		printf("using %s method\n", gsl_root_fsolver_name(s));

		printf("%5s [%9s, %9s] %9s %9s\n",
			"iter", "lower", "upper", "root", "err(est)");
	}

	do
	{
		++iter;
		status = gsl_root_fsolver_iterate(s);
		root = gsl_root_fsolver_root(s);
		low = gsl_root_fsolver_x_lower(s);
		high = gsl_root_fsolver_x_upper(s);
		status = gsl_root_test_interval(low, high,
			0, eps);

		if (status == GSL_SUCCESS && display)
			printf("Converged\n");

		if(display)		
			printf("%5d [%.7f, %.7f] %.7f %.7f\n",
			iter, low, high,
			root, high - low);


	} while (status == GSL_CONTINUE && iter < maxiter);

	gsl_root_fsolver_free(s);
	return status;
}

int
solve_multi_dim_roots(Roots& inputs, bool display)
{
	const gsl_multiroot_fsolver_type* T;
	gsl_multiroot_fsolver* s;

	int status;
	int iter = 0;

	const int n = inputs.size;
	gsl_multiroot_function f = inputs.F;

	T = gsl_multiroot_fsolver_hybrids;
	s = gsl_multiroot_fsolver_alloc(T, n);
	gsl_multiroot_fsolver_set(s, &f, inputs.x_init);
	if(display)
		print_state(iter, s);

	do
	{
		++iter;
		status = gsl_multiroot_fsolver_iterate(s);

		if (display)
			print_state(iter, s);

		if (status)   /* check if solver is stuck */
			break;

		status =
			gsl_multiroot_test_residual(s->f, inputs.tols);
	} while (status == GSL_CONTINUE && iter < inputs.maxiter);

	if(display)
		printf("status = %s\n", gsl_strerror(status));

	gsl_multiroot_fsolver_free(s);

	return 0;
}

int
solve_multi_dims_min(Mins& inputs, bool display)
{
	const gsl_multimin_fminimizer_type* T =
		gsl_multimin_fminimizer_nmsimplex2;
	gsl_multimin_fminimizer* s = NULL;

	size_t iter = 0;
	int status;

	double size = 0.0;
	const int n = inputs.size;
	gsl_vector* ss = gsl_vector_alloc(n);
	gsl_vector_set_all(ss, inputs.step_size);

	s = gsl_multimin_fminimizer_alloc(T, n);
	gsl_multimin_function F = inputs.F;
	gsl_multimin_fminimizer_set(s, &F, inputs.x_init, ss);
	int i = 0;
	do
	{
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);

		if (status)
			break;

		size = gsl_multimin_fminimizer_size(s);
		status = gsl_multimin_test_size(size, inputs.tols);

		if (status == GSL_SUCCESS && display)
		{
			printf("converged to minimum at\n");
		}
		if (display)
		{
			printf("iter = %3d x = ", iter);
			for (i = 0; i != n; ++i)
			{
				printf("% .3f ", gsl_vector_get(s->x, i));
			}
			printf("f(x) = % .3e\n", s->fval);
		}
	} while (status == GSL_CONTINUE && iter < inputs.maxiter);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);
	return status;
}

int
print_state(int iter, gsl_multiroot_fsolver* s)
{
	int size = s->x->size;
	int i = 0;
	printf("iter = %3d x = ", iter);
	for (i = 0; i != size; ++i)
	{
		printf("% .3f ", gsl_vector_get(s->x, i));
	}
	printf("f(x) = ");
	for (i = 0; i != size; ++i)
	{
		printf("% .3e ", gsl_vector_get(s->f, i));
	}
	printf("\n");
	//printf("iter = %3u x = % .3f % .3f "
	//	"f(x) = % .3e % .3e %\n",
	//	iter,
	//	gsl_vector_get(s->x, 0),
	//	gsl_vector_get(s->x, 1),
	//	gsl_vector_get(s->f, 0),
	//	gsl_vector_get(s->f, 1));
	return 0;
}