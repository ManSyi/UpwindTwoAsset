#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <set>
#include <filesystem>
#include <Windows.h>
#include <stdio.h>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct> 

#include "Support.h"
#include "Calibration.h"
#include "Steady_state.h"
#include "Kaplan.h"




using namespace std;

void
solve_model(std::ofstream& of, std::ifstream& ifs, const Model_Options& model)
{
	std::cout << "Initialize options..." << endl;
	Options options(model);
	options.init(ifs);
	std::cout << "Done." << endl;
	std::cout << "Initialize calibration..." << endl;
	Calibration cal(options);

	if (!solve_model(of, ifs, cal))
	{
		std::cout << "Solving fails!" << endl;
	}
	else
	{
		std::cout << "Solving succeeds." << endl;
	}
}

bool
solve_model(std::ofstream& of, std::ifstream& ifs, Calibration& cal)
{
	string info;
	chrono::high_resolution_clock::time_point start;
	chrono::high_resolution_clock::time_point end;
	chrono::duration<long long, nano> dur;
	SteadyState ss;
	string sfold;

	cal.init(ifs);
	std::cout << "Done." << endl;
	omp_set_num_threads(cal.size("num_thread"));
	switch (cal.options().model())
	{
	case Model::KAPLAN:
		TA::KAPLAN::set_parameters(cal, ss);
		ss.init(cal);
		break;
	default:
		break;
	}
	if (!cal.options().run().solve_ss)
	{
		std::cout << "\nRead steady state from files..." << endl;
		solve_ss_from_files(cal, ss, ifs);
	}
	else
	{
		std::cout << "\nSolve steady state..." << endl;
		start = chrono::high_resolution_clock::now();
		solve_ss(cal, ss);
		end = chrono::high_resolution_clock::now();
		dur = end - start;
		std::cout << "\nSolving steady state costs " << chrono::duration_cast<chrono::milliseconds>(dur).count() << " milliseconds" << endl;

		std::cout << "\nWriting calibrations to files..." << endl;

		if (!write_calibrations(cal, of))
		{
			cerr << "Open files fail!" << endl;
			return 0;
		}

		std::cout << "Done." << endl;
		std::cout << "Writing steady state to files..." << endl;
		if (!write_steadystate(ss, of))
		{
			cerr << "Undo" << endl;
			return 0;
		}
		std::cout << "Done." << endl;
	}

	return 1;
}



int
main()
{
	chrono::high_resolution_clock::time_point start;
	chrono::high_resolution_clock::time_point end;
	chrono::duration<long long, nano> dur;
	start = chrono::high_resolution_clock::now();
	ofstream of;
	ifstream ifs;
	of.setf(ios::scientific);
	of.precision(16);
	string info;

	filesystem::path base = "E:/c++-projects/HANK-2.0/results-upwind";
	if (!filesystem::exists(base))
	{
		base = filesystem::current_path();
	}
	filesystem::current_path(base);
	std::cout << "Select model..." << endl;
	Model_Options model;
	model.init(ifs);
	filesystem::create_directories(base += "/" + model.model_string() + "/" + model.model_experiment());
	filesystem::current_path(base);
	std::cout << "Solving model " << model.model_string() << " begins..." << endl; 
	std::cout << "Experiment of parameters: " << model.model_experiment() << endl;
	std::cout << "Current path: " << filesystem::current_path() << endl;
	solve_model(of, ifs, model);
	end = chrono::high_resolution_clock::now();
	dur = end - start;
	std::cout << "\nSolving the model cost " << chrono::duration_cast<chrono::milliseconds>(dur).count() << " milliseconds" << endl;
	return 0;
}