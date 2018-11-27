#include "../breaker.h"

//V ramci vypracovani semestralni prace muzete menit pouze a jedine tento souboru od teto radky dale.


#include <memory>
#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <bitset>
#include <cstdio>
#include <functional>

// vectors
#include <emmintrin.h>

// intel TBB
#include <tbb/tbb.h>

// opencl
#include <CL/cl.hpp>

// Declaration of functions used by ApplyEvolutionStep.
void pick_four_random(int* r1, int* r2, int* r3, int* r4);
void process_individual(TPassword& res, TPassword& individual, const TPassword& v1, const TPassword& v2, const TPassword& v3, const TPassword& v4, const TPassword& vb);

//
// Functor for tbb::parallel_for. Performs 
// evolution step over population chunk.
//
class ApplyEvolutionStep {
public:
	ApplyEvolutionStep(TPassword population[], TPassword new_population[], TPassword& best_vector) :
		population(population), new_population(new_population), best_vector(best_vector)
	{}

	void operator() (const tbb::blocked_range<size_t>& r) const {
		int rand1, rand2, rand3, rand4;
		TPassword* randomly_picked_1;
		TPassword* randomly_picked_2;
		TPassword* randomly_picked_3;
		TPassword* randomly_picked_4;
		TPassword* pop = population;
		TPassword* new_pop = new_population;

		for (size_t i = r.begin(); i != r.end(); i++)
		{
			pick_four_random(&rand1, &rand2, &rand3, &rand4);
			randomly_picked_1 = &(population[rand1]);
			randomly_picked_2 = &(population[rand2]);
			randomly_picked_3 = &(population[rand3]);
			randomly_picked_4 = &(population[rand4]);

			process_individual(new_pop[i], pop[i],
				*randomly_picked_1,
				*randomly_picked_2,
				*randomly_picked_3,
				*randomly_picked_4,
				best_vector);
		}
	}

private:
	TPassword *const population;
	TPassword *const new_population;
	const TPassword& best_vector;
};



//===============================================
// PROGRAM CONSTANTS
//===============================================
//
// program control
const bool USE_PARALLEL = true;
const bool USE_OPENCL = true;
const bool PRINT_KEY = false;
const bool PRINT_BEST = false;

// passwords of various length for testing
// hex form of full password: 0x68 0x65 0x6C 0x6C 0x6F 0x31 0x32 0x34 0x35
// todo: remove before submit
const TPassword reference_password_1{ 'h', 0, 0, 0, 0, 0, 0, 0, 0, 0 };
const TPassword reference_password_2{ 'h', 'e', 0, 0, 0, 0, 0, 0, 0, 0 };
const TPassword reference_password_3{ 'h', 'e', 'l', 0, 0, 0, 0, 0, 0, 0 };
const TPassword reference_password_4{ 'h', 'e', 'l', 'l', 0, 0, 0, 0, 0, 0 };
const TPassword reference_password_5{ 'h', 'e', 'l', 'l', 'o', 0, 0, 0, 0, 0 };
const TPassword reference_password_6{ 'h', 'e', 'l', 'l', 'o', '1', 0, 0, 0, 0 };
const TPassword reference_password_7{ 'h', 'e', 'l', 'l', 'o', '1', '2', 0, 0, 0 };
const TPassword reference_password_8{ 'h', 'e', 'l', 'l', 'o', '1', '2', '3', 0, 0 };
const TPassword reference_password_9{ 'h', 'e', 'l', 'l', 'o', '1', '2', '3', '4', 0 };
const TPassword reference_password_10{ 'h', 'e', 'l', 'l', 'o', '1', '2', '3', '4', '5' };
const TPassword* reference_password = &reference_password_10;

// custom fitness function constants
const double MAX_BIT_DIFF = 10 * 8;

// length of TBlock
const int BLOCK_SIZE = 8;

// parametry gaussovske cenove funkce
// G_A je max výška kopce
// G_S je šíøka kopce
const int G_A = 500;
const int G_S = 15;
//===============================================



//===============================================
// DIFFERENTIAL EVOLUTION CONSTANTS
//===============================================
//
// mutation constant, recommended values: 0.3 - 0.9
// should be in interval [0, 2]
const double F = 0.3;

// cross constant, recommended values: 0.8 - 0.9
// low value -> more crossing
const double CR = 0.5;

// dimension of one individual = password length
const int D = 10;

// population size, recommended: 10*D - 100*D
//const int NP = 420 * D;
const int NP = 5;

// max number of generations
const double GENERATIONS = 800;
//===============================================



//===============================================
// OTHER GLOBAL VARIABLES
//===============================================
// random number generator
std::random_device trng;
std::mt19937 generator(trng());
std::uniform_int_distribution<int> param_value_distribution(0, 255);
std::uniform_int_distribution<int> population_picker_distribution(0, NP);
std::uniform_real_distribution<double> cross_distribution(0, 1);

// encrypted block
TBlock* encrypted_block;

// reference block
const TBlock* reference_block;
//===============================================



//===============================================
// PRINT FUNCTIONS
//===============================================
// todo: remove before submit

// Prints one TBlock in hex.
void print_block(const TBlock& block) {
	int i = 0;
	for (i = 0; i < BLOCK_SIZE; i++) {
		printf("%#x ", block[i]);
	}
	std::cout << std::endl;
}

// Prints TPassword and its' score.
void print_key(const TPassword& key, const double score) {
	int i = 0;
	for (i = 0; i < D; i++) {
		printf("%#x ", key[i]);
	}
	std::cout << "\t;" << score << std::endl;
}
//===============================================



//===============================================
// EVOLUTION FUNCTIONS
//===============================================

#pragma region fitness_functions
// todo: remove unneccessary fitness functions before submit.
//
// # of different bits.
//
double fitness_bit_diff(TPassword& individual) {
	double fit = 0;
	int i = 0;
	TBlock decrypted;
	SJ_context context;
	byte xorRes;
	std::bitset<8> bs;

	makeKey(&context, individual, sizeof(TPassword));
	decrypt_block(&context, decrypted, *encrypted_block);

	for (i = 0; i < BLOCK_SIZE; i++) {
		xorRes = *reference_block[i] ^ decrypted[i];
		bs = (xorRes);
		fit += bs.count();
	}


	return fit;
}
#pragma endregion


#pragma region evolution_helpers
//
//	Randomly gnerates one guy in population.
//
void generate_individual(TPassword& guy) {
	byte i = 0;

	for (i = 0; i < D; i++)
	{
		guy[i] = (byte)param_value_distribution(generator);
	}
}

//
//	Generate first population of DE algorithm.
//
void create_first_population(TPassword *population) {
	int i = 0;

	for ( i = 0; i < NP; i++)
	{
		generate_individual(population[i]);
	}
}

//
//	Same as mutation_best_2_vec but doesn't use vectors internaly.
//
void mutation_best_2(TPassword& res, const TPassword& best, const TPassword& vec1, const TPassword& vec2, const TPassword& vec3, const TPassword& vec4) {
	int i = 0;

	 for (i = 0; i < D; i++) {
		 res[i] = (byte)(best[i] + F * (vec1[i] + vec2[i] - vec3[i] - vec4[i]));
	 }
}

//
//	Binomic cross of two individuals. For each element, random number is generated. 
//	If its <= CR, then noise_vector element, otherwise active_individual element is used.
//
void binomic_cross(TPassword& res, const TPassword& active_individual, const TPassword& noise_vector) {
	double random_val = 0;
	int i = 0;
	for (i = 0; i < D; i++) {
		random_val = cross_distribution(generator);
		if (random_val <= CR) {
			res[i] = noise_vector[i];
		}
		else {
			res[i] = active_individual[i];
		}
	}
}

//
//	Copies elements from src to dest.
//
void copy_individual(TPassword& dest, const TPassword& src) {
	int i = 0;
		std::copy(src, src + D, dest);
}

//
//	Compares two individuals and returns true if they're same.
//
bool compare_individuals(const TPassword& ind1, const TPassword& ind2) {
	int i = 0;
	bool same = true;
	
	while (i < D && same) {
		same = ind1[i] == ind2[i];
		i++;
	}

	return same;
}

//
//	Finds the best individual in the population and returns its index and score.
//
void find_best_individual(TPassword* population, const std::function<double(TPassword&)> fitness_function, int* best_index, double* best_score) {
	int i = 0;
	double fitness = 0;
	double b_s = 0;
	int b_i = -1;

	for (i = 0; i < NP; i++)
	{
		fitness = fitness_function(population[i]);
		if (b_i == -1 || fitness > b_s) {
			b_i = i;
			b_s = fitness;
		}
	}

	*best_index = b_i;
	*best_score = b_s;
}

//
//	Picks 4 different indexes in population.
//
void pick_four_random(int* r1, int* r2, int* r3, int* r4) {
	int rand1, rand2, rand3, rand4;
	rand1 = population_picker_distribution(generator);
	rand2 = rand1;
	while (rand1 == rand2) {
		rand2 = population_picker_distribution(generator);
	}
	rand3 = rand2;
	while (rand2 == rand3) {
		rand3 = population_picker_distribution(generator);
	}
	rand4 = rand3;
	while (rand3 == rand4) {
		rand4 = population_picker_distribution(generator);
	}

	*r1 = rand1;
	*r2 = rand2;
	*r3 = rand3;
	*r4 = rand4;
}
#pragma endregion


//
// Performs evolution of one individual. Result (either old individual or new individual) is copied to res.
//
void process_individual(TPassword& res, TPassword& individual, const TPassword& v1, const TPassword& v2, const TPassword& v3, const TPassword& v4, const TPassword& vb) {
	TPassword noise_vec{};
	TPassword res_vec{};
	double individual_score = 0,
		new_score = 0;

	// fitness of the current individual
	individual_score = fitness_bit_diff(individual);

	// evolution = mutation + cross
	mutation_best_2(noise_vec, vb,
		v1, v2, v3, v4);
	binomic_cross(res_vec, individual, noise_vec);

	// use better one
	new_score = fitness_bit_diff(res_vec);
	if (new_score > individual_score) {
		copy_individual(res, res_vec);
	}
	else {
		copy_individual(res, individual);
	}
}

//
// Performs evolution over population while using opencl
//
void evolution_opencl(TPassword* population, TPassword* new_population) {
	TPassword res[NP]{};
	TPassword v1[NP]{};
	TPassword v2[NP]{};
	TPassword v3[NP]{};
	TPassword v4[NP]{};
	double crs[NP]{};
	TPassword* best;
	cl_int err;

	// fill source arrays

	// prepare it for cl
	cl::Context ctx;
	cl::Program prg;
	cl::Buffer res_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NP * sizeof(TPassword), res, &err);
	cl::Buffer pop_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NP * sizeof(TPassword), population, &err);
	cl::Buffer v1_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NP * sizeof(TPassword), v1, &err);
	cl::Buffer v2_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NP * sizeof(TPassword), v2, &err);
	cl::Buffer v3_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NP * sizeof(TPassword), v3, &err);
	cl::Buffer v4_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NP * sizeof(TPassword), v4, &err);
	cl::Buffer best_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(TPassword), best, &err);
	cl::Buffer cr_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double*), (void *)&CR, &err);
	cl::Buffer f_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double*), (void *)&F, &err);
	cl::Buffer crs_buff(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NP * sizeof(double), crs, &err);

	// create kernel
	cl::Kernel kernel(prg, "mutation_and_cross", &err);
	kernel.setArg(0, best_buff);
	kernel.setArg(1, v1_buff);
	kernel.setArg(2, v2_buff);
	kernel.setArg(3, v3_buff);
	kernel.setArg(4, v4_buff);
	kernel.setArg(5, pop_buff);
	kernel.setArg(6, crs_buff);
	kernel.setArg(7, f_buff);
	kernel.setArg(8, cr_buff);
	kernel.setArg(9, res_buff);
}

//
//	Performs evolution over population array while using parallelism.
//
void evolution_parallel(TPassword* population, TPassword* new_population, TBlock& encrypted, const TBlock& reference, const std::function<double(TPassword&)> fitness_function) {
	size_t i = 0;
	double best_fitness = 0;
	int best_index = 0;

	find_best_individual(population, fitness_function, &best_index, &best_fitness);

	tbb::parallel_for(tbb::blocked_range<size_t>(0, NP), 
		ApplyEvolutionStep(population, new_population, population[best_index])
	);
}

//
//	Performs one cycle of evolution over given population. Encrypted and reference blocks are used
//	for calculating fitness function.
//
//	New population is stored to new_population. 
//
void evolution(TPassword* population, TPassword* new_population, TBlock& encrypted, const TBlock& reference, const std::function<double(TPassword&)> fitness_function) {
	int i = 0;
	int rand1, rand2, rand3, rand4;
	TPassword* active_individual;
	TPassword* randomly_picked_1;
	TPassword* randomly_picked_2;
	TPassword* randomly_picked_3;
	TPassword* randomly_picked_4;
	TPassword diff_vector{ 0 };
	TPassword noise_vector{ 0 };
	TPassword crossed_vector{ 0 };
	double new_score = 0;
	double active_score = 0;
	int best_index = -1;
	double best_fitness = 0;


	// find best individual in current population
	// possible optimization here
	find_best_individual(population, fitness_bit_diff, &best_index, &best_fitness);

	if (PRINT_BEST) {
		print_key((population[best_index]), best_fitness);
	}

	for (i = 0; i < NP; i++)
	{

		// active individual population[i]
		active_individual = &(population[i]);

		pick_four_random(&rand1, &rand2, &rand3, &rand4);
		randomly_picked_1 = &(population[rand1]);
		randomly_picked_2 = &(population[rand2]);
		randomly_picked_3 = &(population[rand3]);
		randomly_picked_4 = &(population[rand4]);

		
		mutation_best_2(noise_vector, population[best_index], *randomly_picked_1, *randomly_picked_2, *randomly_picked_3, *randomly_picked_4);

		// cross
		//	1. y = cross noise_vector (v) with active individual (x_i) (by CR parameter)
		binomic_cross(crossed_vector, *active_individual, noise_vector);
		
		// choose new guy to population
		// evaluate fitness(y)
		// if (fitness(y) > fitness(active_individual)) -> y || active_individual
		new_score = fitness_bit_diff(crossed_vector);
		active_score = fitness_bit_diff(*active_individual);
		if (new_score > active_score) {
			// use crossed_vector for new population
			copy_individual(new_population[i], crossed_vector);
		}
		else {
			// use active_score for new population
			copy_individual(new_population[i], *active_individual);
		}
	}
}
//===============================================



// Cipher breaker.
bool break_the_cipher(TBlock &encrypted, const TBlock &reference, TPassword &password) {
	SJ_context context;
	TBlock decrypted;
	TPassword testing_key{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	int i = 0, j=0;
	int generation = 0;
	bool done = false;
	TPassword population[NP] {0};
	TPassword new_population[NP] {0};
	TPassword *current_population_array;
	//std::function<double(TPassword&)> fitness_lambda = [](TPassword& psw) {return fitness_custom_linear(psw, *reference_password); };
	std::function<double(TPassword&)> fitness_lambda = [](TPassword& psw) {return fitness_bit_diff(psw); };
	//std::function<double(TPassword&)> fitness_lambda = [&encrypted, &reference](TPassword& psw) {return fitness(psw, encrypted, reference); };
	encrypted_block = &encrypted;
	reference_block = &reference;


	// pointer to evolution function
	void(*evolution_function)(TPassword*, TPassword*, TBlock&, const TBlock&, const std::function<double(TPassword&)>);

	if (USE_PARALLEL) {
		evolution_function = evolution_parallel;
	}
	else {
		evolution_function = evolution;
	}


	//std::cout << "Creating first population " << std::endl;
	create_first_population(population);
	//std::cout << "First population created " << std::endl;

	/*for (testing_key[0] = 0; testing_key[0] < 255; testing_key[0]++) {
		print_key(testing_key, fitness_lambda(testing_key));
	}

	return false;*/

	// keep evolving till the key is guessed or max number of generations is reached
	while (generation < GENERATIONS && !done) {

		// try every key in new population
		// generation mod 2 ifs are used to change between population arrays 
		for (i = 0; i < NP; i++) {

			if (generation % 2 == 0) {
				current_population_array = population;
			}
			else {
				current_population_array = new_population;
			}

			if (PRINT_KEY) {
				print_key(current_population_array[i], fitness_lambda(current_population_array[i]));
			}
			makeKey(&context, current_population_array[i], sizeof(TPassword));

			// if the decryption is successfull => bingo
			decrypt_block(&context, decrypted, encrypted);
			if (memcmp(decrypted, reference, sizeof(TBlock)) == 0) {
				memcpy(password, testing_key, sizeof(TPassword));
				copy_individual(password, current_population_array[i]);
				//print_key(current_population_array[i], fitness_lambda(current_population_array[i]));
				//std::cout << "Done in " << std::to_string(generation) << "!" << std::endl;
				std::cout << std::to_string(generation) << ";";
				done = true;
				break;
			}
		}

		// use current array as source of evolution and old array as destination of evolution
		if (generation % 2 == 0) {
			evolution_function(population, new_population, encrypted, reference, fitness_lambda);
		}
		else {
			evolution_function(new_population, population, encrypted, reference, fitness_lambda);
		}
		
		//if (generation % 5 == 0) {
			//std::cout << "Generation: " << std::to_string(generation) << std::endl;
		//}

		generation++;
	}

	return done;
}