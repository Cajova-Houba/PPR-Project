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
#include <map>

// vectors
#include <emmintrin.h>

// intel TBB
#include <tbb/tbb.h>

// opencl
#include <CL/cl.hpp>

//===============================================
// STRUCTURES AND CLASSES
//===============================================
// Declaration of functions used by ApplyEvolutionStep.
void pick_four_random(int* r1, int* r2, int* r3, int* r4);
void process_individual(TPassword& res, TPassword& individual, const TPassword& v1, const TPassword& v2, const TPassword& v3, const TPassword& v4, const TPassword& vb, const int curr_index, double * score_map);

//
// Functor for tbb::parallel_for. Performs 
// evolution step over population chunk.
//
class ApplyEvolutionStep {
public:
	ApplyEvolutionStep(TPassword population[], TPassword new_population[], TPassword& best_vector, double score_map[], const int best_individual_index) :
		population(population), new_population(new_population), best_vector(best_vector), score_map(score_map),  best_individual_index(best_individual_index)
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
				best_vector,
				best_individual_index,
				score_map);
		}
	}

private:
	TPassword *const population;
	TPassword *const new_population;
	const TPassword& best_vector;
	double * const score_map;
	const int best_individual_index;
};


//
// Structure for passing cl data which don't need to be 
// re-initialized every generation.
//
struct CL_DATA {
	cl::Platform platform;
	std::vector<cl::Device> devices;
	cl::Context* context = nullptr;
	cl::Program* program = nullptr;
};
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
// shoud be divisible by 64 if opencl is used.
const int NP = 416 * D;
//const int NP = 7;

// max number of generations
const double GENERATIONS = 500;
//===============================================



//===============================================
// PROGRAM CONSTANTS
//===============================================
//
// program control
// USE_PARALLEL -> TBB is used
// USE_PARALLEL + USE_OPENCL -> OpenCL is used
const bool USE_PARALLEL = true;
const bool USE_OPENCL = false;
const bool PRINT_KEY = false;
const bool PRINT_BEST = false;

// kernel for mutation and binomic cross
const std::string KERNEL_SOURCE =
"__kernel void mutation_and_cross("
"	__global const unsigned char *best,"
"	__global const unsigned char *v1,"
"	__global const unsigned char *v2,"
"	__global const unsigned char *v3,"
"	__global const unsigned char *v4,"
"	__global const unsigned char *individual,"
"	__global const double *cross_rand,"
"	__global const double *f,"
"	__global const double *cr,"
"	__global unsigned char *res) {"
""
"	unsigned char noise_vec_elem = 0;"
// get the index of the current element to be processed
"	const int i = get_global_id(0);"
// mutation
"		noise_vec_elem = best[i % 10] + (*f)*(v1[i] + v2[i] - v3[i] - v4[i]);"
""
// binomic cross"
"		if (cross_rand[i] <= (*cr)) {"
"			res[i] = noise_vec_elem;"
"		}"
"		else {"
"			res[i] = individual[i];"
"		}"
"}";

// Length of TBlock.
const int BLOCK_SIZE = 8;

// Max bit difference of reference block and decrypted block.
// Used in fitness function.
const double MAX_BLOCK_BIT_DIFF = BLOCK_SIZE * 8;


// parametry gaussovske cenove funkce
// G_A je max výška kopce
// G_S je šíøka kopce
const int G_A = 500;
const int G_S = 15;
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

//
//	Returns MAX_BLOCK_BIT_DIFF - # of different bits between reference 
//	block and decrypted block.
//
double fitness_bit_diff(unsigned char* individual) {
	double fit = 0;
	int i = 0;
	TBlock decrypted;
	SJ_context context;
	byte xorRes;
	std::bitset<BLOCK_SIZE> bs;

	makeKey(&context, individual, sizeof(TPassword));
	decrypt_block(&context, decrypted, *encrypted_block);

	for (i = 0; i < BLOCK_SIZE; i++) {
		xorRes = *reference_block[i] ^ decrypted[i];
		bs = (xorRes);
		fit += bs.count();
	}


	return MAX_BLOCK_BIT_DIFF - fit;
}


#pragma region evolution
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
//	Generate first population of DE algorithm. Fills score map
//	and returns index of the best individual.
//
int create_first_population(TPassword *population, double* score_map) {
	int i = 0;
	double best_f = 0;
	double tmp = 0;
	int bi = -1;

	for ( i = 0; i < NP; i++)
	{
		generate_individual(population[i]);
		tmp = fitness_bit_diff(population[i]);
		score_map[i] = tmp;
		if (bi == -1 || tmp > best_f) {
			best_f = tmp;
			bi = i;
		}
	}
	

	return bi;
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
void copy_individual(TPassword& dest, const unsigned char* src) {
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
void find_best_individual(TPassword* population, int* best_index, double* best_score) {
	int i = 0;
	double fitness = 0;
	double b_s = 0;
	int b_i = -1;

	for (i = 0; i < NP; i++)
	{
		fitness = fitness_bit_diff(population[i]);
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


//
//	Performs evolution of one individual. Result (either old individual or new individual) is copied to res.
//
void process_individual(TPassword& res, TPassword& individual, const TPassword& v1, const TPassword& v2, const TPassword& v3, const TPassword& v4, const TPassword& vb, const int curr_index, double * score_map) {
	TPassword noise_vec{};
	TPassword res_vec{};
	double individual_score = 0,
		new_score = 0;

	// fitness of the current individual
	individual_score = score_map[curr_index];

	// evolution = mutation + cross
	mutation_best_2(noise_vec, vb,
		v1, v2, v3, v4);
	binomic_cross(res_vec, individual, noise_vec);

	// use better one
	new_score = fitness_bit_diff(res_vec);
	if (new_score > individual_score) {
		copy_individual(res, res_vec);
		score_map[curr_index] = new_score;
	}
	else {
		copy_individual(res, individual);
	}
}


//
//	Performs evolution over population while using opencl. 
//	current_best_index is expected to point to the index of the best individual in current population.
//		After this function returns, value it points to will be set to the index of the best individual in new population. 
//	score_map is expected to contain fitnesses of the current population.
//		After this function returns, it will contain fitnesses of the new population.
//
//	Returns CL_SUCCESS if everything is OK, otherwise returns error num.
//
signed __int32 evolution_opencl(TPassword* population, TPassword* new_population, CL_DATA cl_data, int* current_best_index, double* score_map) {
	size_t i = 0, j = 0;

	// arrays for opencl buffers
	// all input data are converted to 1D arrays of unsigned char
	const size_t ARRAY_LEN = NP*D;
	const size_t BUFFER_LEN = ARRAY_LEN * sizeof(unsigned char);
	unsigned char res[ARRAY_LEN]{};
	unsigned char v1[ARRAY_LEN]{};
	unsigned char v2[ARRAY_LEN]{};
	unsigned char v3[ARRAY_LEN]{};
	unsigned char v4[ARRAY_LEN]{};
	double crs[ARRAY_LEN]{};
	//unsigned char best[ARRAY_LEN]{};
	cl_int err;
	TPassword tmp;

	// new best
	int new_best = -1;
	double new_best_score = 0;
	double tmp_score = 0;

	// best and random indexes
	int rand1, rand2, rand3, rand4;

	size_t global_item_size = NP;
	size_t local_item_size = 32;

	// score for comparing results
	double curr_score = 0, new_score = 0;

	if (PRINT_BEST) {
		print_key(population[*current_best_index], score_map[*current_best_index]);
	}

	// fill source arrays
	for (i = 0; i < NP; i++)
	{
		pick_four_random(&rand1, &rand2, &rand3, &rand4);
		for (j = 0; j < D; j++)
		{
			crs[i*D + j] = cross_distribution(generator);
			v1[i * 10 + j] = population[rand1][j];
			v2[i * 10 + j] = population[rand2][j];
			v3[i * 10 + j] = population[rand3][j];
			v4[i * 10 + j] = population[rand4][j];
		}
	}

	// prepare buffers for cl
	cl::Buffer res_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_LEN, res, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error while creating buffer for res." << std::endl;
		return err;
	}
	cl::Buffer pop_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_LEN, population, &err);
	cl::Buffer v1_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_LEN, v1, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error while creating buffer for v1." << std::endl;
		return err;
	}
	cl::Buffer v2_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_LEN, v2, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error while creating buffer for v2." << std::endl;
		return err;
	}
	cl::Buffer v3_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_LEN, v3, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error while creating buffer for v3." << std::endl;
		return err;
	}
	cl::Buffer v4_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BUFFER_LEN, v4, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error while creating buffer for v4." << std::endl;
		return err;
	}
	cl::Buffer best_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(TPassword*), population[*current_best_index], &err);
	cl::Buffer cr_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double*), (void *)&CR, &err);
	cl::Buffer f_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double*), (void *)&F, &err);
	cl::Buffer crs_buff(*cl_data.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ARRAY_LEN * sizeof(double), crs, &err);

	// create kernel
	cl::Kernel kernel(*cl_data.program, "mutation_and_cross", &err);
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

	// execute kernel
	// this performs mutation and binomic cross over whole population
	cl::CommandQueue queue(*cl_data.context, cl_data.devices[0], 0, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error while creating queue. " << std::endl;
		return err;
	}
	queue.enqueueNDRangeKernel(kernel,
		cl::NullRange,
		cl::NDRange(global_item_size),
		cl::NDRange(local_item_size));
	err = queue.finish();
	if (err != CL_SUCCESS) {
		std::cout << "Error while running the queue. " << std::endl;
		return err;
	}

	// read and process results
	err = queue.enqueueReadBuffer(res_buff, CL_TRUE, 0, BUFFER_LEN, res);
	if (err != CL_SUCCESS) {
		std::cout << "Error while reading results from queue. " << std::endl;
		return err;
	}
	queue.finish();
	for ( i = 0; i < NP; i++)
	{
		// (res[i*10]) is effectively same as TPassword
		// compare scores and decide what to keep in new population
		curr_score = score_map[i];
		new_score = fitness_bit_diff(&(res[i*10]));
		if (new_score > curr_score) {
			copy_individual(new_population[i], &(res[i*10]));
			score_map[i] = new_score;
			tmp_score = new_score;
		}
		else {
			copy_individual(new_population[i], population[i]);
			tmp_score = curr_score;
		}

		// also search for the best individual in new population
		if (new_best == -1 || new_best_score < tmp_score) {
			new_best = i;
			new_best_score = tmp_score;
		}
	}

	// update best individual index
	(*current_best_index) = new_best;

	return CL_SUCCESS;
}


//
//	Performs evolution over population array while using parallelism.
//
void evolution_parallel(TPassword* population, TPassword* new_population, int* current_best_index, double* score_map) {
	double best_fitness = 0;
	int best_index = 0;

	tbb::parallel_for(tbb::blocked_range<size_t>(0, NP), 
		ApplyEvolutionStep(population, new_population, population[*current_best_index], score_map, *current_best_index)
	);
}


//
//	Performs one cycle of evolution over given population. Encrypted and reference blocks are used
//	for calculating fitness function.
//	current_best_index is expected to point to the index of the best individual in current population.
//		After this function returns, value it points to will be set to the index of the best individual in new population. 
//	score_map is expected to contain fitnesses of the current population.
//		After this function returns, it will contain fitnesses of the new population.
//
//	New population is stored to new_population. 
//
void evolution(TPassword* population, TPassword* new_population, int* current_best_index, double* score_map) {
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
	double tmp_score = 0;

	if (PRINT_BEST) {
		print_key((population[*current_best_index]), score_map[*current_best_index]);
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

		
		mutation_best_2(noise_vector, population[*current_best_index], *randomly_picked_1, *randomly_picked_2, *randomly_picked_3, *randomly_picked_4);

		// cross
		//	1. y = cross noise_vector (v) with active individual (x_i) (by CR parameter)
		binomic_cross(crossed_vector, *active_individual, noise_vector);
		
		// choose new guy to population
		// evaluate fitness(y)
		// if (fitness(y) > fitness(active_individual)) -> y || active_individual
		new_score = fitness_bit_diff(crossed_vector);
		active_score = score_map[i];
		if (new_score > active_score) {
			// use crossed_vector for new population
			copy_individual(new_population[i], crossed_vector);
			score_map[i] = new_score;
			tmp_score = new_score;
		}
		else {
			// use active_score for new population
			copy_individual(new_population[i], *active_individual);
			tmp_score = active_score;
		}

		if (best_index == -1 || tmp_score > best_fitness) {
			best_index = i;
			best_fitness = tmp_score;
		}
	}

	*current_best_index = best_index;
}
//===============================================
#pragma endregion



//===============================================
// OTHER FUNCTIONS
//===============================================

//
//	Initialize OpenCL structure. Tries to load GPU device first 
//	and if no such device is found, uses first available. Returns CL_SUCCESS if everything
//	is ok, otherwise retrns error.
//
signed __int32 initialize_opencl(CL_DATA* data) {
	cl_int err;

	// platform and devices
	err = cl::Platform::get(&(data->platform));
	if (err != CL_SUCCESS) {
		std::cout << "Error while getting platform." << std::endl;
		return err;
	}
	err = data->platform.getDevices(CL_DEVICE_TYPE_ALL, &(data->devices));
	if (err != CL_SUCCESS) {
		std::cout << "Error while geting devices." << std::endl;
		return err;
	}
	if ((data->devices).empty()) {
		err = data->platform.getDevices(CL_DEVICE_TYPE_ALL, &(data->devices));
		if (err != CL_SUCCESS) {
			std::cout << "Error while geting devices." << std::endl;
			return err;
		}
		if ((data->devices).empty()) {
			std::cout << "No devices found." << std::endl;
			return CL_DEVICE_NOT_FOUND;
		}
	}

	// context
	data->context = new cl::Context(data->devices, NULL, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error while creating context." << std::endl;
		return err;
	}

	// program
	cl::Program::Sources source(1, std::make_pair(KERNEL_SOURCE.c_str(), KERNEL_SOURCE.length()));
	data->program = new cl::Program(*(data->context), source, &err);
	err |= data->program->build(data->devices);
	if (err != CL_SUCCESS) {
		std::cout << "Error while creating and building program." << std::endl;
		return err;
	}

	return CL_SUCCESS;
}


// 
// Clean up OpenCL data.
//
void clean_opencl(CL_DATA* data) {
	if (data != nullptr) {
		if (data->context != nullptr) {
			delete data->context;
		}

		if (data->program != nullptr) {
			delete data->program;
		}
	}
}


//
//	Cipher breaker.
//
bool break_the_cipher(TBlock &encrypted, const TBlock &reference, TPassword &password) {
	int generation = 0;
	bool done = false;

	// two array for population and
	// pointers to switch between them 
	// each evolution loop
	TPassword population1[NP] {0};
	TPassword population2[NP] {0};
	TPassword *pop = population1;
	TPassword *new_pop = population2;

	// for openCL
	CL_DATA cl_data;
	cl_int res;

	// fitness of the best individual (in current generation)
	double score_map[NP]{};
	int best_individual_index = 0;

	// fitness of the best individual in whole evolution
	double global_best = 0;

	// set global encrypted block and reference block
	// as those are not going to change
	encrypted_block = &encrypted;
	reference_block = &reference;

	// initialize OpenCL if needed
	if (USE_PARALLEL && USE_OPENCL) {
		res = initialize_opencl(&cl_data);
		if (res != CL_SUCCESS) {
			std::cout << "Failed to initialize opencl data." << std::to_string(res) << std::endl;
			clean_opencl(&cl_data);
			return false;
		}
	}

	// generate first population
	best_individual_index = create_first_population(pop, score_map);
	global_best = score_map[best_individual_index];

	// keep evolving till the key is guessed or max number of generations is reached
	while (generation < GENERATIONS && !done) {

		// if the best_fitness == MAX_BLOCK_BIT_DIFF, it means that the 
		// block decrypted by individual was same as the reference block
		if (std::abs(score_map[best_individual_index] - MAX_BLOCK_BIT_DIFF) < 0.001) {
			done = true;
			break;
		}
		
		if (USE_PARALLEL && USE_OPENCL) {
			res = evolution_opencl(pop, new_pop, cl_data, &best_individual_index, score_map);
			if (res != CL_SUCCESS) {
				std::cout << "Error encoutered in generation " << std::to_string(generation) << std::endl;
				break;
			}
		}
		else if (USE_PARALLEL) {
			evolution_parallel(pop, new_pop, &best_individual_index, score_map);
		}
		else {
			evolution(pop, new_pop, &best_individual_index, score_map);
		}
		
		// global best
		if (score_map[best_individual_index] > global_best) {
			global_best = score_map[best_individual_index];
		}

		generation++;
	}

	if (USE_OPENCL) {
		clean_opencl(&cl_data);
	}

	if (done) {
		std::cout << std::to_string(generation) << ";" << std::to_string(global_best) << ";";
	}
	else {
		std::cout << ";" << std::to_string(global_best) << ";";
	}

	return done;
}
//===============================================