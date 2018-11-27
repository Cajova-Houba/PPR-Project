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
const bool PRINT_KEY = false;
const bool PRINT_BEST = false;

// passwords of various length for testing
// hex form of full password: 0x68 0x65 0x6C 0x6C 0x6F 0x31 0x32 0x34 0x35
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
const int NP = 420 * D;

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
//
// # of different bits.
//
double fitness_diff(TPassword& individual) {
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
/*
	This fitness function uses gaussian curve with the top being the best individual.

	the function has following format:
	fit = a * e^( 
		-(v1-u1)^2 / (2*s^2) 
		-(v2-u2)^2 / (2*s^2)
		-(v3-u3)^2 / (2*s^2)
		...
	)
	Where a and s are standard parameters of gaussian function, v1..vn are components of individual which
	is being tested and u1..un are components of reference block.
*/
double fitness_gauss(TPassword& individual, TBlock& encrypted, const TBlock& reference) {
	double fit = 0;
	int i = 0;
	TBlock decrypted;
	SJ_context context;

	makeKey(&context, individual, sizeof(TPassword));
	decrypt_block(&context, decrypted, encrypted);

	for (i = 0; i < 8; i++)
	{
		fit += -std::pow(decrypted[i] - reference[i], 2) / (2 * G_S*G_S);
	}

	return G_A * std::exp(fit);
}

/*
	Bit difference of individual from reference password. The lower the number, the bigger the difference.
*/
double fitness_custom_bit_diff(TPassword& individual) {
	return fitness_diff(individual);
	/*double fit = 0;
	int i = 0;
	byte xorRes;
	std::bitset<8> bs;

	for (i = 0; i < D; i++) {
		xorRes = individual[i] ^ (*reference_password)[i];
		bs = (xorRes);
		fit += bs.count();
	}


	return MAX_BIT_DIFF - fit;*/
}

/*
	Decrypt encrypted block using individual as key and compare it with encrypted.
	If it's close to reference, small value is returned, otherwise big value
	is returned.
*/
double fitness(TPassword& individual, TBlock& encrypted, const TBlock &reference) {
	double fit = 0;
	int i = 0;
	TBlock decrypted;
	SJ_context context;

	makeKey(&context, individual, sizeof(TPassword));
	decrypt_block(&context, decrypted, encrypted);

	for (i = 0; i < BLOCK_SIZE; i++) {
		fit += (reference[i] - decrypted[i])*(reference[i] - decrypted[i]);
	}

	fit = std::sqrt(fit);

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

	// following loop is replaced with vector library
	 for (i = 0; i < D; i++) {
		 float tmp = (best[i] + F * (vec1[i] + vec2[i] - vec3[i] - vec4[i]));
		 int r = (int)tmp;
		 res[i] = (byte)r;
	 }
}

//
//	Provede mutaci best-2.
//	diff_vec = (vec1 + vec2 - vec3 - vec4) * F
//	noise_vec = diff_vec + best
//
//	Kde vec1 ... vec2 jsou nahodne vybrane, nestejne prvky z populace a best je nejlepsi jedinec ze soucaasne populace.
//
void mutation_best_2_my_vec(TPassword& res, const TPassword& best, const TPassword& vec1, const TPassword& vec2,  const TPassword& vec3, const TPassword& vec4) {
	//__m128i vb{}, v1{}, v2{}, v3{}, v4{};
	//__m128i res_vec{};
	//__m128 tmp1{}, tmp2{}, tmp3;

	////// help vector with F constant for multiplying
	//float f_vec[]{ F,F,F,F };
	//__m128 f_vector = _mm_loadu_ps(f_vec);

	//// load stuff
	//vb = _mm_loadu_si128((const __m128i*)best);
	//v1 = _mm_loadu_si128((const __m128i*)vec1);
	//v2 = _mm_loadu_si128((const __m128i*)vec2);
	//v3 = _mm_loadu_si128((const __m128i*)vec3);
	//v4 = _mm_loadu_si128((const __m128i*)vec4);

	//res_vec = _mm_adds_epi8(v1, v2);
	//res_vec = _mm_subs_epi8(res_vec, v3);
	//res_vec = _mm_subs_epi8(res_vec, v4);

	//tmp1 = _mm_loadu()

	size_t i = 0;

	// tmp arrays for working with bytes as floats
	__int32 v1f[12] {};
	__int32 v2f[12] {};
	__int32 v3f[12] {};
	__int32 v4f[12] {};
	float vbf[12] {};

	// __m128 is 128 bit length vector which can take 4 32-bit floats
	// lower 4, mid 4 and high 4
	__m128i res_v1{}, res_v2{}, res_v3{},
		v1_1{}, v1_2{}, v1_3{},
		v2_1{}, v2_2{}, v2_3{},
		v3_1{}, v3_2{}, v3_3{},
		v4_1{}, v4_2{}, v4_3{};

	__m128 tmp_1{}, tmp_2{}, tmp_3{},
		vb_1{}, vb_2{}, vb_3{};

	// help vector with F constant for multiplying
	float f_vec[]{ F,F,F,F };
	__m128 f_vector = _mm_loadu_ps(f_vec);

	// first, convert from unsigned char to float
	for (i = 0; i < D; i++)
	{
		v1f[i] = (__int32)vec1[i];
		v2f[i] = (__int32)vec2[i];
		v3f[i] = (__int32)vec3[i];
		v4f[i] = (__int32)vec4[i];
		vbf[i] = (float)best[i];
	}

	// now load it to vectors
	v1_1 = _mm_loadu_si128((const __m128i*)v1f);
	v1_2 = _mm_loadu_si128((const __m128i*)(v1f + 4));
	v1_3 = _mm_loadu_si128((const __m128i*)(v1f + 8));
	v2_1 = _mm_loadu_si128((const __m128i*)v2f);
	v2_2 = _mm_loadu_si128((const __m128i*)(v2f + 4));
	v2_3 = _mm_loadu_si128((const __m128i*)(v2f + 8));
	v3_1 = _mm_loadu_si128((const __m128i*)v3f);
	v3_2 = _mm_loadu_si128((const __m128i*)(v3f + 4));
	v3_3 = _mm_loadu_si128((const __m128i*)(v3f + 8));
	v4_1 = _mm_loadu_si128((const __m128i*)v4f);
	v4_2 = _mm_loadu_si128((const __m128i*)(v4f + 4));
	v4_3 = _mm_loadu_si128((const __m128i*)(v4f + 8));
	vb_1 = _mm_loadu_ps(vbf);
	vb_2 = _mm_loadu_ps(vbf + 4);
	vb_3 = _mm_loadu_ps(vbf + 8);

	// res = v1 + v2
	res_v1 = _mm_add_epi32(v1_1, v2_1);
	res_v2 = _mm_add_epi32(v1_2, v2_2);
	res_v3 = _mm_add_epi32(v1_3, v2_3);

	// res = res - v3 - v4
	res_v1 = _mm_sub_epi32(res_v1, v3_1);
	res_v2 = _mm_sub_epi32(res_v2, v3_2);
	res_v3 = _mm_sub_epi32(res_v3, v3_3);
	res_v1 = _mm_sub_epi32(res_v1, v4_1);
	res_v2 = _mm_sub_epi32(res_v2, v4_2);
	res_v3 = _mm_sub_epi32(res_v3, v4_3);

	// tmp = res*F
	tmp_1 = _mm_mul_ps(f_vector, _mm_cvtepi32_ps(res_v1));
	tmp_2 = _mm_mul_ps(f_vector, _mm_cvtepi32_ps(res_v2));
	tmp_3 = _mm_mul_ps(f_vector, _mm_cvtepi32_ps(res_v3));

	// res = best + tmp
	tmp_1 = _mm_add_ps(vb_1, tmp_1);
	tmp_2 = _mm_add_ps(vb_2, tmp_2);
	tmp_3 = _mm_add_ps(vb_3, tmp_3);

	// __m128i is 128 bit length vector which can take 4 32-bit integers
	// lower 4, mid 4 and high 4
	//__m128i res_v, v1, v2, v3, v4, vb;

	//// __m128 for storing tmp floats
	//__m128 tmp_res;

	//// load vectors
	//// _mm_loadu_si128 doesn't require source memory to be aligned
	//v1 = _mm_loadu_si128((const __m128i*)vec1);
	//v2 = _mm_loadu_si128((const __m128i*)vec2);
	//v3 = _mm_loadu_si128((const __m128i*)vec3);
	//v4 = _mm_loadu_si128((const __m128i*)vec4);
	//vb = _mm_loadu_si128((const __m128i*)best);

	//// tmp res = v1 + v2 - (v3 - v4)
	//v1 = _mm_add_epi8(v1, v2);
	//v1 = _mm_sub_epi8(v1, v3);
	//res_v = _mm_sub_epi8(v1, v4);
	//tmp_res = _mm_cvtepi32_ps(_mm_sub_epi8(v1, v3));


	//// tmp_res = F * tmp_res
	////tmp_res = _mm_mul_ps(F, tmp_res);
	//res_v = _mm_add_epi8(res_v, vb);
	for (size_t i = 0; i < D; i++)
	{
		//res[i] = res_v.m128i_u8[i];
	}
	res[0] = (byte)tmp_1.m128_f32[0];
	res[1] = (byte)tmp_1.m128_f32[1];
	res[2] = (byte)tmp_1.m128_f32[2];
	res[3] = (byte)tmp_1.m128_f32[3];
	res[4] = (byte)tmp_2.m128_f32[0];
	res[5] = (byte)tmp_2.m128_f32[1];
	res[6] = (byte)tmp_2.m128_f32[2];
	res[7] = (byte)tmp_2.m128_f32[3];
	res[8] = (byte)tmp_3.m128_f32[0];
	res[9] = (byte)tmp_3.m128_f32[1];
}


void binomic_cross(TPassword& res, const TPassword& active_individual, const TPassword& noise_vector) {
	double random_val = 0;
	int i = 0;
	for (i = 0; i < D; i++) {
		random_val = cross_distribution(generator);
		if (random_val <= CR) {
			// use noise
			res[i] = noise_vector[i];
		}
		else {
			// use active individual
			res[i] = active_individual[i];
		}
	}
}

void copy_individual(TPassword& dest, const TPassword& src) {
	int i = 0;
		std::copy(src, src + D, dest);
}

//
//	Compares two individuals and returns true if they're same.
//
bool compare_individuals(TPassword& ind1, TPassword& ind2) {
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
	individual_score = fitness_custom_bit_diff(individual);

	// evolution = mutation + cross
	/*mutation_best_2_my_vec(noise_vec, vb,
		v1, v2, v3, v4);*/
	mutation_best_2(noise_vec, vb,
		v1, v2, v3, v4);
	binomic_cross(res_vec, individual, noise_vec);

	// use better one
	new_score = fitness_custom_bit_diff(res_vec);
	if (new_score > individual_score) {
		copy_individual(res, res_vec);
	}
	else {
		copy_individual(res, individual);
	}
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
	find_best_individual(population, fitness_custom_bit_diff, &best_index, &best_fitness);

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
		new_score = fitness_custom_bit_diff(crossed_vector);
		active_score = fitness_custom_bit_diff(*active_individual);
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
	std::function<double(TPassword&)> fitness_lambda = [](TPassword& psw) {return fitness_custom_bit_diff(psw); };
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