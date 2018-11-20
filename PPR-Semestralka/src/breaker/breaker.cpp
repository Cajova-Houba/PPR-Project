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

// library for SIMD
#include "../vectorclass/vectorclass.h"

// intel TBB
#include "../tbb/tbb.h"

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
const double MAX_DIFF = 806.38080334294;

// differential evolutions constants
// mutacni konstanta rec: 0.3 - 0.9
// interval [0, 2]
const double F = 0.9;

// prah krizeni 0.8 - 0.9
// cim mene, tim vice krizeni
const double CR = 0.2;

// dimenze reseneo problemu = delka hesla
const int D = 10;

// pocet jedincu v populaci 10*D - 100*D
const int NP = 420 * D;

// pocet evolucnich cyklu
// pak bude alg ukoncen
const double GENERATIONS = 800;

// length of TBlock
const int BLOCK_SIZE = 8;

// parametry gaussovske cenove funkce
// G_A je max výška kopce
// G_S je šíøka kopce
const int G_A = 500;
const int G_S = 15;

// number of items in one batch for parallel_for
const int BATCH_SIZE = 6;

// number of vectors to store 4 individuals
const int VEC_LEN = 4;

/*
	One batch of stuff for evolution.
*/
struct evolution_batch {
	int batch_index;

	// pointer to array to store new population to
	// array can be indexed from batch_index to batch_index + 5
	TPassword *new_population;

	TPassword new_pop[BATCH_SIZE];

	// index of best individual
	int best_individual_index;

	TPassword best_individual;
	
	// active individuals for this batch
	TPassword active_individuals[BATCH_SIZE];

	// 4 random individuals for every batch
	int random_individual_indexes[BATCH_SIZE][VEC_LEN];
	TPassword random_individuals[BATCH_SIZE][VEC_LEN];
};


// generator nahodnych cisel
std::mt19937 generator(123);
std::uniform_int_distribution<int> param_value_distribution(0, 255);
std::uniform_int_distribution<int> population_picker_distribution(0, NP);
std::uniform_real_distribution<double> cross_distribution(0, 1);


void print_block(const TBlock& block) {
	int i = 0;
	for (i = 0; i < BLOCK_SIZE; i++) {
		printf("%#x ", block[i]);
	}
	std::cout << std::endl;
}

void print_key(const TPassword& key, const double score) {
	int i = 0;
	for (i = 0; i < D; i++) {
		printf("%#x ", key[i]);
	}
	std::cout << "\t;" << score << std::endl;
}

#pragma region fitness_functions
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
double fitness_custom_bit_diff(const TPassword& individual) {
	double fit = 0;
	int i = 0;
	byte xorRes;
	std::bitset<D> bs;

	for (i = 0; i < D; i++) {
		xorRes = individual[i] ^ (*reference_password)[i];
		bs = (xorRes);
		fit += bs.count();
	}


	return MAX_BIT_DIFF - fit;
}

/*
	How close is individual to reference. Uses gaussian 'hill' where the reference is on the top. 
	The function has following format:
	fit = a * e^(
		-(v1-u1)^2 / (2*s^2)
		-(v2-u2)^2 / (2*s^2)
		-(v3-u3)^2 / (2*s^2)
		...
	)
	Where a and s are standard parameters of gaussian function, v1..vn are components of individual which
	is being tested and u1..un are components of reference block.
*/
double fitness_custom_gauss(const TPassword& individual) {
	int i = 0;
	double fit = 0;

	for (i = 0; i < D; i++)
	{
		fit += -std::pow(individual[i] - (*reference_password)[i], 2) / (2 * G_S*G_S);
	}

	return G_A * std::exp(fit);
}

/*
	How close is individual to reference. This is not good for higher dimensions as wrong
	individuals still can get pretty high score.
*/
double fitness_custom_linear(const TPassword& individual, const TPassword& reference) {
	double dist = 0;
	// max value of custom fitness function
	const double max = MAX_DIFF;
	//const double max = 216;
	int i = 0;
	for (i = 0; i < D; i++) {
		dist += pow(individual[i] - reference[i], 2);
	}
		
	return max - sqrt(dist);
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

// fitness by # of different bites
double fitness_diff(TPassword& individual, TBlock& encrypted, const TBlock &reference) {
	double fit = 0;
	int i = 0;
	TBlock decrypted;
	SJ_context context;
	byte xorRes;
	std::bitset<8> bs;

	makeKey(&context, individual, sizeof(TPassword));
	decrypt_block(&context, decrypted, encrypted);

	for (i = 0; i < BLOCK_SIZE; i++) {
		xorRes = reference[i] ^ decrypted[i];
		bs = (xorRes);
		fit += bs.count();
	}


	return fit;
}

double fitness_diff_inverse(TPassword& individual, TBlock& encrypted, const TBlock &reference) {
	double fit_diff = fitness_diff(individual, encrypted, reference);
	if (abs(fit_diff - 0) < 0.0000000001) {
		return 0;
	}
	else {
		return 1 / fit_diff;
	}
}

#pragma endregion


#pragma region evolution_helpers
/*
	Randomly gnerates one guy in population.
*/
void generate_individual(TPassword& guy) {
	byte i = 0;

	for (i = 0; i < D; i++)
	{
		guy[i] = (byte)param_value_distribution(generator);
	}
}

/*
	Generate first population of DE algorithm.
*/
void create_first_population(TPassword *population) {
	int i = 0;

	for ( i = 0; i < NP; i++)
	{
		generate_individual(population[i]);
	}
}

void weight_vector_difference(TPassword& res, const TPassword& vec1, const TPassword& vec2) {
	int i = 0;
	for (i = 0; i < D; i++) {
		res[i] = (byte)(abs(vec1[i] - vec2[i]) * F);
	}
}

void add_individuals(TPassword& res, const TPassword& vec1, const TPassword& vec2) {
	int i = 0;
	for (i = 0; i < D; i++) {
		res[i] = (vec1[i] + vec2[i]) % 255;
	}
}

/*
	Provede mutaci rand-1.
	diff_vec = (vec1 - vec2) *F
	noise_vec = diff_vec + vec3
	
	Kde vec1 ... vec3 jsou nahodne vybrane, nestejne prvky z populace. 
	Vysledek noise_vec je ulozeny do res.
*/
void mutation_rand_1(TPassword& res, const TPassword& vec1, const TPassword& vec2, const TPassword& vec3) {
	weight_vector_difference(res, vec1, vec2);
	add_individuals(res, res, vec3);
}


/*
	Same as mutation_best_2_vec but doesn't use vectors internaly.
*/
void mutation_best_2(TPassword& res, const TPassword& best, const TPassword& vec1, TPassword& vec2, TPassword& vec3, TPassword& vec4) {
	int i = 0;

	// following loop is replaced with vector library
	 for (i = 0; i < D; i++) {
		 res[i] = (byte)( best[i] + F * (vec1[i] + vec2[i] - vec3[i] - vec4[i]));
	 }
}

/*
	Provede mutaci best-2.
	diff_vec = (vec1 + vec2 - vec3 - vec4) * F
	noise_vec = diff_vec + best
	
	Kde vec1 ... vec2 jsou nahodne vybrane, nestejne prvky z populace a best je nejlepsi jedinec ze soucaasne populace.
*/
void mutation_best_2_vec(TPassword& res, const TPassword& best, const TPassword& vec1, TPassword& vec2, TPassword& vec3, TPassword& vec4) {
	// Vec16uc = unsigned int with length of 8 bits, it can store 16 elements
	Vec16uc v1, v2, v3, v4, vb;

	// Vec16f has single floating point precision and can take 16 elemenets
	Vec16uc res_v;

	// load vectors
	vb.load(best);
	v1.load(vec1);
	v2.load(vec2);
	v3.load(vec3);
	v4.load(vec4);

	// following loop is replaced with vector library
	// for (i = 0; i < D; i++) {
	//	 res[i] = best[i] + F * (vec1[i] + vec2[i] - vec3[i] - vec4[i]);
	// }
	res_v = vb + F * (v1 + v2 - v3 - v4);
	res_v.store_partial(D, res);
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

/*
	Compares two individuals and returns true if they're same.
*/
bool compare_individuals(TPassword& ind1, TPassword& ind2) {
	int i = 0;
	bool same = true;
	
	while (i < D && same) {
		same = ind1[i] == ind2[i];
		i++;
	}

	return same;
}

/*
	Finds the best individual in the population and returns its index and score.
*/
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
#pragma endregion


void prick_four_random(int* r1, int* r2, int* r3, int* r4) {
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

/*
	Prepares one batch for evolution's parallel for.
	Transforms active individuals and their randoms to vectors.
*/
void prepare_evolution_batch(TPassword* population, const std::function<double(TPassword&)> fitness_function, evolution_batch& ev_batch) {
	int i = 0;
	int rand1, rand2, rand3, rand4;
	for (i = 0; i < BATCH_SIZE; i++)
	{
		copy_individual(ev_batch.active_individuals[i], (population[ev_batch.batch_index + i]));
		//ev_batch.active_individuals[i] = &(population[ev_batch.batch_index + i]);

		// pick four different randoms
		prick_four_random(&rand1, &rand2, &rand3, &rand4);
		
		copy_individual(ev_batch.random_individuals[i][0], population[rand1]);
		copy_individual(ev_batch.random_individuals[i][1], population[rand2]);
		copy_individual(ev_batch.random_individuals[i][2], population[rand3]);
		copy_individual(ev_batch.random_individuals[i][3], population[rand4]);
	}
}

void prepare_bestvector_for_batch(TPassword& best_vector, evolution_batch& ev_batch) {
	copy_individual(ev_batch.best_individual, best_vector);
}

void process_evolution_batch(evolution_batch& ev_batch, const std::function<double(TPassword&)> fitness_function) {
	int act_ind_cntr = 0;
	TPassword noise_vec;
	TPassword res_vec;
	double new_score, active_score;

	for (act_ind_cntr = 0; act_ind_cntr < BATCH_SIZE; act_ind_cntr++)
	{
		// mutation 
		mutation_best_2_vec(noise_vec, ev_batch.best_individual,
			ev_batch.random_individuals[act_ind_cntr][0],
			ev_batch.random_individuals[act_ind_cntr][1],
			ev_batch.random_individuals[act_ind_cntr][2],
			ev_batch.random_individuals[act_ind_cntr][3]
		);

		// cross
		binomic_cross(res_vec, ev_batch.active_individuals[act_ind_cntr], noise_vec);

		// add to new pop
		new_score = fitness_function(res_vec);
		active_score = fitness_function(ev_batch.active_individuals[act_ind_cntr]);
		if (new_score > active_score) {
			// use crossed_vector for new population
			copy_individual(ev_batch.new_pop[act_ind_cntr], res_vec);
		}
		else {
			// use active_score for new population
			copy_individual(ev_batch.new_pop[act_ind_cntr], ev_batch.active_individuals[act_ind_cntr]);
		}

	}
}

/*
	Same as evolution(...) but uses parallel for and vector instructions.
	Vec16uc is used for vector type as two vectors can fit 3 individuals. One batch of parallel_for is 
	2 vectors -> 6 individuals.
*/
void evolution_parallel(TPassword* population, TPassword* new_population, TBlock& encrypted, const TBlock& reference, const std::function<double(TPassword&)> fitness_function) {
	int i = 0;
	int j = 0;
	int k = 0;
	int best_index = -1;
	double best_fitness = 0;
	const size_t batch_count = 5;
	evolution_batch batches[batch_count];

	// best individual, this will be the same for whole evolution cycle
	find_best_individual(population, fitness_function, &best_index, &best_fitness);
	if (PRINT_BEST) {
		print_key((population[best_index]), best_fitness);
	}
	for (i = 0; i < batch_count; i++)
	{
		prepare_bestvector_for_batch(population[best_index], batches[i]);
	}

	for (i = 0; i < NP; i += BATCH_SIZE*batch_count) 
	{
		// prepare batches
		for (j = 0; j < batch_count; j++)
		{
			batches[j].batch_index = i + j * BATCH_SIZE;
			prepare_evolution_batch(population, fitness_function, batches[j]);
		}

		// do stuff with the batches
		tbb::parallel_for(size_t(0), batch_count, [&batches, &fitness_function]( size_t(i)) {
			process_evolution_batch(batches[i], fitness_function);
		});

		// results from proccessed batches
		for (j = 0; j < batch_count; j++) {
			for ( k = 0; k < batch_count; k++)
			{
				copy_individual(new_population[i + j*BATCH_SIZE + k], batches[j].new_pop[k]);
			}
		}
	}
}

/*
	Performs one cycle of evolution over given population. Encrypted and reference blocks are used
	for calculating fitness function.

	New population is stored to new_population. 
*/
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
	find_best_individual(population, fitness_function, &best_index, &best_fitness);

	if (PRINT_BEST) {
		print_key((population[best_index]), best_fitness);
	}

	for (i = 0; i < NP; i++)
	{

		// active individual population[i]
		active_individual = &(population[i]);

		prick_four_random(&rand1, &rand2, &rand3, &rand4);
		randomly_picked_1 = &(population[rand1]);
		randomly_picked_2 = &(population[rand2]);
		randomly_picked_3 = &(population[rand3]);
		randomly_picked_4 = &(population[rand4]);

		//mutation_rand_1(noise_vector, *randomly_picked_1, *randomly_picked_2, *randomly_picked_3);
		mutation_best_2(noise_vector, population[best_index], *randomly_picked_1, *randomly_picked_2, *randomly_picked_3, *randomly_picked_4);



		// cross
		//	1. y = cross noise_vector (v) with active individual (x_i) (by CR parameter)
		binomic_cross(crossed_vector, *active_individual, noise_vector);
		
		// choose new guy to population
		// evaluate fitness(y)
		// if (fitness(y) > fitness(active_individual)) -> y || active_individual
		new_score = fitness_function(crossed_vector);
		active_score = fitness_function(*active_individual);
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