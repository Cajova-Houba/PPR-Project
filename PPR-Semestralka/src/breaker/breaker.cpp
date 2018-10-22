#include "../breaker.h"

//V ramci vypracovani semestralni prace muzete menit pouze a jedine tento souboru od teto radky dale.


#include <memory.h>
#include <iostream>
#include <cmath>
#include <string>
#include <random>

// differential evolutions constants
// mutacni konstanta 0.3 - 0.9
const double F = 0.6;

// prah krizeni 0.8 - 0.9
const double CR = 0.85;

// dimenze reseneo problemu = delka hesla - 1
const int D = 2;

// pocet jedincu v populaci 10*D - 100*D
const int NP = 30 * D;

// prototyp jedince
const double SPECIMEN = 0;

// pocet evolucnich cyklu
// pak bude alg ukoncen
const double GENERATIONS = 10;

// generator nahodnych cisel
std::mt19937 generator(123);
std::uniform_int_distribution<int> param_value_distribution(0, 255);
std::uniform_int_distribution<int> population_picker_distribution(0, NP);
std::uniform_real_distribution<double> cross_distribution(0, 1);


void print_block(const TBlock& block) {
	int i = 0;
	for (i = 0; i < 8; i++) {
		printf("%#x ", block[i]);
	}
	std::cout << std::endl;
}

void print_key(const TPassword& key, const double score) {
	int i = 0;
	for (i = 0; i < 10; i++) {
		printf("%#x ", key[i]);
	}
	std::cout << "\t" << score << std::endl;
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

	for (i = 0; i < 8; i++) {
		fit += (reference[i] - decrypted[i])*(reference[i] - decrypted[i]);
	}

	fit = std::sqrt(fit);

	return fit;
}

/*
	Randomly gnerates one guy in population.
*/
void generate_individual(TPassword& guy) {
	int i = 0;

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

void copy_individual(TPassword& res, const TPassword& src) {
	int i = 0;
	for ( i = 0; i < D; i++)
	{
		res[i] = src[i];
	}
}

/*
	Performs one cycle of evolution over given population. Encrypted and reference blocks are used
	for calculating fitness function.

	New population is stored to new_population. 
*/
void evolution(TPassword *population, TPassword *new_population, TBlock& encrypted, const TBlock &reference) {
	int i = 0;
	int rand1, rand2, rand3;
	TPassword* active_individual;
	TPassword* randomly_picked_1;
	TPassword* randomly_picked_2;
	TPassword* randomly_picked_3;
	TPassword diff_vector;
	TPassword noise_vector;
	TPassword crossed_vector;
	double new_score = 0;
	double active_score = 0;


	for (i = 0; i < NP; i++)
	{
		// active individual population[i]
		active_individual = &(population[i]);

		// mutation
		//	1. randomly pick 2 (different) individuals from population
		rand1 = population_picker_distribution(generator);
		rand2 = rand1;
		while (rand2 != rand1) {
			rand2 = population_picker_distribution(generator);
		}
		randomly_picked_1 = &(population[rand1]);
		randomly_picked_2 = &(population[rand2]);

		//  2. diff_vector = substract those two individuals
		//  3. weighted_diff_vector *= F
		weight_vector_difference(diff_vector, *randomly_picked_1, *randomly_picked_2);

		//	4. noise_vector = weighted_diff_vector + population[random]
		// noise vector is result of mutation
		rand3 = population_picker_distribution(generator);
		randomly_picked_3 = &(population[rand3]);
		add_individuals(noise_vector, diff_vector, *randomly_picked_3);

		// cross
		//	1. y = cross noise_vector (v) with active individual (x_i) (by CR parameter)
		binomic_cross(crossed_vector, *active_individual, noise_vector);
		
		// choose new guy to population
		// evaluate fitness(y)
		// if (fitness(y) > fitness(active_individual)) -> y || active_individual
		new_score = fitness(crossed_vector, encrypted, reference);
		active_score = fitness(*active_individual, encrypted, reference);
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

	std::cout << "Creating first population " << std::endl;
	create_first_population(population);
	std::cout << "First population created " << std::endl;

	// keep evolving till the key is guessed or max number of generations is reached
	while (generation < GENERATIONS && !done) {

		// try every key in new population
		// generation mod 2 ifs are used to change between population arrays 
		for (i = 0; i < NP; i++) {

			if (generation % 2 == 0) {
				print_key(population[i], fitness(population[i], encrypted, reference));
				makeKey(&context, population[i], sizeof(TPassword));
			}
			else {
				print_key(new_population[i], fitness(new_population[i], encrypted, reference));
				makeKey(&context, new_population[i], sizeof(TPassword));
			}

			// if the decryption is successfull => bingo
			decrypt_block(&context, decrypted, encrypted);
			if (memcmp(decrypted, reference, sizeof(TBlock)) == 0) {
				memcpy(password, testing_key, sizeof(TPassword));
				std::cout << "Done!" << std::endl;
				done = true;
				break;
			}
		}

		// use current array as source of evolution and old array as destination of evolution
		if (generation % 2 == 0) {
			evolution(population, new_population, encrypted, reference);
		}
		else {
			evolution(new_population, population, encrypted, reference);
		}
		
		if (generation % 1 == 0) {
			std::cout << "Generation: " << std::to_string(generation) << std::endl;
		}

		generation++;


	}

	return done;

	/*for (testing_key[0] = 0; testing_key[0] < 255; testing_key[0]++) {
		for (testing_key[1] = 0; testing_key[1] < 255; testing_key[1]++) {
			makeKey(&context, testing_key, sizeof(TPassword));
			decrypt_block(&context, decrypted, encrypted);
			if (memcmp(decrypted, reference, sizeof(TBlock)) == 0) {
				memcpy(password, testing_key, sizeof(TPassword));
				std::cout << "Done!" << std::endl;
				return true;
			}
		}
	}

	return false;*/
}