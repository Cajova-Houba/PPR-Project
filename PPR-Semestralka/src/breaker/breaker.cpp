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

const bool PRINT_KEY = false;

const TPassword reference_password{ 'h', 'e', 'l', 0, 0, 0, 0, 0, 0, 0 };

// differential evolutions constants
// mutacni konstanta rec: 0.3 - 0.9
// interval [0, 2]
const double F = 1.5;

// prah krizeni 0.8 - 0.9
// cim mene, tim vice krizeni
const double CR = 0.2;

// dimenze reseneo problemu = delka hesla
const int D = 3;

// pocet jedincu v populaci 10*D - 100*D
const int NP = 440 * D;

// prototyp jedince
const double SPECIMEN = 0;

// pocet evolucnich cyklu
// pak bude alg ukoncen
const double GENERATIONS = 1000;

// length of TBlock
const int BLOCK_SIZE = 8;

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

// deprecated
void print_key(const TPassword& key, const double score, const double score2) {
	int i = 0;
	for (i = 0; i < D; i++) {
		printf("%#x ", key[i]);
	}
	std::cout << "\t;" << score << ";" << score2 << std::endl;
}

void print_key(const TPassword& key, const double score) {
	int i = 0;
	for (i = 0; i < D; i++) {
		printf("%#x ", key[i]);
	}
	std::cout << "\t;" << score << std::endl;
}

/*
	How close is individual to reference.
*/
double fitness_custom(const TPassword& individual, const TPassword& reference) {
	double dist = 0;
	// max value of custom fitness function
	const double max = 262;
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
	Provede mutaci best-2.
	diff_vec = (vec1 + vec2 - vec3 - vec4) * F
	noise_vec = diff_vec + best
	
	Kde vec1 ... vec2 jsou nahodne vybrane, nestejne prvky z populace a best je nejlepsi jedinec ze soucaasne populace.
*/
void mutation_best_2(TPassword& res, const TPassword& best, const TPassword& vec1, TPassword& vec2, TPassword& vec3, TPassword& vec4) {
	int i = 0;
	for (i = 0; i < D; i++) {
		res[i] = best[i] + F * (vec1[i] + vec2[i] - vec3[i] - vec4[i]);
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

void copy_individual(TPassword& dest, const TPassword& src) {
	int i = 0;
	for ( i = 0; i < D; i++)
	{
		dest[i] = src[i];
	}
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
	int bestIndex = -1;
	double bestFitness = 0;
	double fitness = 0;


	// find best individual in current population
	// possible optimization here
	for (i = 0; i < NP; i++)
	{
		fitness = fitness_function(population[i]);
		if (bestIndex == -1 || fitness > bestFitness) {
			bestIndex = i;
			bestFitness = fitness;
		}
	}
	
	for (i = 0; i < NP; i++)
	{

		// active individual population[i]
		active_individual = &(population[i]);

		// mutation
		rand1 = population_picker_distribution(generator);
		rand2 = rand1;
		while (!compare_individuals(population[rand1], population[rand2])) {
			rand2 = population_picker_distribution(generator);
		}
		rand3 = rand2;
		while (!compare_individuals(population[rand2], population[rand3])) {
			rand3 = population_picker_distribution(generator);
		}
		rand4 = rand3;
		while (!compare_individuals(population[rand3], population[rand4])) {
			rand4 = population_picker_distribution(generator);
		}
		randomly_picked_1 = &(population[rand1]);
		randomly_picked_2 = &(population[rand2]);
		randomly_picked_3 = &(population[rand3]);
		randomly_picked_4 = &(population[rand4]);

		//mutation_rand_1(noise_vector, *randomly_picked_1, *randomly_picked_2, *randomly_picked_3);
		mutation_best_2(noise_vector, population[bestIndex], *randomly_picked_1, *randomly_picked_2, *randomly_picked_3, *randomly_picked_4);



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
	std::function<double(TPassword&)> fitness_lambda = [](TPassword& psw) {return fitness_custom(psw, reference_password); };

	std::cout << "Creating first population " << std::endl;
	create_first_population(population);
	std::cout << "First population created " << std::endl;

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
				print_key(current_population_array[i], fitness_lambda(current_population_array[i]));
				std::cout << "Done in " << generation << "!" << std::endl;
				done = true;
				break;
			}
		}

		// use current array as source of evolution and old array as destination of evolution
		if (generation % 2 == 0) {
			evolution(population, new_population, encrypted, reference, fitness_lambda);
		}
		else {
			evolution(new_population, population, encrypted, reference, fitness_lambda);
		}
		
		if (generation % 5 == 0) {
			std::cout << "Generation: " << std::to_string(generation) << std::endl;
		}

		generation++;
	}

	return done;

	//for (testing_key[0] = 0; testing_key[0] < 255; testing_key[0]++) {
	//	//for (testing_key[1] = 0; testing_key[1] < 255; testing_key[1]++) {
	//		print_key(testing_key, fitness_custom(testing_key, reference_password), fitness_diff_inverse(testing_key, encrypted, reference));
	//		makeKey(&context, testing_key, sizeof(TPassword));
	//		decrypt_block(&context, decrypted, encrypted);
	//		if (memcmp(decrypted, reference, sizeof(TBlock)) == 0) {
	//			memcpy(password, testing_key, sizeof(TPassword));
	//			//std::cout << "Done!" << std::endl;
	//			//return true;
	//		}
	//	//}
	//}

	//return false;
}