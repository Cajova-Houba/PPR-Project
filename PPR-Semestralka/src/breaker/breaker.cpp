#include "../breaker.h"

//V ramci vypracovani semestralni prace muzete menit pouze a jedine tento souboru od teto radky dale.


#include <memory.h>
#include <iostream>
#include <cmath>
#include <string>

double fitness(TBlock& block) {
	double fit = 0;
	int i = 0;
	TBlock reference{ '#','p','r','a','g','m','a','\0' };

	for (i = 0; i < 8; i++) {
		fit += (reference[i] - block[i])*(reference[i] - block[i]);
	}

	fit = std::sqrt(fit);

	return fit;
}

void print_block(const TBlock& block) {
	int i = 0;
	for (i = 0; i < 8; i++) {
		printf("%#x ", block[i]);
	}
	std::cout << std::endl;
}

bool break_the_cipher(TBlock &encrypted, const TBlock &reference, TPassword &password) {
	SJ_context context;
	TBlock decrypted;
	TPassword testing_key{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	int i = 0, j=0;
	TBlock plain_texts[512];
	TBlock encrypted_texts[512];
	byte cntr1 = 0;
	byte cntr2 = 0;
	for ( i = 0; i < 512; i++)
	{
		for (j = 0; j < 8; j++) {
			if (j == 1) {
				plain_texts[i][1] = cntr1;
				cntr1++;
				if (cntr1 == 0) {
					cntr2++;
				}
			}
			else if (j == 2) {
				plain_texts[i][2] = cntr2;
			}
			else {
				plain_texts[i][j] = j;
			}
		}
		//print_block(plain_texts[i]);

	}

	// todo: rest of the attack: http://www.cs.technion.ac.il/~biham/Reports/SkipJack/note3.html
	
	for (testing_key[0] = 0; testing_key[0] < 255; testing_key[0]++) {
		for (testing_key[1] = 0; testing_key[1] < 255; testing_key[1]++) {
			makeKey(&context, testing_key, sizeof(TPassword));
			decrypt_block(&context, decrypted, encrypted);
			//std::cout << "Fitness of key " << std::to_string(testing_key[0]) << " = " + std::to_string(fitness(decrypted)) << std::endl;
			//std::cout << std::to_string(testing_key[0]) << "," + std::to_string(fitness(decrypted)) << std::endl;
			//std::cout << std::to_string(testing_key[0]) << ": " << std::to_string(decrypted[0]) << ", " << std::to_string('#') << std::endl;
			//std::cout << "Fitness=" << std::to_string(fitness(decrypted)) << std::endl;
			if (memcmp(decrypted, reference, sizeof(TBlock)) == 0) {
				memcpy(password, testing_key, sizeof(TPassword));
				std::cout << "Done!" << std::endl;
				return true;
			}
		}
	}

	return false;
}