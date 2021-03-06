// PPR-Semestralka.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>

#include "skipjack.h"
#include "breaker.h"

constexpr size_t reference_fname_idx = 1;
constexpr size_t encrypted_fname_idx = 2;
constexpr size_t password_fname_idx = 3;

void encrypt_reference(TBlock &reference, TPassword &password, TBlock &encrypted) {
	SJ_context context;
	makeKey(&context, password, sizeof(TPassword));
	encrypt_block(&context, encrypted, reference);
}

template <typename T>
void read_input(T &block, const char* prompt) {
	std::string input;
	while ((input.size() > sizeof(T)) || input.empty()) {
		std::cout << prompt << std::endl;
		std::getline(std::cin, input);
	}
	std::copy(input.begin(), input.end(), block);
}

void read_file(const char* file_name, TBlock &block) {
	std::ifstream out{ file_name };
	out.read(reinterpret_cast<char*>(block), sizeof(TBlock));
}

template <typename T>
void write_file(const char* file_name, T &block) {
	std::ofstream out{ file_name };
	out.write(reinterpret_cast<char*>(block), sizeof(T));
};


int __cdecl main(int argc, char **argv) {

	TBlock encrypted;
	TBlock reference{ 0 };

	if (argc != 3) {

		if (argc == 4) {
			TPassword password{ 0 };

			//zapisujeme referencni a sifrovany sifrovany blok, ktery vygenerujeme
			read_input(password, "Zadej heslo: ");
			read_input(reference, "Zadej text: ");
			encrypt_reference(reference, password, encrypted);
			write_file(argv[reference_fname_idx], reference);
			write_file(argv[encrypted_fname_idx], encrypted);
			write_file(argv[password_fname_idx], password);
		}
		else {
			std::cout << "Pouziti programu:" << std::endl;
			std::cout << "generovani souboru: skipjack.exe referencni_blok zasifrovany_blok heslo" << std::endl;
			std::cout << "cteni souboru: skipjack.exe referencni_blok zasifrovany_blok " << std::endl;
			std::cout << std::endl;
			std::cout << "Spoustim demoverzi..." << std::endl;

			//jenom pripravime demo bloky
			memcpy(reference, "#pragma", sizeof(TBlock));
			TPassword password{ 'h', 'e', 'l', 'l', 'o', '1', '2', '3', '4', '5' };
			encrypt_reference(reference, password, encrypted);
		}
	}
	else {
		//nacteme predem vygenerovany referencni a zasifrovany blok
		read_file(argv[reference_fname_idx], reference);
		read_file(argv[encrypted_fname_idx], encrypted);
	}


	//mame vsechna vstupni data, pokusime se tedy prolomit heslo
	TPassword found_password;
	if (break_the_cipher(encrypted, reference, found_password)) {
		std::cout << std::setfill('0') << std::hex;
		for (size_t i = 0; i < sizeof(TPassword); i++)
			std::cout << "0x" << std::setw(2) << (int)found_password[i] << " ";
	}
	else
		std::cout << "Sifra neprolomena:(" << std::endl;


	getchar();

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
