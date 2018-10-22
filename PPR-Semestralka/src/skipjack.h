#pragma once

//See the original at ftp://ftp.funet.fi/pub/crypt/cryptography/symmetric/skipjack/

typedef unsigned char  byte;

typedef struct {
	byte tab[10][256];
} SJ_context;

using TPassword = byte[10];
using TBlock = byte[8];

void makeKey(SJ_context *ctx, byte *key, unsigned keylen);
void encrypt_block(SJ_context *ctx, byte *out, byte *in);
void decrypt_block(SJ_context *ctx, byte *out, byte *in);