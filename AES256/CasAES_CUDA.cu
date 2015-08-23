//
//  CasAES_CUDA.c
//  CasAES_CUDA
//  Created by Carter McCardwell on 11/11/14.
//

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime.h>

const int Nb_h = 4;
const int Nr_h = 14;
const int Nk_h = 8;

const uint8_t s_h[256] = {
		0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
		0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
		0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
		0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
		0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
		0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
		0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
		0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
		0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
		0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
		0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
		0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
		0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
		0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
		0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
		0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
};

uint8_t Rcon_h[256] = {
		0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a,
		0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39,
		0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a,
		0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8,
		0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef,
		0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc,
		0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b,
		0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3,
		0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94,
		0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
		0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35,
		0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f,
		0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04,
		0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63,
		0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd,
		0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d
};

__constant__ uint8_t s[256];
__constant__ int Nb;
__constant__ int Nr;
__constant__ int Nk;
__constant__ uint32_t ek[60];

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void cudaDevAssist(cudaError_t code, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"cudaDevAssistant: %s %d\n", cudaGetErrorString(code), line);
		if (abort) exit(code);
	}
}

uint32_t sw(uint32_t word)
{
	union {
		uint32_t word;
		uint8_t bytes[4];
	} subWord  __attribute__ ((aligned));
	subWord.word = word;

	subWord.bytes[3] = s_h[subWord.bytes[3]];
	subWord.bytes[2] = s_h[subWord.bytes[2]];
	subWord.bytes[1] = s_h[subWord.bytes[1]];
	subWord.bytes[0] = s_h[subWord.bytes[0]];

	return subWord.word;
}

__device__ void sb(uint8_t* in)
{
	for (int i = 0; i < 32; i++) { in[i] = s[in[i]]; }
}

__device__ void sb_st(uint8_t* in)
{
	for (int i = 0; i < 16; i++) { in[i] = s[in[i]]; }

}

__device__ void mc(uint8_t* arr)
{
	for (int i = 0; i < 4; i++)
	{
		uint8_t a[4];
		uint8_t b[4];
		uint8_t c;
		uint8_t h;
		for(c=0;c<4;c++) {
			a[c] = arr[(4*c+i)];
			h = (uint8_t)((signed char)arr[(4*c+i)] >> 7);
			b[c] = arr[(4*c+i)] << 1;
			b[c] ^= 0x1B & h;
		}
		arr[(i)] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
		arr[(4+i)] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
		arr[(8+i)] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
		arr[(12+i)] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0];
	}

}

__device__ void sr(uint8_t* arr)
{
	uint8_t out[16];
	//On per-row basis (+1 shift ea row)
	//Row 1
	out[0] = arr[0];
	out[1] = arr[1];
	out[2] = arr[2];
	out[3] = arr[3];
	//Row 2
	out[4] = arr[5];
	out[5] = arr[6];
	out[6] = arr[7];
	out[7] = arr[4];
	//Row 3
	out[8] = arr[10];
	out[9] = arr[11];
	out[10] = arr[8];
	out[11] = arr[9];
	//Row 4
	out[12] = arr[15];
	out[13] = arr[12];
	out[14] = arr[13];
	out[15] = arr[14];

	for (int i = 0; i < 16; i++)
	{
		arr[i] = out[i];
	}
}

uint32_t rw(uint32_t word)
{
	union {
		uint8_t bytes[4];
		uint32_t word;
	} subWord  __attribute__ ((aligned));
	subWord.word = word;

	uint8_t B0 = subWord.bytes[3], B1 = subWord.bytes[2], B2 = subWord.bytes[1], B3 = subWord.bytes[0];
	subWord.bytes[3] = B1; //0
	subWord.bytes[2] = B2; //1
	subWord.bytes[1] = B3; //2
	subWord.bytes[0] = B0; //3

	return subWord.word;
}

void K_Exp(uint8_t* pk, uint32_t* out)
{
	int i = 0;
	union {
		uint8_t bytes[4];
		uint32_t word;
	} temp __attribute__ ((aligned));
	union {
		uint8_t bytes[4];
		uint32_t word;
	} univar[60] __attribute__ ((aligned));

	for (i = 0; i < Nk_h; i++)
	{
		univar[i].bytes[3] = pk[i*4];
		univar[i].bytes[2] = pk[i*4+1];
		univar[i].bytes[1] = pk[i*4+2];
		univar[i].bytes[0] = pk[i*4+3];
	}

	for (i = Nk_h; i < Nb_h*(Nr_h+1); i++)
	{
		temp.word = univar[i-1].word;
		if (i % Nk_h == 0)
		{
			temp.word = (sw(rw(temp.word)));
			temp.bytes[3] = temp.bytes[3] ^ (Rcon_h[i/Nk_h]);
		}
		else if (Nk_h > 6 && i % Nk_h == 4)
		{
			temp.word = sw(temp.word);
		}
		if (i-4 % Nk_h == 0)
		{
			temp.word = sw(temp.word);
		}
		univar[i].word = univar[i-Nk_h].word ^ temp.word;
	}
	for (i = 0; i < 60; i++)
	{
		out[i] = univar[i].word;
	}
}

__device__ void ark(uint8_t* state, int strD, uint32_t* eK)
{
	union {
		uint32_t word;
		uint8_t bytes[4];
	} kb[4] __attribute__ ((aligned));

	kb[0].word = eK[strD];
	kb[1].word = eK[strD+1];
	kb[2].word = eK[strD+2];
	kb[3].word = eK[strD+3];

	for (int i = 0; i < 4; i++)
	{
		state[i] = state[i] ^ kb[i].bytes[3];
		state[i+4] = state[i+4] ^ kb[i].bytes[2];
		state[i+8] = state[i+8] ^ kb[i].bytes[1];
		state[i+12] = state[i+12] ^ kb[i].bytes[0];
	}
}

__global__ void cudaRunner(uint8_t *in)
{
	uint8_t state[16];
  int localid = blockDim.x * blockIdx.x + threadIdx.x; //Data is shifted by 16 * ID of worker
  for (int i = 0; i < 16; i++) { state[i] = in[(localid*16)+i]; }

	ark(state, 0, ek);
	for (int i = 1; i < 14; i++)
	{
		sb_st(state);
		sr(state);
		mc(state);
		ark(state, i*Nb, ek);
	}

	sb_st(state);
	sr(state);
	ark(state, Nr*Nb, ek);

	for (int i = 0; i < 16; i++) { in[(localid*16)+i] = state[i]; }
}

int main(int argc, const char * argv[])
{
	printf("CasAES_CUDA Hyperthreaded AES-256 Encryption for CUDA processors - compiled 3/25/2015 Rev. 4\nCarter McCardwell, Northeastern University NUCAR - http://coe.neu.edu/~cmccardw - mccardwell.net\nPlease Wait...\n");

  clock_t c_start, c_stop;
  c_start = clock();

	FILE *infile;
	FILE *keyfile;
	FILE *outfile;

	infile = fopen(argv[2], "r");
    if (infile == NULL) { printf("error (infile)\n"); return(1); }
	keyfile = fopen(argv[3], "rb");
    if (keyfile == NULL) { printf("error (keyfile)\n"); return(1); }
	outfile = fopen(argv[4], "w");
    if (outfile == NULL) { printf("error (outfile permission error, run with sudo)\n"); return(1); }

    //Hex info, or ASCII
    bool hexMode = false;
    if (strcmp(argv[1], "h") == 0) { hexMode = true; }
    else if (strcmp(argv[1], "a") == 0) { hexMode = false; }
    else { printf("error: first argument must be \'a\' for ASCII interpretation or \'h\' for hex interpretation\n"); return(1); }

	uint8_t key[32];
	uint32_t ek_h[60];

	for (int i = 0; i < 32; i++)
	{
		fscanf(keyfile, "%x", &key[i]);
	}

	K_Exp(key, ek_h);

	//send constants to GPU
	cudaSetDevice(0);
	cudaDevAssist(cudaMemcpyToSymbol(Nk, &Nk_h, sizeof(int), 0, cudaMemcpyHostToDevice), 535, true);
	cudaDevAssist(cudaMemcpyToSymbol(Nr, &Nr_h, sizeof(int), 0, cudaMemcpyHostToDevice), 543, true);
	cudaDevAssist(cudaMemcpyToSymbol(Nb, &Nb_h, sizeof(int), 0, cudaMemcpyHostToDevice), 903, true);
	cudaDevAssist(cudaMemcpyToSymbol(s, &s_h, 256*sizeof(uint8_t), 0, cudaMemcpyHostToDevice), 920, true);
	cudaDevAssist(cudaMemcpyToSymbol(ek, &ek_h, 60*sizeof(uint32_t), 0, cudaMemcpyHostToDevice), 823, true);
	cudaThreadSynchronize();

	const int BLOCKS = -1; //Not used
	const int RUNNING_THREADS = 512;

	uint8_t *devState = NULL;
	cudaDevAssist(cudaMalloc((void**)&devState, RUNNING_THREADS*16*sizeof(uint8_t)), 425, true);

	uint8_t states[RUNNING_THREADS][16] = { 0x00 };
  int ch = 0;
	int spawn = 0;
	int end = 1;
	while (end)
	{
		spawn = 0;
		for (int i = 0; i < RUNNING_THREADS; i++) //Dispatch many control threads that will report back to main (for now 5x) - 1 worker per state
		{
            spawn++;
			for (int ix = 0; ix < 16; ix++)
			{
                if (hexMode)
                {
                    if (fscanf(infile, "%x", &states[i][ix]) != EOF) { ; }
                    else
                    {
                        if (ix > 0) { for (int ixx = ix; ixx < 16; ixx++) { states[i][ixx] = 0x00; } }
                        else { spawn--; }
                        i = RUNNING_THREADS + 1;
                        end = 0;
                        break;
                    }
                }
                else
                {
                    ch = getc(infile);
                    if (ch != EOF) { states[i][ix] = ch; }
                    else
                    {
                        if (ix > 0) { for (int ixx = ix; ixx < 16; ixx++) { states[i][ixx] = 0x00; } }
                        else { spawn--; }
                        i = RUNNING_THREADS + 1;
                        end = 0;
                        break;
                    }
                }
			}
		}
		//arrange data correctly
		for (int i = 0; i < spawn; i++)
		{
			uint8_t temp[16];
			memcpy(&temp[0], &states[i][0], sizeof(uint8_t));
			memcpy(&temp[4], &states[i][1], sizeof(uint8_t));
			memcpy(&temp[8], &states[i][2], sizeof(uint8_t));
			memcpy(&temp[12], &states[i][3], sizeof(uint8_t));
			memcpy(&temp[1], &states[i][4], sizeof(uint8_t));
			memcpy(&temp[5], &states[i][5], sizeof(uint8_t));
			memcpy(&temp[9], &states[i][6], sizeof(uint8_t));
			memcpy(&temp[13], &states[i][7], sizeof(uint8_t));
			memcpy(&temp[2], &states[i][8], sizeof(uint8_t));
			memcpy(&temp[6], &states[i][9], sizeof(uint8_t));
			memcpy(&temp[10], &states[i][10], sizeof(uint8_t));
			memcpy(&temp[14], &states[i][11], sizeof(uint8_t));
			memcpy(&temp[3], &states[i][12], sizeof(uint8_t));
			memcpy(&temp[7], &states[i][13], sizeof(uint8_t));
			memcpy(&temp[11], &states[i][14], sizeof(uint8_t));
			memcpy(&temp[15], &states[i][15], sizeof(uint8_t));
			for (int c = 0; c < 16; c++) { memcpy(&states[i][c], &temp[c], sizeof(uint8_t)); }
		}

		//printf("\nCycle!: Spawn = %i", spawn);

		cudaDevAssist(cudaMemcpy(devState, *states, spawn*16*sizeof(uint8_t), cudaMemcpyHostToDevice), 426, true);
		cudaDevAssist(cudaDeviceSynchronize(), 268, true);
		cudaRunner<<<1,spawn>>>(devState);

		cudaDevAssist(cudaDeviceSynchronize(), 270, true);
		cudaDevAssist(cudaMemcpy(*states, devState, spawn*16*sizeof(uint8_t), cudaMemcpyDeviceToHost), 431, true);


		//Write results to out
		for (int i = 0; i < spawn; i++)
		{
			for (int ix = 0; ix < 4; ix++)
			{
				char hex[3];
				sprintf(hex, "%02x", states[i][ix]);
				for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
				sprintf(hex, "%02x", states[i][ix+4]);
				for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
				sprintf(hex, "%02x", states[i][ix+8]);
				for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
				sprintf(hex, "%02x", states[i][ix+12]);
				for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
			}
		}
	}
  c_stop = clock();
  float diff = (((float)c_stop - (float)c_start) / CLOCKS_PER_SEC ) * 1000;

  printf("Done - Time taken: %f ms\n", diff);
	cudaFree(devState);
	cudaDeviceReset();
	fclose(infile);
	fclose(outfile);
	fclose(keyfile);
	return 0;
}
