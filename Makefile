CC =gcc
CFLAGS=-std=c99
CPPFLAGS =

TARGETS = init keygen encrypt decrypt

all: run

init: init.o matrix.o qc_mdpc.o mceliece.o utility.o
	$(CC) init.o matrix.o mceliece.o qc_mdpc.o utility.o -o init

keygen: key_gen.o matrix.o qc_mdpc.o utility.o
	$(CC) key_gen.o matrix.o qc_mdpc.o  utility.o -o keygen

encrypt: encrypt.o matrix.o qc_mdpc.o mceliece.o utility.o
	$(CC) encrypt.o matrix.o mceliece.o qc_mdpc.o utility.o -o encrypt

decrypt: decrypt.o matrix.o qc_mdpc.o mceliece.o utility.o
	$(CC) decrypt.o matrix.o mceliece.o qc_mdpc.o utility.o -o decrypt

run: test.o matrix.o qc_mdpc.o mceliece.o utility.o
	$(CC) test.o matrix.o qc_mdpc.o mceliece.o utility.o -o run

init.o: init.c matrix.c qc_mdpc.c mceliece.c utility.c
	$(CC) -c init.c matrix.c qc_mdpc.c mceliece.c utility.c

keygen.o: key_gen.c matrix.c qc_mdpc.c utility.c
	$(CC) -c key_gen.c matrix.c qc_mdpc.c utility.c

encrypt.o: encrypt.c matrix.c qc_mdpc.c mceliece.c utility.c 
	$(CC) -c encrypt.c matrix.c qc_mdpc.c mceliece.c utility.c

decrypt.o: decrypt.c matrix.c qc_mdpc.c mceliece.c utility.c
	$(CC) -c decrypt.c matrix.c qc_mdpc.c mceliece.c utility.c

test.o:	test.c matrix.c qc_mdpc.c mceliece.c utility.c
	$(CC) -c test.c matrix.c qc_mdpc.c mceliece.c utility.c

min_dist.o:	min_dist.c matrix.c qc_mdpc.c mceliece.c utility.c
	$(CC) -c min_dist.c matrix.c qc_mdpc.c mceliece.c utility.c

matrix.o:	matrix.c matrix.h
	$(CC) -c matrix.c

utility.o: utility.c utility.h
	$(CC) -c utility.c

qc_mdpc.o:	qc_mdpc.c qc_mdpc.h matrix.c utility.c
	$(CC) -c qc_mdpc.c matrix.c utility.c

mceliece.o:	mceliece.c mceliece.h qc_mdpc.c matrix.c utility.c
	$(CC) -c mceliece.c matrix.c qc_mdpc.c utility.c
clean:
	rm -rf *o run
	rm -rf *o decrypt
	rm -rf *o encrypt
	rm -rf *o keygen
	rm -rf *o init
