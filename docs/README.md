# Hardware Accelerated McEliece Cryptosystem

CUDA based implementation of https://github.com/Varad0612/The-McEliece-Cryptosystem


## How to build and run:
First build the software:
```bash
$ mkdir -p build_dir && cd build_dir
$ cmake ../src/
$ make
```

and then run the program:
```bash
$ hamc_Solution -h
```

## CPU based execution:

```bash
HAMC/Varad0612-C-Implementation$ make && ./run
Starting Encryption...
Input seed or -1 to use default seed: -1
MDPC code generated....
Time for H: 0.037965
Construction of G started...
Time for G: 6.039283
Generator matrix generated....
Time for H: 0.037504
Decryption successful...
Time taken by cryptosystem: 6.139059
```
### CPU based execution specs:
CPU: Intel(R) Core(TM) i5-4670K CPU @ 3.40GHz (1 Socket)

# Developers
* Mitchell Dzurick
* Mitchell Russel
* James Kuban
