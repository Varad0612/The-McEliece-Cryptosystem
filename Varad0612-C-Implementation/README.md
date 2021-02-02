# The-McEliece-Crytosystem
This is the C implementation of the McEliece Crytosystem based on QC-MDPC codes.
To compile and run the program:
1. Go to the directory containing the files
2. Type the following commands in the command prompt:<br />

$ make <br />
$./run <br />

When prompted for seed, enter a large number (>10^9) or enter -1 for a random seed value.

For the input parameters n0 = 2, p = 4800, t = 84, w = 90, the program takes about 15 mins to run the encryption
and decryption algorithms.

matrix_mult() and matrix_inverse() methods in the matrix.c files are the most time consuming steps.

You can also try running the program on smaller input parameters for observation and debugging purposes. (n0 = 2, p = 500, t = 10, w = 30)

To modify the input, edit the test.c file.

The message here is a random vector of length 'k'.

The following targets are provided: keygen, encrypt, decrypt

To initialize the system:<br />
$ make init<br />
$./init<br />
$Enter the parameters<br />

Enter 0, 1, 2 or 3 depending on the action.

To genrate keys:<br />
$ make keygen<br />
$./keygen<br />

The public and the private keys will be stored in the files Public_Key.txt and Private_Key.txt respectively.

To encrypt:<br />
$ make encrypt<br />
$./encrypt<br />

Result stored in Encryption.txt.

To decrypt:<br />
$ make decrypt<br />
$./decrypt<br />

Result stored in Decryption.txt.
