# The-McEliece-Crytosystem
This is the C implementation of the McEliece Crytosystem based on QC-MDPC codes.
To compile and run the program:
1. Go to the directory containing the files
2. Type the following commands in the command prompt:
        >>make
        >>./run

For the input parameters n0 = 2, p = 4800, t = 84, w = 90, the program takes about 15 mins to run the encryption
and decryption algorithms.

matrix_mult() and matrix_inverse() methods in the matrix.c files are the most time consuming steps.

You can also try running the program on smaller input parameters for observation and debugging purposes. (n0 = 2, p = 500, t = 10, w = 30)

To modify the input, edit the test.c file.

The message here is a random vector of length 'k' and is defined in the test.c file.

The following targets are provided: keygen, encrypt, decrypt

To initialize the system:
>> make init
>>./init
>>Enter the parameters
>>Enter 0, 1, 2 or 3 depending on the action

To genrate keys:
>> make keygen
>>./keygen

The public and the private keys will be stored in the files Public_Key.txt and Private_Key.txt respectively.

To encrypt:
>> make encrypt
>>./encrypt

Result stored in Encryption.txt

To decrypt:
>> make decrypt
>>./decrypt

Result stored in Decryption.txt
