# PE2DAAPE2D
Permutation entropy and Amplitude-aware permutation entropy 2D

Entropy algorithms can be used to study two-dimensional (2D) data. These can be used to study the texture and irregularity properties of images.
Two-dimensional permutation entropy (PE2D) is based on estimating the probability associated with permutation patterns that can be extracted from the image. In addition, an alternative method can be two-dimensional amplitude-aware permutation entropy (AAPE2D) that uses amplitude variations to obtain the probability associated with each permutationn pattern.

These algorithms are implemented in Python and require the use of the numba package. PE2D only has as inputs the image that is being studied (X) and the embedding dimension (m) parameter that defines the size of the permutation patterns (m^2). Besides these, AAPE2D requires the definition of a third input, the coefficient A that determines the balance of considering the samples' average and absolute difference values of these samples within the extracted patterns. 
