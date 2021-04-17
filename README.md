# Robust_PCA_tensorflow
Because the algorithm of Robust PCA need iteration to converge, the speed processed in CPU is slow. We run the Robust PCA on GPU,whose speed will promote 10 times.

inexact_augmented_lagrange_multiplier.py : include two function to achieve the iteration of RobustPCA include CPU version (inexact_augmented_lagrange_multiplier) and tensorflow version(inexact_augmented_lagrange_multiplier_tf).

if you want to use the GPU version function, please replace the 74th code with   A, E = inexact_augmented_lagrange_multiplier_tf([X:,:sz])
