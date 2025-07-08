The repository hosts the information and scripts for the work experience
project "Data analysis: Scientific image processing by Graphics Processing Unit
(GPU)" at Diamond Light Source.

These materials were developed in 2025 when I was preparing this work experience
project for the first time. This serves as a reference for people who want to
organise it in the future (which I strongly encourage you to do so!).

The scripts and notebooks were written in a way that the logic in CPU
operations, such as addition of a number to an an array, can be easily
translated to corresponding CUDA C codes for educational purposes. You will
find for-loop iterates on every element of a NumPy array and process the entry
individually. This is NOT how you should use NumPy normally.

The implementation of different image filters is not necessarily the most
optimised version as the focus is on the understanding of convolution kernels.

AI tools have been used to prepare some of the materials, in particular layout
of the TeX documents and the HTML presentation. I remain solely responsible for
all the mistakes. Pull requests are more than welcomed to improve the
materials!
