# Day 2: Foundations - GPU Concepts, HPC Intro, CPU Kernels

Date: 2025-07-01

Room: G53

Time: 09:30 - 17:00, with breaks at least once an hour.

## Morning (09:00 - 12:00)

### (09:00 - 09:30) Arrival and settling

### (09:30 - 10:00) Welcome and introductions

#### Activity

- Self-introduction (e.g. What do I do in Diamond? How I end up here?)
- Students introduce themselves (background etc.) and their interests briefly.
- Brief overview of the work experience project.

#### Materials

- Note about project goals and what we will achieve (welcome.pdf)
- A schedule overview (schedule.pdf)
- Few simple questions to prompt students' introduction (intro\_prompts.pdf)

### (10:00 - 11:00) Introduction to GPU Architecture & CUDA Programming Model

#### Activity

- Go through the prepared interactive tutorial/materials on GPU concepts (SPMD/SIMD, Host vs. Device, SMs, threads, warps).
- Understanding facilitated by discussion, answering questions and analogies.
- Watch 1-2 short supplementary videos.

#### Materials

- Interactive Jupyter notebook tutorial covering GPU concepts with visualisations (gpu\_concept.ipynb)
- Slide on GPU architecture (gpu\_architecture.html)
- Analogy about parallel processing in GPU (gpu\_analogy.pdf)
- GPU terminology glossary (gpu\_glossary.pdf)
- Comparison between CPU vs GPU characteristics (cpu\_gpu\_comparison.pdf)
- Links of some educational videos (video\_link.txt)

### (11:00 - 12:00) Introduction to High-Performance Computing (HPC) System

#### Activity

Students use the prepared guide to log into the HPC system.

- Practice basic terminal commands: ssh, cd, ls, pwd.
- Learn to use nano to create a dummy text file.
- Explain the concept of a queuing system (Slurm).

#### Materials

- Terminal basic navigation reference (terminal\_cmd\_reference.pdf)
- Nano editor quick reference (nano\_reference.pdf)
- Slurm commands reference (slurm\_reference.pdf)
- Practice exercise sheet with terminal navigation tasks (terminal\_exercises.pdf)

## Lunch Break (12:00 - 13:00)

## Afternoon (13:00 - 17:00)

### (13:00 - 14:00) First CUDA Program on HPC

#### Activity

Students are given the pre-written cupy "hello world" raw kernel example.

- Learn to submit it using sbatch (with a simple submission script provided).
- Learn to check job status (squeue) and view output (cat/tail).
- Modify a very simple part of the "hello world" (e.g., the constant value
being added, the size of the array) and re-run.

#### Materials

- Complete CuPy raw kernel example with extensive comments (gpu\_hello\_world.py
and kernel\_add\_constant.cu)
- Same CuPy raw kernel for addding numbers to bigger array to demonstrate
speed-up (gpu\_big\_array.py)
- CuPy raw kerenl for multiplication to demonstrate differet operation
(gpu\_multiply.py and kernel\_multiply.cu)
- Template Slurm submission script with explanations (submit\_job.sh)
- Practical modification guide for submitting Slurm jobs
(slurm\_modification.pdf)

### (14:00 - 16:30) Understanding Kernels on the CPU (Part 1)

#### Activity

Introduces the concept of a "kernel" as the core computation repeated for each
data element.

- Students work with the prepared notebooks in CPU for 1D array operations
(e.g. adding a constant).
- Run the notebook, observe input/output.
- Examine the code, focusing on the loop and the "per-element" calculation and
helper functions.
- Modify parameters (e.g. the constant value, array size) and observe effects.

Introduce 1D median/Gaussian filter and pixel binning on the CPU.

- Examine the code, especially the windowing/neighborhood logic and edge
handling.
- Modify parameters (window size, Gaussian sigma, binning factor).

#### Materials

- Notebook showing the concept of kernel in 1D by adding constant to an array
(cpu\_kernel\_add\_constant.ipynb)
- Notebook showing the concept of different 1D kernel operations
(cpu\_kernels\_1d\_basic.ipynb)
- Notebook showing the implementation of 1D median filter with edge handling
(cpu\_median\_filter\_1d.ipynb)
- Notebook showing the implementation of 1D Gaussian filter with edge handling
(cpu\_gaussian\_filter\_1d.ipynb)
- Notebook showing the implementation of 1D pixel binning with signal-to-noise
ratio calculation (cpu\_binning\_1d.ipynb)

### (16:30 - 17:00) Q&A and wrap-up

Review what was learnt.

- What do you think about today? (overwhelming/too complicated or easy)
- Address any outstanding questions.
- Briefly outline Day 3 (morning only).

# Day 3: GPU Kernels

Date: 2025-07-02

Room: MR 1-10 (i14 external building)

Time: 09:00 - 12:00, with breaks at least once an hour.

## Morning (09:00 - 12:00)

### (09:00 - 09:30) Arrival and settling

### (09:30 - 10:30) Understanding Kernels on the CPU (Part 2)

#### Activity

Students work with the prepared notebooks for 2D image operations
(e.g., brightness adjustment, median filter, Gaussian filter, pixel binning on
small dummy images).

- Focus on how 2D loops translate to processing each pixel.
- Discuss 2D windowing and edge handling strategies.
- Visualise "before" and "after" images using matplotlib.
- Modify parameters and observe effects.

#### Materials

- Notebook showing the concept of kernel in 2D by adjusting brightness
(cpu\_brightness\_adjust.ipynb)
- Notebook showing the concept of different 2D kernel operations
(cpu\_kernels\_2d\_basics.ipynb)
- Notebook showing the implementation of 2D median filter with edge handling
(cpu\_median\_filter\_2d.ipynb)
- Notebook showing the implementation of 2D Gaussian filter with edge handling
(cpu\_gaussian\_filter\_2d.ipynb)
- Notebook showing the implementation of 2D pixel binning with signal-to-noise
ratio calculation (cpu\_binning\_2d.ipynb)

### (10:30 - 11:30) Transition to GPU Kernels with CuPy

#### Activity

Review the CuPy "hello world" raw kernel, specifically the C kernel string.

- Introduce basic C syntax needed within the kernel (variable declaration,
arithmetic).
- Explain CUDA thread indexing: threadIdx.x, blockIdx.x, blockDim.x etc.
- Explain how each thread maps to a data element.

#### Materials

- Essential C syntax for CUDA kernel (cuda\_c\_syntax.tex)
- Notebook to visualise thread indexing (threads\_indexing.ipynb)
- Notebook to illustrate components of CUDA kernel with examples
(kernel\_anatomy.ipynb)

### (11:30 - 12:00) Q&A and wrap-up

Review what was learnt.

- What do you think about today? (overwhelming/too complicated or easy)
- Address any outstanding questions.
- Briefly outline Day 4.

## Lunch Break (12:00 - 13:00)

## No afternoon work

# Day 4: GPU Kernels for Image Processing

Date: 2025-07-03

Room: G53

Time: 09:00 - 17:00, with breaks at least once an hour.

## Morning (09:00 - 12:00)

### (09:00 - 09:30) Arrival and settling

### (09:30 - 12:00) 2D GPU Kernels

#### Activity

Students adapt CPU 1D and 2D kernels to write 2D CuPy GPU raw kernels for
common image processing operations.

- Focus on translating the "per-element" Python logic into the C kernel string,
using thread indexing to select the element.
- Run on HPC, verify correctness against CPU version.

#### Materials

For each of the following 2D operations, there is a template for students to
start with and a completed script as a reference:

- brightness adjustment
- median filter
- Gaussian filter
- pixel binning

## Lunch Break (12:00 - 13:00)
