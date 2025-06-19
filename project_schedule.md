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
