# metalens_inverse_design

This repository contains code for the inverse design of a metalens with a wide field of view (FOV). Each file is annotated with detailed comments for ease of understanding.

* `inverse_design_WFOV_metalens.jl`: Build, simulate and optimize a metalens
* `asp.jl`: Implement angular spectrum propagation for field transmission from the metalens exit surface to the focal plane  


To conduct efficient optimizations, follow these steps:
* Install [MESTI.jl](https://github.com/complexphoton/MESTI.jl), an open-source software for full-wave electromagnetic simulations. Prior to this, install the parallel version of [MUMPS](https://mumps-solver.org/index.php) for efficient matrix handling. For other required packages, please refer to the comments of `inverse_design_WFOV_metalens.jl`.
* Install [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) for access to well-developed optimization algorithms.
* Run `inverse_design_WFOV_metalens.jl`  


An animation demonstrating the evolution of both the metalens and its focusing performance is shown below.
<img align="center" src="https://github.com/complexphoton/metalens_inverse_design/blob/main/code/animated_opt.gif" width=70% height=70%>

The related paper will soon be available on arXiv.
