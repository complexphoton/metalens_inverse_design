# In this file, we inverse design a metalens with wide feild of view (FOV).
# We change the permittivity of each design pixel inside the metalens,
# maximizing the objective function (the product of transmission efficiency and Strehl ratio for each incident angle within the FOV of interest) [See Eq.(1) of the paper].
# The gradient of the objective function is computed using adjoint method [See Eq.(3) of the paper].

include("asp.jl")
# The environmental variable for MUMPS3 (it should be the path to libraries of MUMPS)
ENV["MUMPS_PREFIX"] = "the path to libraries of MUMPS"

# Load the essential modules
using MESTI
using Random # Used for the repeatability of random numbers
using Interpolations # Perform interplations
using FFTW # Perform fast Fourier transform
using NLopt # Used to perform nonlinear optimization
using Printf
using LinearAlgebra
using SparseArrays # Use sparse matrices to improve computational efficiency
using MAT # Used to save data to *.mat file

# Set the seed of random number generator to generate different initial guesses for reproducibility
seed_rand = 01

################################# Define the simulation domain #####################################

n_air = 1.0         # Refractive index of air on the right
n_sub = 1.0         # Refractive index of substrate on the left
n_struct= 2.0       # Refractive index of silicon nitride
wavelength = 1      # Vacuum wavelength
dx = wavelength/40  # Discretization grid size [wavelength]
FOV = 60            # FOV of the metalens in the air [deg]

# Thickness of the metalens [wavelength]
h = 5.0*wavelength
# Output diameter of the metalens [wavelength]
D_out = 50*wavelength
# Input diameter of the metalens [wavelength]
D_in = 25*wavelength
# Numerical aperture NA
NA = 0.9
# Focal length [wavelength]
focal_length = D_out/2/tan(asin(NA))

# Parameters for the input and output
W_in  = D_in                  # Input window width [wavelength]
W_out = D_out + 2*wavelength  # Width where we sample the transmitted field [wavelength]
                              # (a larger W_out ensures all transmitted light is captured)

# W_out > D_out, so we will pad extra pixels
ny_R_extra_half = Int(round((W_out-D_out)/dx/2))

# Number of pixels in the y direction for the source (on the left) and the projection (on the right)
ny = Int(ceil(D_out/dx))
ny_L = Int(ceil(D_in/dx))
ny_R = ny + 2*ny_R_extra_half
# Number of pixels of the metasurface in the z direction
nz = Int(ceil(h/dx))

nPML = 20      # Number of pixels of PMLs
# PMLs on the z direction, add one pixel of free space for source and projection
nz_extra_left = 1 + nPML
nz_extra_right = nz_extra_left
# PMLs on the y direction
ny_extra_low = ny_R_extra_half + nPML
ny_extra_high = ny_extra_low
# Number of pixels for the whole system with PMLs
ny_tot = ny + ny_extra_low + ny_extra_high
nz_tot = nz + nz_extra_left + nz_extra_right

nyz = ny*nz
nyz_tot = ny_tot*nz_tot

k0dx = 2*pi/wavelength*dx   # Dimensionless frequency k0*dx
epsilon_L = n_sub^2         # Relative permittivity on the left
epsilon_R = n_air^2         # Relative permittivity on the right and on the top & bottom

# Obtain properties of propagating channels on the two sides.
BC = "periodic"  # Periodic boundary condition means the propagating channels are plane waves
use_continuous_dispersion = true  # Use discrete dispersion relation for (kx,ky)
channels_L = mesti_build_channels(ny, BC, k0dx, epsilon_L, nothing, use_continuous_dispersion)
channels_R = mesti_build_channels(ny_R, BC, k0dx, epsilon_R, nothing, use_continuous_dispersion)

# We use all propagating plane-wave channels on the right
N_R = channels_R.N_prop     # Number of channels on the right

# For the incident plane waves, we only take channels within the desired FOV:
# |ky| < n_air*k0*sin(FOV/2)
kydx_bound = n_air*k0dx*sind((FOV+2)/2) # Use a slightly wider FOV to make sure that all incident angles of interest are included
ind_kydx_FOV = findall(x-> (abs(x) < kydx_bound), channels_L.kydx_prop)
kydx_FOV = channels_L.kydx_prop[ind_kydx_FOV]   # kydx within the FOV
kxdx_FOV = sqrt.((k0dx)^2 .- (kydx_FOV).^2)     # kxdx within the FOV
N_L = length(kydx_FOV)   # Number of inputs

B_basis = channels_L.u_x_m(channels_L.kydx_prop)       # Build the propagating channels on the left of metalens
C_R = conj(channels_R.u_x_m(channels_R.kydx_prop))     # Build the propagating channels on the right of metalens

# Multiply the flux-normalized prefactor sqrt(nu)
# Note that N_R includes all propagating channels on the right, but N_L
# only includes propagating channels within the FOV on the left
sqrt_nu_L_basis = reshape(channels_L.sqrt_nu_prop,1,:)  # row vector
sqrt_nu_R = channels_R.sqrt_nu_prop                     # column vector

# We project each truncated input plane wave with |y|<=D_in/2 onto propagating channels and use their combinations as sources, such that the inputs satisfy the free-space wave equation.
y_list = (collect(0.5:1:ny) .- ny/2).*dx  
ind_source_out = findall(x-> (abs(x) > D_in/2), y_list)  # y coordinates outside of the input aperture
B_trunc = B_basis[:,ind_kydx_FOV]*sqrt(ny/ny_L)
B_trunc[ind_source_out,:] .= 0      # Block light outside of the input aperture

B_L = zeros(ComplexF64,ny,N_L)
for ii = 1:N_L
	for jj = 1:channels_L.N_prop
		Psi_in = sum(B_trunc[:,ii].*conj(B_basis[:,jj]),dims=1)
		B_L[:,ii] .= B_L[:,ii] .+ Psi_in*sqrt_nu_L_basis[jj]*exp((-1im*1/2)*channels_L.kzdx_prop[jj]).*B_basis[:,jj]
	end
end

# In mesti(), B_struct.pos = [m1, n1, h, w] specifies the position of a
# block source, where (m1, n1) is the index of the smaller-(y,z) corner,
# and (h, w) is the height and width of the block. Here, we put line
# sources (w=1) on the left surface (n1=n_L) and the right surface
# (n1=n_R) with height ny_L and ny_R centered around the metalens.
n_L = nz_extra_left                    # z pixel immediately before the metalens
n_R = n_L + nz + 1                     # z pixel immediately after the metalens
m1_L = ny_extra_low + 1                # first y pixel of the metalens
m1_R = nPML + 1                        # first y pixel of the output projection window

##################################### Angular Spectrum Propagation (ASP) #####################################
# System width used for ASP to remove periodic wrapping artifact.
W_ASP_min = 2*D_out  # Minimal ASP window [wavelength]

# Since we want to extract the intensity at the exact focal spot position [See Eq.(1)] later, we keep a finer sampling here
dy_ASP = 1*dx    # ASP grid size [wavelength]

# fft is more efficient when the length is a power of 2, so we make ny_ASP a power of 2.
ny_ASP = nextpow(2,Int(round(W_ASP_min/dy_ASP))) # Number of pixels for ASP
W_ASP = ny_ASP*dy_ASP                            # Actual ASP window [micron]

# y index of the points we down-sample for ASP (if any).
ind_ASP = Int.(collect(1:(dy_ASP/dx):ny_R))

# Make the sampling points symmetric around the middle.
if ind_ASP[end] != ny_R
    # Make sure that ny_R - ind_ASP[end] is an even number.
    if mod(ny_R - ind_ASP[end], 2) != 0
        ind_ASP = ind_ASP[1:(end-1)]
    end
    ind_ASP = Int.(ind_ASP .+ (ny_R .- ind_ASP[end])./2)
end

ny_ASP_pad = ny_ASP - length(ind_ASP)         # Total number of zeros to pad
ny_ASP_pad_low = Int(round(ny_ASP_pad/2))     # Number of zeros to pad on the low side
ny_ASP_pad_high = ny_ASP_pad - ny_ASP_pad_low # Number of zeros to pad on the high side

# y position of the ASP points, including the padded zeros [micron] 
y_ASP = (collect(0.5:ny_ASP) .- 0.5*(ny_ASP + ny_ASP_pad_low - ny_ASP_pad_high))*dy_ASP

ny_ASP_half = Int(round(ny_ASP/2))   # recall that ny_ASP is an even number

# List of (kx,ky) in ASP
# Note that after fft, ky_ASP will be (2*pi/W_ASP)*(0:(ny_ASP-1)), where the
# latter half needs to be unwrapped to negative ky. We can either use fftshift
# and iffshift while centering ky_ASP, or just unwrap ky_ASP itself; we do the
# latter here.
ky_ASP = (2*pi/W_ASP).*[collect(0:(ny_ASP_half-1)); collect(-ny_ASP_half:-1)]
kx_ASP = sqrt.(Complex.((n_air*2*pi/wavelength)^2 .- ky_ASP.^2))

# We only use the propagating components in ASP
kx_ASP_prop = kx_ASP[findall(x-> (abs(x) < (n_air*2*pi/wavelength)), ky_ASP)] # must be a column vector per asp() syntax

# List of incident angles in air [degree]
# n_air*sin(theta_in) = n_sub*sin(theta_sub), 
# where theta_sub = atan(channels_L.kydx/channels_L.kxdx)
theta_in_list = asind.(sin.(atan.(kydx_FOV./kxdx_FOV)).*n_sub./n_air) 

# Later we will use ifft to reconstruct field profile immediately after the
# metalens from the transmission matrix, i.e. Eq (S7) of the APF paper,
# and this is the prefactor we need.
prefactor_ifft = sqrt(ny_R).*exp.((-2im*pi/ny_R*(N_R-1)/2).*collect(0:(ny_R-1)))

# Focal spot position for each incident angle
focal_spot_list = focal_length.*tand.(theta_in_list)
interp_linear = linear_interpolation(y_ASP,collect(1:ny_ASP))
ind_plot = Int.(round.(interp_linear(focal_spot_list)))
# Extract field information at the desired focal spot position for each incident angle from ASP results
y_plot = y_ASP[ind_plot]

################################ Ideal focusing under different incident angles #####################################
# Focal field & focal intensity of an ideal metalens free of aberrations
focal_intensity_ideal = zeros(N_L,1)
focal_field_ideal = zeros(ny_ASP,N_L)
for ii = 1:N_L
    ideal_phase = -(2*pi/wavelength).*sqrt.(focal_length^2 .+ (y_list.-focal_spot_list[ii]).^2)
    # Ideal field on the exit surface of the metalens [See Eq.(S5) of the Supplementary document] 
    # Outside the metalens, the Ez_ideal is zero.
    Ez_ideal = [zeros(ny_R_extra_half,1); exp.(1im*ideal_phase)./(focal_length^2 .+(y_list.-focal_spot_list[ii]).^2).^(1/4); zeros(ny_R_extra_half,1)]
    
    # Project Ez_ideal onto the propagating channels on the right
    temp  = circshift(exp.(-1im*2*pi/ny_R.*collect(0:ny_R-1)).*fft(Ez_ideal), Int(floor(channels_R.N_prop/2)))./sqrt(ny_R)
    t_ideal = (channels_R.sqrt_nu_prop).*temp[1:channels_R.N_prop]
    
    # Use ASP to propagate Ez_ideal to the focal plane
    Ez0 = circshift(prefactor_ifft.*ifft([(1 ./sqrt_nu_R).*t_ideal; zeros(ny_R-N_R,1)],(1,)), -1)
    Ez0_ASP = Ez0[ind_ASP,:]
    Ez_focal_ideal = asp(Ez0_ASP, focal_length, kx_ASP_prop, ny_ASP, ny_ASP_pad_low)
    
    # Ideal focal field (set angle-dependent prefactor A in Eq.(S5) of the Supplementary document to ensure unitary transmission)
    focal_field_ideal[:,ii] = abs.(Ez_focal_ideal).^2/sum(abs.(t_ideal).^2)
    # Ideal focal intensity (See the denominator of Eq.(1))
    focal_intensity_ideal[ii] = maximum(abs.(Ez_focal_ideal))^2/sum(abs.(t_ideal).^2)
end

######################### Integrate ASP into projection matrix C (See Supplementary Sec.1) ###########################
pad_matrix_1 = [Matrix{Int64}(I, N_R, N_R); zeros(ny_R-N_R, N_R)]
num_ind_ASP = length(ind_ASP)
ind_ASP_matrix = zeros(num_ind_ASP, ny_R)
for ii = 1:num_ind_ASP
    ind_ASP_matrix[ii,ind_ASP[ii]] = 1
end

pad_matrix_2 = [zeros(ny_ASP_pad_high,num_ind_ASP); Matrix{Int64}(I, num_ind_ASP, num_ind_ASP); zeros(ny_ASP_pad_low,num_ind_ASP)]

ind_ASP_prop = findall(x-> (real(x)>0), kx_ASP)
num_ASP_prop = length(ind_ASP_prop)
ind_ASP_prop_matrix = zeros(num_ASP_prop, ny_ASP)
for ii = 1:num_ASP_prop
    ind_ASP_prop_matrix[ii,ind_ASP_prop[ii]] = 1
end

kx_prop_matrix = exp.(1im*focal_length.*kx_ASP_prop)
pad_matrix_3 = zeros(ny_ASP, num_ASP_prop)
pad_matrix_3[ind_ASP_prop,:] = Matrix{Int64}(I, num_ASP_prop, num_ASP_prop)

n_ind_prop = length(ind_plot)
ind_plot_matrix = zeros(n_ind_prop,ny_ASP)
for ii = 1:n_ind_prop
    ind_plot_matrix[ii,ind_plot[ii]] = 1
end

C_R_new = ind_plot_matrix*ifft(pad_matrix_3*(kx_prop_matrix.*ind_ASP_prop_matrix*fft(pad_matrix_2*ind_ASP_matrix*circshift(prefactor_ifft.*ifft(pad_matrix_1*transpose(C_R),(1,)), -1),(1,))),(1,))

# Compute inv(A)*[B, C^T] for both the forward simulation and gradient information (See Eq.(3) and Supplementary Sec.3 of the paper)
# 2-element structure array B_struct
B_struct = Source_struct()
B_struct.pos = [[m1_L, n_L, ny, 1], [m1_R, n_R, ny_R, 1]]
B_struct.data = [B_L, permutedims(C_R_new, (2,1))]

# Multiply matrix C to inv(A)*B, and get the focal field of an actual design at the desired focus positions
# First, we build matrix C
m2 = m1_R + ny_R - 1      # last index in y
n2 = n_R                  # last index in z
# Stack reshaped B_R with zeros to build matrix C.
nyz_before = (n_R-1)*ny_tot + m1_R - 1
nyz_after  = ny_tot*nz_tot - ((n2-1)*ny_tot+m2)
C_matrix = hcat(spzeros(N_L, nyz_before), C_R_new, spzeros(N_L, nyz_after))

############################### Initialize the simulation ############################################
# Update the permittivity profile with a macropixel size of 4*dx to reduce the dimension of the design space and keep the minimal feature size large for the ease of fabrication
# Build a metalens with inequal input and output aperture sizes
# Keep track of the indices of the updated pixels 
ny_opt = Int(ny/4)
ny_L_opt = Int(ny_L/4)
nz_opt = Int(nz/4)
increment_per_layer = (ny_opt - ny_L_opt)/2/(nz_opt-1)
num_struct = 0
for ii = 1:nz_opt
  global  num_struct += Int(round(ny_L_opt/2 + (ii-1)*increment_per_layer))
end
y_list_opt = (collect(0.5:1:ny_opt) .- ny_opt/2).*dx*4   # coordinates in the y direction
ind_struct_left = vec(zeros(num_struct,1))
num_struct_init = 0
for ii = 1:nz_opt
    ind_struct = findall(x-> (-x<=(D_in/2+increment_per_layer*(ii-1)*dx*4)), y_list_opt[1:Int(ny_opt/2)])
    ind_struct_left[num_struct_init.+(1:length(ind_struct))] = Base._sub2ind((ny_opt,nz_opt), ind_struct, ii*vec(Int.(ones(length(ind_struct),1))))
    global num_struct_init += length(ind_struct)
end
ind_struct_left = Int.(ind_struct_left)
                                    
ind_struct_right = vec(zeros(num_struct,1))
num_struct_init = 0
for ii = 1:nz_opt
    ind_struct = findall(x-> (-x<=(D_in/2+increment_per_layer*(ii-1)*dx*4)), y_list_opt[Int(ny_opt/2):-1:1])
    ind_struct_right[num_struct_init.+(1:length(ind_struct))] = Base._sub2ind((ny_opt,nz_opt), (Int(ny_opt/2)).+reverse(ind_struct,dims=1), ii*vec(Int.(ones(length(ind_struct),1))))
   global  num_struct_init += length(ind_struct)
end
ind_struct_right = Int.(ind_struct_right)

ind_epsilon_left = zeros(4^2*num_struct,1)
ind_epsilon_right = zeros(4^2*num_struct,1)
for ii = 1:num_struct
    epsilon_ind_opt = zeros(ny_opt,nz_opt)
    epsilon_ind_opt[ind_struct_left[ii]] = 1
    epsilon_ind = zeros(ny_tot,nz_tot)
    epsilon_ind[ny_extra_low.+(1:ny),nz_extra_left.+(1:nz)] = kron(epsilon_ind_opt,ones(4,4))
    ind_epsilon_left[(ii-1)*16 .+ (1:4^2)] = findall(x-> (x > 0), epsilon_ind[:])
    temp = getindex.(vec(reverse(reshape(findall(x-> (x > 0), [zeros(Int(ny_tot/2),nz_tot);reverse(epsilon_ind[1:Int(ny_tot/2),:],dims=1)]),(4,4)),dims=1)),[1 2])
    ind_epsilon_right[(ii-1)*16 .+ (1:4^2)] = Base._sub2ind((ny_tot,nz_tot),temp[:,1],temp[:,2])
end
ind_epsilon_left = vec(Int.(ind_epsilon_left))
ind_epsilon_right = vec(Int.(ind_epsilon_right))
                                                    
# Generate random initial guesses 
epsilon = rand(MersenneTwister(seed_rand),Float64,(num_struct,1))

# Construct the permittivity profile over all simulation domain (including
# metalens, PML, and extra padded pixels)
epsilon_real_opt = ones(ny_opt,nz_opt)
epsilon_real_opt[ind_struct_left] = epsilon*n_struct^2 .+ (1 .- epsilon)*n_air^2
epsilon_real = ones(ny_tot,nz_tot)
epsilon_real[ny_extra_low.+(1:ny),nz_extra_left.+(1:nz)] = kron(epsilon_real_opt,ones(4,4)) # Use a finer discretization of dx compared to the minimal feature size 4*dx for accurate simulations

# The metalens is mirror symmetric with respect to y=0
syst = Syst()
syst.epsilon_xx = [epsilon_real[1:Int(ny_tot/2),:]; reverse(epsilon_real[1:Int(ny_tot/2),:],dims=1)]
syst.length_unit = "µm"
syst.wavelength = wavelength
syst.dx = dx
syst.PML = [PML(nPML)]  # Number of PML pixels (on all four sides)

# Use MUMPS with single-precision arithmetic to improve the computational efficiency
opts = Opts()
opts.method = "FS"
opts.solver = "MUMPS"
opts.use_single_precision_MUMPS = true

# Use MESTI to run simulations of the metalens, and propagate to the desired focal plane via ASP
B_struct_i = Source_struct()
C_struct_i = Source_struct()
B_struct_i.pos = [[m1_L, n_L, ny, 1]]
B_struct_i.data = [B_L]
C_struct_i.pos = [[m1_R, n_R, ny_R, 1]]
C_struct_i.data = [permutedims(C_R_new, (2,1))]
(T_init, info) = mesti(syst, [B_struct_i], [C_struct_i], opts)
T_init = T_init*(-2im)

# Build the initial guess of the optimizations (dummy variable g and permittivity) [see Eqs.(1-2) of the paper]
# The first element is the dummy variable g <= I_a
x_init = [minimum(abs.(diag(T_init)).^2 ./focal_intensity_ideal); vec(epsilon)]

# Factors related to the binarization (See Supplementary Sec.5)
norm_norm = norm(ones(num_struct,1) .- 0.5)
max_eval = 400

######################### Define the objective function and inequality constraints #####################
# Define the objective function with a binarization regularizer (See Eq.(2) and Supplementary Sec.5) and its gradient
function obj_func(x::Vector{Float64}, grad::Vector{Float64}, norm_norm, max_eval)
   # Gradient of the objective function + regularizer
   if length(grad) > 0
       grad[:] = vec(zeros(length(x),1))
       grad[1] = 1
       if opt.numevals < max_eval
          grad[2:end] = 2*(opt.numevals/max_eval)^2/norm_norm^2*(x[2:end] .- 0.5)
       else
          grad[2:end] = 2/norm_norm^2*(x[2:end] .- 0.5)
       end
   end
   # Objective function + regularizer
   if opt.numevals < max_eval
       @printf("two-material factor = %7.6f\n", norm(x[2:end] .- 0.5)/norm_norm)
       return x[1] + 1*(opt.numevals/max_eval)^2*(norm(x[2:end] .- 0.5)/norm_norm)^2  # x[1] is a dummy variable determined by the nonlinear inequality
   else
       @printf("two-material factor = %7.6f\n", norm(x[2:end] .- 0.5)/norm_norm)
       return x[1] + 1*(norm(x[2:end] .- 0.5)/norm_norm)^2  # x[1] is a dummy variable determined by the nonlinear inequality
   end
end

# Define inequality constraints and its gradient (See Eqs.(2-3) and Supplementary Sec.3)
function constraint_func(result::Vector{Float64}, x::Vector{Float64}, grad::Matrix{Float64}, syst::Syst, B::Vector{Source_struct}, C_matrix::Union{SparseMatrixCSC{Int64,Int64},SparseMatrixCSC{Float64, Int64},SparseMatrixCSC{ComplexF64,Int64}}, opts::Opts, n_struct::Float64, n_air::Float64, k0dx::Union{Float64,ComplexF64}, ny::Int64, nz::Int64, ny_extra_low::Int64, nz_extra_left::Int64, ny_tot::Int64, nz_tot::Int64, N_L::Int64, sqrt_nu_R::Vector{Float64}, focal_intensity_ideal::Matrix{Float64},ind_struct_left::Vector{Int64},ind_struct_right::Vector{Int64},ind_epsilon_left::Vector{Int64},ind_epsilon_right::Vector{Int64})
    # Gradient of the nonlinear constraints (Eq.(3) and Supplementary Sec.3 of the paper)
    if length(grad) > 0
        # Build the permittivity profile of the metalens
	    epsilon_real_opt = ones(ny_opt,nz_opt)
        epsilon_real_opt[ind_struct_left] = x[2:end]*n_struct^2 .+ (1 .- x[2:end])*n_air^2
        epsilon_real = ones(ny_tot,nz_tot)
        epsilon_real[ny_extra_low.+(1:ny),nz_extra_left.+(1:nz)] = kron(epsilon_real_opt,ones(4,4))
        syst.epsilon_xx = [epsilon_real[1:Int(ny_tot/2),:]; reverse(epsilon_real[1:Int(ny_tot/2),:],dims=1)] 

	    num_ind_left = length(ind_struct_left)

        opts.verbal = false   # Surpress verbal outputs
        # Compute inv(A)*[B,C^T]
        (field, info) = mesti(syst, B, opts)

        nyz_tot = ny_tot*nz_tot
        field = reshape(field, nyz_tot, N_L*2)

        AB = field[:, 1:N_L]*(-2im)  # Extract inv(A)*B
        AC = field[:, N_L+1:end]     # Extract inv(A)*C^T
        field = nothing              # clear field

        # Get T = C*inv(A)*B, get the focal field of an actual design at the desired focus positions
        T_ang = C_matrix*AB

        # Get the gradient with respect to all optimization variables and incident angles
        AB_design_1 = reshape(AB[ind_epsilon_left,:],4^2,num_ind_left,N_L)
        AC_design_1 = reshape(AC[ind_epsilon_left,:].*(diag(T_ang)./focal_intensity_ideal)',4^2,num_ind_left,N_L)
        AB_design_2 = reshape(AB[ind_epsilon_right,:],4^2,num_ind_left,N_L)
        AC_design_2 = reshape(AC[ind_epsilon_right,:].*(diag(T_ang)./focal_intensity_ideal)',4^2,num_ind_left,N_L)
        
        grad .= 1 .*ones(length(ind_struct_left)+1,N_L)
        grad[2:end,:] = -2*k0dx^2*(n_struct^2-n_air^2)*dropdims(real(sum(AC_design_1.*AB_design_1,dims=1)).+real(sum(AC_design_2.*AB_design_2,dims=1));dims=1)
    end
    # Nonlinear constraints (Eq.(2) of the paper)
    result[:] = vec(-abs.(diag(T_ang)).^2 ./focal_intensity_ideal .+ x[1])
    @printf("FoM = %7.6f, t = %7.6f at Iteration %d\n", minimum(abs.(diag(T_ang)).^2 ./focal_intensity_ideal), x[1], opt.numevals)

end

###################### Perform the optimization using NLopt #####################################
opt = Opt(:LD_MMA, num_struct+1)                       # Specify the optimization algorithm: MMA
opt.lower_bounds = [0; vec(zeros(num_struct,1))]       # Lower bounds of optimization variables
opt.upper_bounds = [1; vec(ones(num_struct,1))]        # Upper bounds of optimization variables
opt.xtol_abs = [1e-7; 1e-20*vec(ones(num_struct,1))]   # Absolute tolerance of optimization variables
opt.maxtime = 1*16*60*60                               # Maximal optimization time

# Build nonlinear constraints
opt.max_objective = (x,g) -> obj_func(x,g,norm_norm,max_eval)
inequality_constraint!(opt, (r,x,g) -> constraint_func(r,x,g,syst,[B_struct],C_matrix,opts,n_struct,n_air,k0dx,ny,nz,ny_extra_low,nz_extra_left,ny_tot,nz_tot,N_L,sqrt_nu_R,focal_intensity_ideal,ind_struct_left,ind_struct_right,ind_epsilon_left,ind_epsilon_right), 1e-2*vec(ones(N_L,1)))

# Run optimizations using Nlopt
(maxf,maxx,ret) = optimize!(opt, x_init)
numevals = opt.numevals # the number of function evaluations
println("got $maxf after $numevals iterations (returned $ret)")

                                                                                        
###################################### Obtain the optimized results #####################################
# Get I_a = SR_a*T_a for each incident angle [Eq.(1) of the paper]
epsilon_real_opt = ones(ny_opt,nz_opt)
epsilon_real_opt[ind_struct_left] = maxx[2:end]*n_struct^2 .+ (1 .- maxx[2:end])*n_air^2
epsilon_real = ones(ny_tot,nz_tot)
epsilon_real[ny_extra_low.+(1:ny),nz_extra_left.+(1:nz)] = kron(epsilon_real_opt,ones(4,4))
syst.epsilon_xx = [epsilon_real[1:Int(ny_tot/2),:]; reverse(epsilon_real[1:Int(ny_tot/2),:],dims=1)]
C_struct_i.data = [permutedims(C_R_new, (2,1))]
(T_fin, info) = mesti(syst, [B_struct_i], [C_struct_i], opts)
T_fin = T_fin*(-2im)
FoM_opt = abs.(diag(T_fin)).^2 ./focal_intensity_ideal

# Compute the transmission efficiency T_a for each incident angle [Supplementary Eq.(44) of the APF paper]
C_struct_i = Source_struct()
C_struct_i.pos = [[m1_R, n_R, ny_R, 1]]
C_struct_i.data = [C_R]
(T_eff, info) = mesti(syst, [B_struct_i], [C_struct_i], opts)
T_eff = sqrt_nu_R.*T_eff*(-2im)
T_actual = sum(abs.(T_eff).^2, dims=1)

# Get the Strehl ratio for each incident angle
SR = FoM_opt./vec(T_actual)

# Save optimized data
matwrite("optimized_data.mat", Dict(
    "FoM_opt" => FoM_opt,
    "T_actual" => T_actual,
    "SR" => SR,
    "theta_in_list" => theta_in_list
); compress = true)