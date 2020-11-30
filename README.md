# BeamPropagation.jl
This repository implements simulation of particle trajectories in Julia.

# Installation
The package can be installed by running
```
    ] add https://github.com/christian-hh/BeamPropagation.jl
```
in the REPL. The `]` command activates Julia's package manager. Alternatively, use
```
    using Pkg; Pkg.clone(https://github.com/christian-hh/BeamPropagation.jl)
```

# Usage
The main function of the code is `propagate!`, whose signature is shown below.
```
propagate!(rs, vs, as, gs, f, dead, is_dead, η, dt, max_steps, p)
    rs:         array of initial positions (usually defined as static vectors)
    vs:         array of initial velocities
    as:         array of initial accelerations
    gs:         array of initial ground states
    f:          function to update a particle's acceleration and ground state
    dead:       array indicating which particles are "dead"
    is_dead:    function to define when particles "die"
    η:          specifies number of time steps between removing "dead" particles
    dt:         time step
    max_steps:  maximum number of time steps
    p:          parameters to be passed to the propagation
```

The use of the function is illustrated through the example below, where the trajectories of particles experiencing magneto-optical forces ("Zeeman-Sisyphus") are simulated. We first load the `Propagation` module and define constants specific to the example.
```
using Propagation

# Define physical constants
const h = 6.63e-34
const ħ = h / 2π
const M = 190 * 1.66e-27
const λ = 577 * 1e-9
const μ = 9.27e-24
const Γ = 2π * 8.3e6
const g = -9.8
const Iₛ = 5.0
const w = 0.5
const P = 250.0
const P0 = 2P / (Iₛ * π * w^2)
```

Next, we define arrays of type `SVector` (via the package `StaticArrays`), one for each of positions (`rs`), velocities (`vs`), and accelerations (`as`). They are initialized to the desired initial values of the particles. `n` indicates the number of particles to be simulated.
```
n = 10000

r_transform(r) = @. 2.5e-3 * (2r - 1)
rx = r_transform.(rand(Float64, n))
ry = r_transform.(rand(Float64, n))
rz = zeros(n)

v_transform(v) = @. 4.0 * (2v - 1)
vx = v_transform.(rand(Float64, n))
vy = v_transform.(rand(Float64, n))
vz = randn(n) .* (30 / 2.35) .+ 70

rs = SVector.(rx, ry, rz)
vs = SVector.(vx, vy, vz)
as = SVector.(zeros(n), zeros(n), zeros(n))
```

Two additional arrays are also required to indicate the initial ground states of the particles (`gs`) and which particles are "dead" (`dead`). For this example, we assume that there are 12 ground states (and 4 excited states), and that the particles have uniform probability of starting in any one of them.
```
num_gs = 12
num_es = 4
gs = rand(1:num_gs, n)
dead = [false for i = 1:n]
```

All particles are initially alive, but particles may be "removed" once they are no longer of interest (e.g., they may collide with a limiting aperture in the propagation path). This is to be specified through a function `is_dead`, which must return `true` if a particle is dead. In this example, we require particles to stay within a defined radius, and we stop propagating their trajectories once they have reached `z = 1.2` meters.
```
@inline function is_dead(r)
    x, y, z = r
    return (x^2 + y^2 > 5.76e-6) | (z > 1.2)
end
```
(The `@inline` macro here ensures that the tuple `x, y, z` is not allocated when `is_dead` is called.)

The magnetic field and its gradient are loaded to pre-allocated arrays from a pre-computed text file. These arrays may then be used to update the magnetic field at the location of particles throughout its propagation.
```
Barr = readdlm("./Bfieldmap.txt")
xlut   = Barr[:,1];         ylut   = Barr[:,2];         zlut   = Barr[:,3];
Bxlut  = Barr[:,4];         Bylut  = Barr[:,5];         Bzlut  = Barr[:,6];
dBxlut = Barr[:,7] ./ 100;  dBylut = Barr[:,8] ./ 100;  dBzlut = Barr[:,9] ./ 100;

dFlut = Array{SVector{3, Float64}}(undef, 51, 51, 201)
Bnormlut = zeros(Float64, (51, 51, 201))
for (i, (x, y, z)) in enumerate(zip(xlut, ylut, zlut))

    # Create indices for mapping a position (x, y, z) to a set of corresponding indices (ix, iy, iz)
    ix = round(Int, 1e4 * x + 25) + 1
    iy = round(Int, 1e4 * y + 25) + 1
    iz = mod(round(Int, z / 20e-5), 201) + 1

    dFlut[ix, iy, iz] = SVector(dBxlut[i], dBylut[i], dBzlut[i]) * (μ / M)
    Bnormlut[ix, iy, iz] = sqrt(Bxlut[i]^2 + Bylut[i]^2 + Bzlut[i]^2)
end
```

In this next code block, a function `scatterphoton` is defined to simulate a photon scattering event from an optical field. This is to be called at each step throughout a particle's propagation.
```
# Array specifying the transition intensity from the ground to excited states
TItab = [0 0.0003 0 0.0833 0 0.1656 0 0.0833 0 0.0008 0 0;
         0.0835 0 0.0830 0 0.0003 0 0.0007 0 0 0 0.1657 0;
         0 0.1662 0 0.0007 0 0.0004 0 0 0 0.0827 0 0.0833;
         0 0 0.0007 0 0.1662 0 0.0829 0 0.0831 0 0.0004 0]

# Array specifying the branching ratios of the excited to ground states
BRtab = [0 0.001 0 0.167 0 0.331 0 0.167 0.333 0.002 0 0;
         0.167 0 0.166 0 0.001 0.002 0.001 0 0 0.332 0.331 0;
         0.333 0.332 0 0.001 0 0.001 0 0 0 0.165 0 0.167;
         0 0.001 0.001 0.332 0.332 0 0.166 0 0.166 0 0.001 0]

# Calculate the total transition intensity from each ground state
const TI_total = sum(TItab, dims=1)

# Indicate the sign of the g-factor corresponding to each ground state
const μ_signs = SVector(-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1)

# Calculate probabilities of g -> g′ given TItab and BRtab, and convert to an array of weights vectors
# (one for each g) that can be sampled from (using `StatsBase.sample`)
using StatsBase
TIBRtab = zeros(Float64, num_gs, num_gs);
for g in 1:num_gs, g′ in 1:num_gs
    for e in 1:num_es
        TIBRtab[g, g′] += TItab[e, g] * BRtab[e, g′]
    end
end
const wvs = Array{Weights{Float64,Float64,Array{Float64,1}}}(undef, num_gs)
for g in 1:num_gs
    wvs[g] = Weights(TIBRtab[g, :])
end

@inline Rsc(s, δ) =  (Γ / 2) * (s / (1 + s + 4(δ/Γ)^2))     # scattering rate
@inline detuning(Δ0, UB, vz) = 2π * (Δ0 + UB / h + vz / λ)  # detuning (w/ Zeeman and Doppler shifts)
@inline Prob(g, s, δ, dt) = Rsc(s, δ) * dt * TI_total[g]    # probability to scatter a photon

@inline function scatterphoton(g, s, δ, dt, rnd)::Tuple{Bool, Int64}
    if rnd > Prob(g, s, δ, dt)
        return false, g
    else
        g′ = sample(end_gs, wvs[g])
        return true, g′
    end
end

# Specify detunings of optical fields
δs = [-13.75e9, 2.5e9];
```
(Note that functions are again defined inline to prevent allocation.)

Any parameters to the simulation are to be passed as a `NamedTuple`. The macro `@param` may be used to easily define this variable. Individual parameters may then be called with the usual `NamedTuple` syntax, `p.<parameter name>`.
```
p = @params (dFlut, Bnormlut, δs)
```

The parameters and the `scatterphoton` function are finally used to define the function `f`, which computes the instantaneous acceleration (`a′`) and current ground state (`g′′`) after two potential photon scattering events (in this example, there are two optical fields).
```
@inline function f(r, v, a, g, dt, p)
    x, y, z = r

    # Compute "lookup" indices corresponding to the current particle position
    # These are used to index into any arrays utilized -- here `Bnormlut` and `dFlut`
    ix = round(Int, 1e4 * x + 25) + 1
    iy = round(Int, 1e4 * y + 25) + 1
    iz = mod(round(Int, z / 20e-5), 201) + 1

    s = P0 * exp(-(x^2 + y^2) / (2(w/100)^2))
    UB = μ * μ_signs[g] * p.Bnormlut[ix, iy, iz]
    d1 = detuning(p.δs[1], UB, v[3])
    scat1, g′  = scatterphoton(g, s, d1, dt, rand())
    d2 = detuning(p.δs[2], UB, v[3])
    scat2, g′′ = scatterphoton(g′, s, d2, dt, rand())
    a′ = μ_signs[g] * p.dFlut[ix, iy, iz]

    return (a′, g′′)
end
```


```
η = 35; @btime propagate!(rs_, vs_, as_, gs_, dead_, $dt, $max_steps, $η, $is_dead, $f, $p) setup=(rs_=deepcopy($rs); vs_=deepcopy($vs); as_=deepcopy($as); gs_=deepcopy($gs); dead_=deepcopy($dead)) evals=1;
```
