using Revise
using
    BeamPropagation, StaticArrays, DelimitedFiles, BenchmarkTools, Plots, StructArrays, StatsBase

# Define constants
const h = @with_unit 1 "h"
const ħ = @with_unit 1 "ħ"
const μ = @with_unit 1 "μB"
const M = @with_unit 190 "u"
const λ = @with_unit 577 "nm"
const Γ = @with_unit 2π * 8.3e6 "MHz"
const Iₛ = @with_unit 5.0 "mW/cm^2"
const w = @with_unit 0.5 "cm"
const P = @with_unit 250.0 "mW"

# Define parameters
n           = 1000000
dt          = 5e-7
max_steps   = 1e5

# Generate initial positions and velocities
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
gs = rand(1:12, n)

# Define a function `magnetic_force` that maps particle → force (based on a combination or `v` and `v`)
Barr = readdlm("examples/Bfieldmap.txt")
xlut   = Barr[:,1];         ylut   = Barr[:,2];         zlut   = Barr[:,3];
Bxlut  = Barr[:,4];         Bylut  = Barr[:,5];         Bzlut  = Barr[:,6];
dBxlut = Barr[:,7] ./ 100;  dBylut = Barr[:,8] ./ 100;  dBzlut = Barr[:,9] ./ 100;

dFlut = Array{SVector{3, Float64}}(undef, 51, 51, 201)
Bnormlut = zeros(Float64, (51, 51, 201))
for (i, (x, y, z)) in enumerate(zip(xlut, ylut, zlut))

    ix = round(Int, 1e4 * x + 25) + 1
    iy = round(Int, 1e4 * y + 25) + 1
    iz = mod(round(Int, z / 20e-5), 201) + 1

    dFlut[ix, iy, iz] = SVector(dBxlut[i], dBylut[i], dBzlut[i]) * (μ / M)
    Bnormlut[ix, iy, iz] = sqrt(Bxlut[i]^2 + Bylut[i]^2 + Bzlut[i]^2)
end

@inline function is_dead(r)
    x, y, z = r
    return (x^2 + y^2 > 5.76e-6) | (z > 1.2)
end

num_gs = 12; num_es = 4;

TItab = [0 0.0003 0 0.0833 0 0.1656 0 0.0833 0 0.0008 0 0;
         0.0835 0 0.0830 0 0.0003 0 0.0007 0 0 0 0.1657 0;
         0 0.1662 0 0.0007 0 0.0004 0 0 0 0.0827 0 0.0833;
         0 0 0.0007 0 0.1662 0 0.0829 0 0.0831 0 0.0004 0]

BRtab = [0 0.001 0 0.167 0 0.331 0 0.167 0.333 0.002 0 0;
         0.167 0 0.166 0 0.001 0.002 0.001 0 0 0.332 0.331 0;
         0.333 0.332 0 0.001 0 0.001 0 0 0 0.165 0 0.167;
         0 0.001 0.001 0.332 0.332 0 0.166 0 0.166 0 0.001 0]

const g_to_e_TotalProb = sum(TItab, dims=1)
const end_gs = collect(1:n)
const μ_signs = SVector(-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1)
const wvs = Array{Weights{Float64,Float64,Array{Float64,1}}}(undef, num_gs)

TIBRtab = zeros(Float64, num_gs, num_gs);
for g in 1:num_gs, g′ in 1:num_gs
    for e in 1:num_es
        TIBRtab[g, g′] += TItab[e, g] * BRtab[e, g′] # Probability of gstate -> estate -> gstate′
    end
end

for g in 1:num_gs
    wvs[g] = Weights(TIBRtab[g, :])
end

δs = [-13.75e9, 2.5e9];
p = @params (dFlut, Bnormlut, δs)

@inline Rsc(s, δ) =  (Γ / 2) * (s / (1 + s + 4(δ/Γ)^2))
@inline detuning(Δ0, UB, vz) = 2π * (Δ0 + UB / h + vz / λ)
@inline Prob(g, s, δ, dt) = Rsc(s, δ) * dt * g_to_e_TotalProb[g]

@inline function scatterphoton(g, s, δ, dt)::Tuple{Bool, Int64}
    if rand() > Prob(g, s, δ, dt)
        return false, g
    else
        g′ = sample(end_gs, wvs[g])
        return true, g′
    end
end

@inline function f(r, v, a, g, dt, p)
    x, y, z = r

    # Compute "lookup" indices corresponding to the current particle position
    # These are used to index into any arrays utilized -- here `Bnormlut` and `dFlut`
    ix = round(Int, 1e4 * x + 25) + 1
    iy = round(Int, 1e4 * y + 25) + 1
    iz = mod(round(Int, z / 20e-5), 201) + 1

    P0 = 2P / (Iₛ * π * w^2)
    s = P0 * exp(-(x^2 + y^2) / (2w^2))
    UB = μ * μ_signs[g] * p.Bnormlut[ix, iy, iz]
    d1 = detuning(p.δs[1], UB, v[3])
    scat1, g′  = scatterphoton(g, s, d1, dt)
    d2 = detuning(p.δs[2], UB, v[3])
    scat2, g′′ = scatterphoton(g′, s, d2, dt)
    a′ = μ_signs[g] * p.dFlut[ix, iy, iz]

    return (a′, g′′)
end

η = 35; ξ = 100;

@btime propagate!(rs_, vs_, as_, gs_, $f, $is_dead, $η, $ξ, $dt, $max_steps, $p) setup=(rs_=deepcopy($rs); vs_=deepcopy($vs); as_=deepcopy($as); gs_=deepcopy($gs)) evals=1;

rs_=deepcopy(rs); vs_=deepcopy(vs); as_=deepcopy(as); gs_=deepcopy(gs);
rs_traj, vs_traj = propagate!(rs_, vs_, as_, gs_, f, is_dead, η, ξ, dt, max_steps, p);

@inline function f_nothing(r, v, a, g, dt, p)
    return (a, g)
end

rs_=deepcopy(rs); vs_=deepcopy(vs); as_=deepcopy(as); gs_=deepcopy(gs);
rs_traj, vs_traj = propagate!(rs_, vs_, as_, gs_, f_nothing, is_dead, η, ξ, dt, max_steps, p);

# plot()
# anim = @animate for i in 1:100
#     r = rs_traj[i]
#     scatter!(
#         [r_[3] for r_ in r],
#         [r_[1] for r_ in r],
#         [r_[2] for r_ in r],
#         xlims=(0, 1.2),
#         ylims=(-0.005, 0.005),
#         zlims=(-0.005, 0.005),
#         alpha=0.30,
#         legend=false,
#         markersize=2
#         )
# end
# gif(anim, "some_file_name.gif", fps=10)
#
# rz_final = [r[end][3] for r in rs_traj]
# vz_final = [v[end][3] for v in vs_traj]
# idxs = rz_final .> 1.14
# histogram(vz_final[idxs])
