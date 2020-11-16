using Revise
using
    BeamPropagation, StaticArrays, DelimitedFiles, BenchmarkTools, Plots, StructArrays, StatsBase

# Define constants
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

# Define parameters
dt          = 5e-7
n           = 100000
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

    #
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

@inline function scatterphoton(g, s, δ, dt, rnd)::Tuple{Bool, Int64}
    if rnd > Prob(g, s, δ, dt)
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

    s = P0 * exp(-(x^2 + y^2) / (2(w/100)^2))
    UB = μ * μ_signs[g] * p.Bnormlut[ix, iy, iz]
    d1 = detuning(p.δs[1], UB, v[3])
    scat1, g′  = scatterphoton(g, s, d1, dt, rand())
    d2 = detuning(p.δs[2], UB, v[3])
    scat2, g′′ = scatterphoton(g′, s, d2, dt, rand())
    a′ = μ_signs[g] * p.dFlut[ix, iy, iz]

    return (a′, g′′)
end

η = 35; @btime propagate!(rs_, vs_, as_, gs_, $f, $is_dead, $η, $dt, $max_steps, $p) setup=(rs_=deepcopy($rs); vs_=deepcopy($vs); as_=deepcopy($as); gs_=deepcopy($gs)) evals=1;

# η = 35; rs_=deepcopy(rs); vs_=deepcopy(vs); as_=deepcopy(as); gs_=deepcopy(gs)); rz_final, vz_final, az_final = propagate!(rs_, vs_, as_, gs_, dead_, dt, max_steps, η, is_dead, f, p);

survived = rz_final .> (0.95 * 1.2)
histogram(vz_final[survived], bins=0:5:75)
