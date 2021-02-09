using Distributed
rmprocs(workers())
addprocs(2)

@everywhere using
    Revise, BeamPropagation, Distributions, StatsBase, StaticArrays, PhysicalConstants.CODATA2018, Plots

const h = PlanckConstant.val
const ħ = h / 2π
const λ = @with_unit 626 "nm"
const k = 2π / λ
const m = @with_unit 57 "u"

@everywhere n = Int(1e5)

vz_μ = @with_unit 110 "m/s"
vz_σ = @with_unit 25 "m/s"
vxy_μ = @with_unit 0 "m/s"
vxy_σ = @with_unit 20 "m/s"
exit_radius = @with_unit 4 "mm"

r = (Normal(0, exit_radius/2), Normal(0, exit_radius/2), Normal(0, 0))
v = (Normal(vxy_μ, vxy_σ), Normal(vxy_μ, vxy_σ), Normal(vz_μ, vz_σ))
a = (Normal(0, 0), Normal(0, 0), Normal(0, 0))

# vz = rand(Normal(vz_μ, vz_σ), n)
# vx = rand(Normal(vxy_μ, vxy_σ), n)
# vy = rand(Normal(vxy_μ, vxy_σ), n)
#
# rx = rand(Normal(0, exit_radius/2), n)
# ry = rand(Normal(0, exit_radius/2), n)
# rz = zeros(n)
#
# rs = MVector.(rx, ry, rz)
# vs = MVector.(vx, vy, vz)
# as = MVector.(zeros(n), zeros(n), zeros(n))
#
# states = ones(Int64, n)

@everywhere VBRs = Weights([
    0.9457,     # to 000
    0.0447,     # 100
    3.9e-3,     # 0200
    2.7e-3,     # 200
    9.9e-4,     # 0220
    11.3e-4,    # 0110
    3.9e-4,     # 1200
    1.5e-4,     # 300
    1.3e-4,     # 1220
    0.7e-4,     # 110 (only N=1, assuming roughly 2/3 to 1/3 rotational branching)
    0.4e-4,     # 110, N=2
    5.7e-5,     # 220
    4.3e-5      # other states
])

@everywhere addressed = [
    true,   # to 000
    true,   # 100
    true,   # 0200
    true,   # 200
    true,   # 0220
    true,   # 0110
    true,   # 1200
    true,   # 300
    true,   # 1220
    true,   # 110
    false,   # 110, N=2
    false,  # 220
    false   # other states
]

@everywhere transverse = [
    false,  # to 000
    false,  # 100
    false,  # 0200
    false,  # 200
    false,  # 0220
    false,  # 0110
    false,  # 1200
    false,  # 300
    true,   # 1220
    true,   # 110
    false,   # 110, N=2
    false,   # 220
    false   # other states
]

@inline function transverse_on(z)
    if 0.175 < z < 0.20     # 1st transverse region 17.5 - 20 cm after cell
        return true
    elseif 0.45 < z < 0.50  # 2nd transverse region 45 - 50 cm after cell
        return true
    elseif 0.56 < z < 0.61  # 2nd transverse region 56 - 61 cm after cell
        return true
    else
        return false
    end
end

@everywhere @inline is_dead(r) = sqrt(r[1]^2 + r[2]^2) > 10e-3 || r[3] > 0.71

@everywhere @inline function f(i, r, v, a, state, dt, p)
    if addressed[state] & (~transverse[state] | transverse_on(r[3]))
        state′ = sample(1:13, p.VBRs)
        v′ = v
        v′[3] -= ħ * k / m
        p.photons[i] += 1
        p.vzs[i] = v′[3]
    else
        state′ = state
        v′ = v
        p.vzs[i] = v′[3]
    end
    return (state′, v′, a)
end

@everywhere photons = zeros(Int64, n)
@everywhere vzs     = zeros(Float64, n)
@everywhere p = @params (VBRs, addressed, photons, vzs)

scattering_rate = @with_unit 1.0 "MHz"
save_every      = 100
delete_every    = 10
dt              = 1 / scattering_rate
max_steps       = Int(3.5e4)

chunk_size = round(Int64, n / nworkers())
@sync @distributed for _ in 1:nworkers()
    rs, vs, as, states, dead, idxs = initialize_dists(chunk_size, r, v, a)
    propagate_nosave(rs, vs, as, states, dead, idxs, f, is_dead, delete_every, dt, max_steps, p)
end
return nothing

# @time propagate_parallel!(n, r, v, a, f, is_dead, delete_every, dt, max_steps, p)
# @time propagate!(n, r, v, a, f, is_dead, delete_every, dt, max_steps, p)

# @time @time propagate!(rs, vs, as, states, f, is_dead, delete_every, dt, max_steps, p)

# v_start     = [v[1][3] for v in vs]
# v_end       = p.vzs
# alive       = [r[end][3] >= 0.71 for r in rs_]
# # detectable  = [addressed[g[end]] & ~transverse[g[end]] for g in gs_]
# detectable  = [addressed[g[end]] for g in gs_]
#
# histogram(v_start, alpha=0.5, bins=50, xlim=[0, 200], ylim=[0, 150], weights=alive)
# histogram!(v_end[alive .& detectable], alpha=0.5, bins=50)

# @inline function transverse_on_TEST(z)
#     # if 0.175 < z < 0.225     # 1st transverse region 17.5 - 22.5 cm after cell
#         # return true
#     if 0.275 < z < 0.325  # 2nd transverse region 27.5 - 32.5 cm after cell
#         return true
#     else
#         return false
#     end
# end
#
# @inline is_dead_TEST(r) = sqrt(r[1]^2 + r[2]^2) > 10e-3 || r[3] > 0.50
