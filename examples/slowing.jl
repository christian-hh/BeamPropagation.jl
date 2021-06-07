using Revise, BeamPropagation, Distributions, StatsBase, StaticArrays, PhysicalConstants.CODATA2018, Plots, BenchmarkTools

const h = PlanckConstant.val
const ħ = h / 2π
const λ = @with_unit 626 "nm"
const k = 2π / λ
const m = @with_unit 57 "u"
const Δv = ħ * k / m

vz_μ = @with_unit 125 "m/s"
vz_σ = @with_unit 25 "m/s"
vxy_μ = @with_unit 0 "m/s"
vxy_σ = @with_unit 25 "m/s"
exit_radius = @with_unit 4 "mm"

const r = (Normal(0, exit_radius/2), Normal(0, exit_radius/2), Normal(0, 0))
const v = (Normal(vxy_μ, vxy_σ), Normal(vxy_μ, vxy_σ), Normal(vz_μ, vz_σ))
const a = (Normal(0, 0), Normal(0, 0), Normal(0, 0))

VBRs = Weights([
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

addressed = [
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
    false,   # 220
    false   # other states
]

transverse = [
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

@inline function random_unit3Dvector()
    θ = rand(Uniform(0, 2π))
    z = rand(Uniform(-1, 1))
    return @SVector [sqrt(1-z^2)*cos(θ), sqrt(1-z^2)*sin(θ), z]
end

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

const detect_rad  = @with_unit 0.5 "cm"
const detect_zloc = @with_unit 70 "cm"
const detect_zlen = @with_unit 0.5 "cm"
const dead_rad = @with_unit 1 "cm"
const dead_len = detect_zloc + detect_zlen

n = Int64(40e6)
scattering_rate = @with_unit 1.2 "MHz"
save_every      = 100
delete_every    = 15
dt              = 1 / scattering_rate
max_steps       = Int64(3.5e4)

@inline function simple_prop(r, v)
    dist_detect = detect_zloc - r[3]
    x_final = r[1] + v[1] * dist_detect / v[3]
    y_final = r[2] + v[2] * dist_detect / v[3]
    return sqrt(x_final^2 + y_final^2)
end
@inline discard(r, v) = (simple_prop(r, v) > dead_rad) || (r[3] > dead_len)
@inline is_detectable(r) = v_perp(r) < detect_rad && (detect_zloc + detect_zlen > r[3] > detect_zloc)

@inline function save(i, r, v, a, state, s)
    s.vzs[i] = v[3]
    s.detectable[i] = simple_prop(r, v) < detect_rad
    return nothing
end

@inline function f(i, r, v, a, state, dt, p, s)
    if is_detectable(r) && ~s.detectable[i]
        s.detectable[i] = true
    end
    if p.addressed[state] & (~p.transverse[state] | transverse_on(r[3]))
        state′ = sample(1:13, p.VBRs)
        v′ = @SVector [v[1], v[2], v[3] - Δv]
        v′ += Δv .* random_unit3Dvector()
        s.photons[i] += 1
    else
        state′ = state
        v′ = v
    end
    s.vzs[i] = v′[3]
    s.states[i] = state′
    return (state′, v′, a)
end

vzs           = zeros(Float64, n)
photons       = zeros(Int64, n)
detectable    = zeros(Bool, n)
states        = ones(Int64, n)

p = @params (VBRs, transverse, addressed)
s = @params (vzs, photons, detectable, states)

@time s₀, sₜ = propagate!(n, r, v, a, f, save, discard, delete_every, dt, max_steps, p, s)

bright = [addressed[final_state] for final_state in sₜ.states]

barhist(s₀.vzs[s₀.detectable], bins=0:25/4:225, alpha=0.5, xlim=[0,225])
barhist!(sₜ.vzs[sₜ.detectable], bins=0:25/4:225, alpha=0.5)
barhist!(sₜ.vzs[sₜ.detectable .& bright], bins=0:25/4:225, alpha=0.5)

sum(abs.(s₀.vzs[s₀.detectable]) .< 10) / n
sum(abs.(sₜ.vzs[sₜ.detectable]) .< 10) / n
sum(abs.(sₜ.vzs[sₜ.detectable .& bright]) .< 10) / n

v_class = 0 .< sₜ.vzs .< 100
histogram2d(s₀.vzs[sₜ.detectable .& bright .& v_class], sₜ.vzs[sₜ.detectable .& bright.& v_class], bins=20)
