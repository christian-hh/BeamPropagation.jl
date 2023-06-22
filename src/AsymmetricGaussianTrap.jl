export AsymmetricGaussianTrap
export U
export F
export make_distributions

using PhysicalConstants.CODATA2018
using StaticArrays
using Distributions

const kB = BoltzmannConstant.val

Base.@kwdef struct AsymmetricGaussianTrap
    λ::Float64
    w0x::Float64
    w0y::Float64
    U_max::Float64
    m::Float64    
    zRx::Float64 = π * w0x^2 / λ
    zRy::Float64 = π * w0y^2 / λ
    ω::SVector{3, Float64} = SVector( sqrt(4*U_max/(m*w0x^2)), sqrt(4*U_max/(m*w0y^2)), sqrt(2*U_max/(m*zRx*zRy)) )
end

wx(z, trap::AsymmetricGaussianTrap) = trap.w0x * sqrt(1 + z^2 / trap.zRx^2)
wy(z, trap::AsymmetricGaussianTrap) = trap.w0y * sqrt(1 + z^2 / trap.zRy^2)

function U(r, trap::AsymmetricGaussianTrap)
    w0x, w0y, zRx, zRy = trap.w0x, trap.w0y, trap.zRx, trap.zRy
    x, y, z = r
    
    wx = waist(z, w0x, zRx, 0)
    wy = waist(z, w0y, zRy, 0)
    return trap.U_max * (1 - ((w0x * w0y) / (wx * wy)) * exp(-2 * (x^2 / wx^2 + y^2 / wy^2)))
end

function F(x, y, z, trap::AsymmetricGaussianTrap)
    λ, w0x, w0y, zRx, zRy = trap.λ, trap.w0x, trap.w0y, trap.zRx, trap.zRy
    
    wx_ = wx(z, trap)
    wy_ = wy(z, trap)
    exp_term = exp(-2(x^2 / wx_^2 + y^2 / wy_^2))
    denom_term_x = wx_^2 * sqrt(1 + (z/zRx)^2) * sqrt(1 + (z/zRy)^2)
    denom_term_y = wy_^2 * sqrt(1 + (z/zRx)^2) * sqrt(1 + (z/zRy)^2)
    
    F_x = 4x / denom_term_x
    F_y = 4y / denom_term_y
    F_z = z * (w0x / zRx)^2 / denom_term_x + z * (w0y / zRy)^2 / denom_term_y -
        (4 * x^2 * z) / (denom_term_x * zRx^2 * (1 + (z/zRx)^2)) - (4 * y^2 * z) / (denom_term_y * zRy^2 * (1 + (z/zRy)^2))

    return -exp_term .* SVector(F_x, F_y, F_z)
end

F(r, trap) = F(r[1], r[2], r[3], trap)

function make_distributions(T, trap::AsymmetricGaussianTrap)
    σ = sqrt(kB * T / trap.m)
    σ_rx = σ / trap.ω[1]
    σ_ry = σ / trap.ω[2]
    σ_rz = σ / trap.ω[3]
    σ_vx = σ_vy = σ_vz = σ
    r = (Normal(0, σ_rx), Normal(0, σ_ry), Normal(0, σ_rz))
    v = (Normal(0, σ_vx), Normal(0, σ_vy), Normal(0, σ_vz))
    a = (Normal(0, 0), Normal(0, 0), Normal(0, 0))
    return r, v, a
end