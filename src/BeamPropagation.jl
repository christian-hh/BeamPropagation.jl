module BeamPropagation

using StaticArrays, Unitful

macro params(fields_tuple)
    fields = fields_tuple.args
    esc(
        quote
            NamedTuple{Tuple($fields)}(($fields_tuple))
        end)
end
export @params

macro with_unit(arg1, arg2)
    arg2 = @eval @u_str $arg2
    return convert(Float64, upreferred(eval(arg1) .* arg2).val)
end
export @with_unit

function dtstep!(rs, vs, as, dt)
    @inbounds @simd for i in eachindex(rs, vs, as)
        rs[i] += vs[i] * dt + as[i] * (0.5 * dt^2)
        vs[i] += as[i] * dt
    end
    return nothing
end

function update_dead!(rs, dead, is_dead)
    @inbounds @simd for i in eachindex(rs)
        dead[i] = is_dead(rs[i])
    end
    return nothing
end

function update!(f, rs, vs, as, gs, dt, p)
    @inbounds @simd for i in eachindex(rs, vs, as)
        a′, g′ = f(rs[i], vs[i], as[i], gs[i], dt, p)
        as[i] = a′
        gs[i] = g′
    end
    return nothing
end

"""
    propagate!(rs, vs, as, gs, f, is_dead, η, dt, max_steps, p)
"""
function propagate!(rs, vs, as, gs, f, is_dead, η, ξ, dt, max_steps, p)
    n = length(rs)

    idxs = collect(1:n)
    dead = zeros(Bool, n)
    rs_ = [Array{SVector{3, Float64}, 1}(undef, 0) for _ in idxs]
    vs_ = [Array{SVector{3, Float64}, 1}(undef, 0) for _ in idxs]

    step = 0
    save_step = 1
    while (step < max_steps)

        if step % η == 0
            update_dead!(rs, dead, is_dead)

            # Positions and velocities for "dead" particles are saved
            dead_rs = view(rs, dead)
            dead_vs = view(vs, dead)
            dead_idxs = idxs[dead]
            @inbounds @simd for i in eachindex(dead_rs, dead_vs, dead_idxs)
                push!(rs_[dead_idxs[i]], copy(dead_rs[i]))
                push!(vs_[dead_idxs[i]], copy(dead_vs[i]))
            end

            deleteat!(rs, dead)
            deleteat!(vs, dead)
            deleteat!(as, dead)
            deleteat!(gs, dead)
            deleteat!(idxs, dead)
            deleteat!(dead, dead)
        end

        # Save positions and velocities
        if (step % ξ == 0) | (step == max_steps - 1)
            @inbounds @simd for i in eachindex(rs, idxs)
                push!(rs_[idxs[i]], copy(rs[i]))
                push!(vs_[idxs[i]], copy(vs[i]))
            end
            save_step += 1
        end

        update!(f, rs, vs, as, gs, dt, p)
        dtstep!(rs, vs, as, dt)
        step += 1

    end
    return rs_, vs_
end
export propagate!

end
