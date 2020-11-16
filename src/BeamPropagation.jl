module BeamPropagation

macro params(fields_tuple)
    fields = fields_tuple.args
    esc(
        quote
            NamedTuple{Tuple($fields)}(($fields_tuple))
        end)
end
export @params

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
function propagate!(rs, vs, as, gs, f, is_dead, η, dt, max_steps, p)
    n = length(rs)
    idxs = collect(1:n)

    dead = [false for _ in idxs]
    rs_ = [SVector{3, Float64}[] for _ in idxs]
    vs_ = [SVector{3, Float64}[] for _ in idxs]

    step = 0
    while (step < max_steps) & !(isempty(rs))

        if step % η == 0
            update_dead!(rs, dead, is_dead)
            deleteat!(rs, dead)
            deleteat!(vs, dead)
            deleteat!(as, dead)
            deleteat!(gs, dead)
            deleteat(idxs, dead)
            deleteat!(dead, dead)
        end

        update!(f, rs, vs, as, gs, dt, p)
        dtstep!(rs, vs, as, dt)
        step += 1

        # Push results for data to be saved
        for i in eachindex(rs, vs)
            push!(rs_[idxs[i]], rs[i])
            push!(vs_[idxs[i]], rs[i])
        end

    end
    return rs_, vs_
end
export propagate!

end
