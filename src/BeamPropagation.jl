module BeamPropagation

using Distributed, StaticArrays, Unitful, LoopVectorization, StatsBase

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
    @avx for i in eachindex(rs, vs, as)
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

function update!(f, idxs, rs, vs, as, states, dt, p)
    @inbounds for i in eachindex(rs, vs, as)
        s′, v′, a′ = f(idxs[i], rs[i], vs[i], as[i], states[i], dt, p)
        states[i] = s′
        vs[i] = v′
        as[i] = a′
    end
    return nothing
end

function setup(idxs, r, v, a, states, p)
    dead_copy   = zeros(Bool, length(idxs))
    rs_copy     = deepcopy(rs[idxs])
    vs_copy     = deepcopy(vs[idxs])
    as_copy     = deepcopy(as[idxs])
    states_copy = deepcopy(states[idxs])
    p_copy      = deepcopy(p)
    return rs_copy, vs_copy, as_copy, states_copy, dead_copy, p_copy, idxs
end

function initialize_dists(n, r, v, a)
    rx = rand(r[1], n); ry = rand(r[2], n); rz = rand(r[3], n)
    vx = rand(v[1], n); vy = rand(v[2], n); vz = rand(v[3], n)
    ax = rand(a[1], n); ay = rand(a[2], n); az = rand(a[3], n)

    rs = MVector.(rx, ry, rz)
    vs = MVector.(vx, vy, vz)
    as = MVector.(ax, ay, az)

    states = ones(Int64, n)
    dead = zeros(Bool, n)
    idxs = collect(1:n)

    return rs, vs, as, states, dead, idxs
end
export initialize_dists

function propagate_parallel!(n, r, v, a, f, is_dead, delete_every, dt, max_steps, p)

    chunk_size = round(Int64, n / nworkers())
    @sync @distributed for _ in 1:nworkers()
        rs, vs, as, states, dead, idxs = initialize_dists(chunk_size, r, v, a)
        # propagate_nosave(rs, vs, as, states, dead, idxs, f, is_dead, delete_every, dt, max_steps, p)
    end
    return nothing
end
export propagate_parallel!

function propagate!(n, r, v, a, f, is_dead, delete_every, dt, max_steps, p)
    chunk_size = round(Int64, n / Threads.nthreads())
    for _ in 1:Threads.nthreads()
        rs, vs, as, states, dead, idxs = initialize_dists(chunk_size, r, v, a)
        propagate_nosave(rs, vs, as, states, dead, idxs, f, is_dead, delete_every, dt, max_steps, p)
    end
    return nothing
end
export propagate!

function propagate_nosave(rs, vs, as, states, dead, idxs, f, is_dead, delete_every, dt, max_steps, p)

    step = 0
    while (step <= max_steps)

        if step % delete_every == 0
            update_dead!(rs, dead, is_dead)
            deleteat!(rs, dead)
            deleteat!(vs, dead)
            deleteat!(as, dead)
            deleteat!(states, dead)
            deleteat!(idxs, dead)
            deleteat!(dead, dead)
        end

        update!(f, idxs, rs, vs, as, states, dt, p)
        dtstep!(rs, vs, as, dt)

        step += 1

    end
    return
end
export propagate_nosave

"""
    propagate!(rs, vs, as, gs, f, is_dead, η, ξ, dt, max_steps, p)
"""
function propagate(rs, vs, as, states, f, is_dead, save_every, delete_every, dt, max_steps, p)

    step = 0
    while (step <= max_steps)

        if step % delete_every == 0

            update_dead!(rs, dead, is_dead)

            # Positions and velocities for "dead" particles are saved
            if save_every != -1
                dead_rs = view(rs, dead)
                dead_vs = view(vs, dead)
                dead_gs = view(gs, dead)
                dead_idxs = idxs[dead]
                for i in eachindex(dead_rs, dead_vs, dead_idxs)
                    push!(rs_[dead_idxs[i]], copy(dead_rs[i]))
                    push!(vs_[dead_idxs[i]], copy(dead_vs[i]))
                    push!(gs_[dead_idxs[i]], copy(dead_gs[i]))
                end
            end

            deleteat!(rs, dead)
            deleteat!(vs, dead)
            deleteat!(as, dead)
            deleteat!(gs, dead)
            deleteat!(idxs, dead)
            deleteat!(dead, dead)
        end

        # Save positions and velocities
        if (step % save_every == 0) | (step == max_steps)
            for i in eachindex(rs, idxs)
                push!(rs_[idxs[i]], copy(rs[i]))
                push!(vs_[idxs[i]], copy(vs[i]))
                push!(gs_[idxs[i]], copy(gs[i]))
            end
        end

        update!(f, idxs, rs, vs, as, gs, dt, p)
        dtstep!(rs, vs, as, dt)

        step += 1

    end
    return rs_, vs_, gs_
end
export propagate!

end
