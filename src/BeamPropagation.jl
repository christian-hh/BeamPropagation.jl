module BeamPropagation

using StaticArrays, Unitful, LoopVectorization, StatsBase

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

function update_dead!(rs, vs, dead, is_dead)
    @inbounds for i in 1:size(rs,1)
        dead[i] = is_dead(rs[i], vs[i])
    end
    return nothing
end

function update!(f, idxs, rs, vs, as, states, dt, p, s)
    @inbounds for i in 1:size(rs,1)
        s′, v′, a′ = f(idxs[i], rs[i], vs[i], as[i], states[i], dt, p, s)
        states[i] = s′
        vs[i] = v′
        as[i] = a′
    end
    return nothing
end

function save!(save, idxs, rs, vs, as, states, s)
    @inbounds for i in 1:size(rs,1)
        save(idxs[i], rs[i], vs[i], as[i], states[i], s)
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

    rs = SVector.(rx, ry, rz)
    vs = SVector.(vx, vy, vz)
    as = SVector.(ax, ay, az)

    states = ones(Int64, n)
    dead = zeros(Bool, n)
    idxs = collect(1:n)

    return rs, vs, as, states, dead, idxs
end
export initialize_dists

function copy_save_data(s, actual_chunk_size)
    s_copy = NamedTuple()
    for key in keys(s)
        array_to_save = deepcopy(s[key][1:actual_chunk_size])
        s_copy = (; s_copy..., key => array_to_save)
    end
    return s_copy
end

function write_data(s, s_copy, chunk_idxs)
    for key in keys(s)
        s[key][chunk_idxs] .= s_copy[key]
    end
    return nothing
end

function delete_data(rs, vs, as, states, idxs, dead)
    deleteat!(rs, dead)
    deleteat!(vs, dead)
    deleteat!(as, dead)
    deleteat!(states, dead)
    deleteat!(idxs, dead)
    deleteat!(dead, dead)
    return nothing
end

function propagate!(n, r, v, a, f, save, discard, delete_every, dt, max_steps, p, s)

    s_initial = deepcopy(s)
    s_final = deepcopy(s)

    chunk_size = round(Int64, n / Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()

        start_idx   = (i-1)*chunk_size+1
        end_idx     = min(i*chunk_size, n)
        chunk_idxs  = start_idx:end_idx
        actual_chunk_size = length(chunk_idxs)

        rs, vs, as, states, dead, idxs = initialize_dists(actual_chunk_size, r, v, a)

        s_initial_copy = copy_save_data(s_initial, actual_chunk_size)
        save!(save, idxs, rs, vs, as, states, s_initial_copy)
        write_data(s_initial, s_initial_copy, chunk_idxs)

        # Do one round of discards before propagation starts
        update_dead!(rs, vs, dead, discard)
        delete_data(rs, vs, as, states, idxs, dead)

        # Copy parameters and data arrays to be saved to avoid thread race conditions
        s_copy = copy_save_data(s_final, actual_chunk_size)
        p_copy = deepcopy(p)

        propagate_nosave(rs, vs, as, states, dead, idxs, f, discard, delete_every, dt, max_steps, p_copy, s_copy)

        write_data(s_final, s_copy, chunk_idxs)
    end

    return s_initial, s_final
end
export propagate!

function propagate_nosave(rs, vs, as, states, dead, idxs, f, discard, delete_every, dt, max_steps, p, s)
    step = 0
    while (step <= max_steps)
        if step % delete_every == 0
            update_dead!(rs, vs, dead, discard)
            delete_data(rs, vs, as, states, idxs, dead)
        end
        update!(f, idxs, rs, vs, as, states, dt, p, s)
        dtstep!(rs, vs, as, dt)
        step += 1
    end
    return nothing
end
export propagate_nosave

# function propagate_insteps(rs, vs, as, states, dead, idxs, f, is_dead, discarded, delete_every, dt, max_steps, p)
#
#     for

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
