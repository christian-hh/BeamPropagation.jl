module BeamPropagation

include("AsymmetricGaussianTrap.jl")

using
    StaticArrays,
    Unitful,
    LoopVectorization,
    StatsBase,
    StructArrays,
    MutableNamedTuples

macro params(fields_tuple)
    fields = fields_tuple.args
    esc(
        quote
            MutableNamedTuples.MutableNamedTuple{Tuple($fields)}(($fields_tuple))
        end)
end
export @params

function dtstep_euler!(particles, f, abstol, p, dt_min, dt_max, time)

    @inbounds for i in 1:size(particles, 1)
        particles.a[i] = f(particles.idx[i], particles[i].r, particles[i].v, p, time)
    end

    for i in 1:size(particles, 1)
        r = particles.r[i]
        v = particles.v[i]
        a = particles.a[i]
        dt = particles.dt[i]

        # Perform an Euler method step (first order)
        particles.v[i] = v + a * dt
        particles.r[i] = r + v * dt
    end
    return nothing
end

function dtstep_eulerrich!(particles, f, abstol, p, dt_min, dt_max, time)

    @inbounds for i in 1:size(particles, 1)
        particles.a[i] = f(particles.idx[i], particles.r[i], particles.v[i], p, time)
    end

    for i in 1:size(particles, 1)
        r = particles.r[i]
        v = particles.v[i]
        a = particles.a[i]
        dt = particles.dt[i]

        # Perform an Euler method step (first order)
        particles.v1[i] = v + a * dt
        particles.r1[i] = r + v * dt
    end

    @inbounds for i in 1:size(particles, 1)
        particles.a[i] = f(particles.idx[i], particles.r1[i], particles[i].v1, p, time)
    end

    for i in 1:size(particles, 1)
        r = particles.r[i]
        v = particles.v[i]
        r1 = particles.r1[i]
        v1 = particles.v1[i]
        a = particles.a[i]
        dt = particles.dt[i]

        # Perform a Heun method step (second order)
        v2 = 0.5v + 0.5 * (v1 + a * dt)
        particles.v2[i] = v2
        particles.r2[i] = 0.5r + 0.5 * (r1 + v2 * dt)
    end

    @inbounds for i in 1:size(particles, 1)
        r1 = particles.r1[i]
        r2 = particles.r2[i]
        v2 = particles.v2[i]
        dt = particles.dt[i]

        if particles.use_adaptive[i]
            error = sqrt(sum((r2 - r1)).^2)
            particles.error[i] = error

            if !iszero(error)
                dt_corr = dt * 0.9 * sqrt(abstol / (2 * error))
            else
                dt_corr = dt_max
            end
            new_dt = min(max(dt_corr, dt_min), dt_max)
            particles.dt[i] = new_dt

            if error < abstol
                particles.r[i] = r2
                particles.v[i] = v2
            end
        else
            particles.r[i] = r2
            particles.v[i] = v2
        end

    end

    return nothing
end
export dtstep_eulerrich!

mutable struct Particle
    r::SVector{3, Float64}
    v::SVector{3, Float64}
    a::SVector{3, Float64}
    dt::Float64
    use_adaptive::Bool
    error::Float64
    idx::Int64
    r1::SVector{3, Float64}
    v1::SVector{3, Float64}
    r2::SVector{3, Float64}
    v2::SVector{3, Float64}
    dead::Bool
end
export Particle

function save!(save, idxs, rs, vs, as, states, s)
    @inbounds for i in 1:size(rs,1)
        save(idxs[i], rs[i], vs[i], as[i], states[i], s)
    end
    return nothing
end

function initialize_dists_particles!(r, v, a, i_start, particles, dt, use_adaptive)
    @inbounds for i in 1:size(particles, 1)
        particles.r[i] = particles.r1[i] = SVector.(rand(r[1]), rand(r[2]), rand(r[3]))
        particles.v[i] = particles.v1[i] = SVector.(rand(v[1]), rand(v[2]), rand(v[3]))
        particles.a[i] = SVector.(rand(a[1]), rand(a[2]), rand(a[3]))
        particles.dead[i] = false
        particles.idx[i] = i_start + i - 1
        particles.dt[i] = dt
        particles.use_adaptive[i] = use_adaptive
        particles.error[i] = 0.0
    end
    return nothing
end
export initialize_dists_particles!

function discard_particles!(particles, discard)
    @inbounds for i in 1:size(particles, 1)
        particles.dead[i] = discard(particles.r[i], particles.v[i])
    end
    StructArrays.foreachfield(x -> deleteat!(x, particles.dead), particles)
    return nothing
end

function propagate_particles_single!(
        r, v, a, alg, 
        particles, f::F1, save::F2, discard::F3, save_every, delete_every, max_steps, 
        update, p, s, dt, use_adaptive, dt_min, dt_max, abstol) where {F1, F2, F3}

    initialize_dists_particles!(r, v, a, 1, particles, dt, use_adaptive)
    print(particles[1].r)
    
    step = 0
    
    while (step <= max_steps)

        update(particles, p, s, dt)

        if step % save_every == 0
            save(particles, p, s)
        end

        if step % delete_every == 0
            discard_particles!(particles, discard)
        end

        if alg == "euler"
            dtstep_euler!(particles, f, abstol, p, dt_min, dt_max)
        elseif alg == "rkf12"
            dtstep_eulerrich!(particles, f, abstol, p, dt_min, dt_max)
        end

        step += 1
    end

    return nothing
end
export propagate_particles_single!

function propagate_particles!(r, v, a, alg, particles, f::F1, save::F2, discard::F3, save_every, delete_every, max_steps, update, p, s, dt, use_adaptive, dt_min, dt_max, abstol, randomize) where {F1, F2, F3}

    n = length(particles)
    n_threads = Threads.nthreads()
    chunk_size = ceil(Int64, n / n_threads)
    
    Threads.@threads for i in 1:n_threads

        p_ = deepcopy(p)
        
        start_idx   = (i-1)*chunk_size+1
        end_idx     = min(i*chunk_size, n)
        chunk_idxs  = start_idx:end_idx
        actual_chunk_size = length(chunk_idxs)
        
        particles_chunk = particles[chunk_idxs]
        if randomize
            initialize_dists_particles!(r, v, a, start_idx, particles_chunk, dt, use_adaptive)
        end

        idx = 1
        save_idx = 1
        time = 0.0

        for step in 0:(max_steps - 1)

            update(particles_chunk, p_, s, dt, time, idx)

            if step % save_every == 0
                save(particles_chunk, p_, s)
            end

            if step % delete_every == 0
                discard_particles!(particles_chunk, discard)
            end

            if alg == "euler"
                dtstep_euler!(particles_chunk, f, abstol, p_, dt_min, dt_max, time)
            elseif alg == "rkf12"
                dtstep_eulerrich!(particles_chunk, f, abstol, p_, dt_min, dt_max, time)
            end

            idx += 1
            time += dt

            if iszero(length(particles_chunk))
                break
            end
        end
    end

    return nothing
end
export propagate_particles!

end
