using DrWatson
@quickactivate "project"

using DifferentialEquations
using Plots
using DataFrames
using JLD2

script_name = isempty(PROGRAM_FILE) ? "interactive" : splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

n  = 5
s  = 20
fi = 3/4 * pi

p = (n = n, s = s, fi = fi)

function cutter_ode!(dr, r, p, θ)
    dr[1] = r[1] / sqrt(p.n^2 - 1)
end

function boat_polar(t, p)
    k = tan(p.fi + pi)
    x = t
    y = k * t
    r = hypot(x, y)          # sqrt(x^2 + y^2)
    θ = atan(y, x)
    return r, θ
end

function make_boat_curve(tgrid, p)
    r = Vector{Float64}(undef, length(tgrid))
    θ = Vector{Float64}(undef, length(tgrid))
    for (i, tt) in pairs(tgrid)
        ri, θi = boat_polar(tt, p)
        r[i] = ri
        θ[i] = θi
    end
    return r, θ
end

function run_case(case_name; r0, θspan=(0.0, 2pi), nθ=10_000, tmin=1e-9, tmax=8.0, nt=1_000, p)

    θgrid = collect(LinRange(θspan[1], θspan[2], nθ))
    prob = ODEProblem(cutter_ode!, [r0], θspan, p)
    sol  = solve(prob, Tsit5(), saveat=θgrid)

    df_cutter = DataFrame(θ = sol.t, r = first.(sol.u))

    tgrid = collect(LinRange(tmin, tmax, nt))
    r_boat, θ_boat = make_boat_curve(tgrid, p)
    df_boat = DataFrame(t = tgrid, θ = θ_boat, r = r_boat)

    plt = plot(sol, proj=:polar, label="катер", xlabel="θ", ylabel="r",
               title="Полярные траектории — $case_name", lw=2, legend=:topleft)
    plot!(plt, θ_boat, r_boat, proj=:polar, label="лодка", lw=2)

    savefig(plt, plotsdir(script_name, "polar_$case_name.png"))
    @save datadir(script_name, "data_$case_name.jld2") df_cutter df_boat p r0 θspan nθ tmin tmax nt

    return (plt=plt, df_cutter=df_cutter, df_boat=df_boat)
end

case1_r0 = p.s / (p.n + 1)
res1 = run_case("r0=s_div_(n+1)"; r0=case1_r0, tmax=8.0, p=p)

println("Кейс 1 — первые 5 строк (катер):")
println(first(res1.df_cutter, 5))
println("\nКейс 1 — первые 5 строк (лодка):")
println(first(res1.df_boat, 5))

case2_r0 = p.s / (p.n - 1)
res2 = run_case("r0=s_div_(n-1)"; r0=case2_r0, tmax=15.0, p=p)

println("\nКейс 2 — первые 5 строк (катер):")
println(first(res2.df_cutter, 5))
println("\nКейс 2 — первые 5 строк (лодка):")
println(first(res2.df_boat, 5))

res2.plt
