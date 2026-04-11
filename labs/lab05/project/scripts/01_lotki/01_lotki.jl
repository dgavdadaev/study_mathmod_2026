using DrWatson
@quickactivate "project"

using DifferentialEquations
using Plots
using DataFrames
using CSV
using JLD2

script_name = isempty(PROGRAM_FILE) ? "interactive" : splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

x0 = 8.0
y0 = 11.0

u0 = [x0, y0]

t0 = 0.0
tmax = 150.0
tspan = (t0, tmax)

a = 0.25
b = 0.025
c = 0.45
d = 0.045

p = (a = a, b = b, c = c, d = d)

function predator_prey!(dy, y, p, t)
    dy[1] = -p.a * y[1] + p.b * y[1] * y[2]
    dy[2] =  p.c * y[2] - p.d * y[1] * y[2]
end

function run_case(case_name; model!, u0, tspan, p, nt=1000)

    tgrid = collect(LinRange(tspan[1], tspan[2], nt))

    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tgrid)

    df = DataFrame(
        t = sol.t,
        x = first.(sol.u),
        y = last.(sol.u)
    )

    plt_time = plot(
        sol,
        xlabel = "t",
        ylabel = "Численность",
        label = ["x(t) — жертвы" "y(t) — хищники"],
        lw = 2,
        title = "Динамика популяций — $case_name",
        legend = :topright
    )

    plt_phase = plot(
        sol,
        idxs = (1, 2),
        xlabel = "x",
        ylabel = "y",
        lw = 2,
        label = "Фазовая траектория",
        title = "Фазовый портрет — $case_name",
        legend = :topright
    )

    savefig(plt_time, plotsdir(script_name, "$(case_name)_time.png"))
    savefig(plt_phase, plotsdir(script_name, "$(case_name)_phase.png"))

    CSV.write(datadir(script_name, "table_$(case_name).csv"), df)

    @save datadir(script_name, "data_$(case_name).jld2") df p u0 tspan nt

    return (sol = sol, df = df, plt_time = plt_time, plt_phase = plt_phase)
end

res = run_case(
    "predator_prey";
    model! = predator_prey!,
    u0 = u0,
    tspan = tspan,
    p = p,
    nt = 1000
)

println("Первые 5 строк таблицы результатов:")
println(first(res.df, 5))

res.plt_time
