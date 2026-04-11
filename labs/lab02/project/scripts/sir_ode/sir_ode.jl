using DrWatson
@quickactivate "project"

using DifferentialEquations
using SimpleDiffEq
using Tables
using DataFrames
using StatsPlots
using LaTeXStrings
using Plots
using BenchmarkTools

script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

function sir_ode!(du, u, p, t)
    (S, I, R) = u
    (β, c, γ) = p
    N = S + I + R

    @inbounds begin
        du[1] = -β * c * I / N * S   # dS/dt
        du[2] = β * c * I / N * S - γ * I  # dI/dt
        du[3] = γ * I                 # dR/dt
    end
    nothing
end

δt = 0.1
tmax = 40.0
tspan = (0.0, tmax)

u0 = [990.0, 10.0, 0.0]  # S, I, R

p = [0.05, 10.0, 0.25]    # β, c, γ

R0 = (p[2] * p[1]) / p[3]

println("="^60)
println("МОДЕЛЬ SIR: ПАРАМЕТРЫ")
println("="^60)
println("β (вероятность заражения) = ", p[1])
println("c (среднее число контактов) = ", p[2])
println("γ (скорость выздоровления) = ", p[3])
println("R0 = c * β / γ = ", round(R0, digits=3))
println("Средняя продолжительность болезни = ", round(1/p[3], digits=2), " дней")
println("Начальные условия: S0 = ", u0[1], " I0 = ", u0[2], " R0 = ", u0[3])
println("="^60)

prob_ode = ODEProblem(sir_ode!, u0, tspan, p)
sol_ode = solve(prob_ode, dt = δt)

df_ode = DataFrame(Tables.table(sol_ode'))
rename!(df_ode, ["S", "I", "R"])
df_ode[!, :t] = sol_ode.t
df_ode[!, :N] = df_ode.S + df_ode.I + df_ode.R

plt1 = @df df_ode plot(:t, [:S :I :R],
    label=[L"S(t)" L"I(t)" L"R(t)"],
    xlabel="Время, дни",
    ylabel="Количество людей",
    title="Модель SIR: Динамика эпидемии",
    linewidth=2,
    legend=:right,
    grid=true,
    size=(800, 500))

annotate!(plt1, maximum(df_ode.t) * 0.7, maximum(df_ode.N) * 0.8,
    text("Параметры:\nβ = $(p[1])\nc = $(p[2])\nγ = $(p[3])\nR0 = $(round(R0, digits=2))", 8, :left))

plt2 = @df df_ode plot(:t, :I,
    label=L"I(t)",
    xlabel="Время, дни",
    ylabel="Количество инфицированных",
    title="Динамика числа зараженных",
    color=:red,
    linewidth=2,
    fill=(0, 0.3, :red),
    grid=true,
    size=(800, 400))

peak_idx = argmax(df_ode.I)
peak_time = df_ode.t[peak_idx]
peak_value = df_ode.I[peak_idx]
vline!(plt2, [peak_time], color=:black, linestyle=:dash, label=false, linewidth=1)
annotate!(plt2, peak_time, peak_value * 1.05,
    text("Пик: $(round(peak_value, digits=1)) на $(round(peak_time, digits=1)) день", 8, :top))

plt3 = @df df_ode plot(:t, :I,
    label=L"I(t)",
    xlabel="Время, дни",
    ylabel="Количество инфицированных (лог. масштаб)",
    title="Экспоненциальный рост (лог. шкала)",
    yscale=:log10,
    color=:red,
    linewidth=2,
    grid=true,
    size=(800, 400))

plt4 = @df df_ode plot(:t, [df_ode.S ./ df_ode.N .* 100 df_ode.I ./ df_ode.N .* 100 df_ode.R ./ df_ode.N .* 100],
    label=[L"S(t)/N" L"I(t)/N" L"R(t)/N"],
    xlabel="Время, дни",
    ylabel="Доля популяции, %",
    title="Динамика эпидемии (в процентах)",
    linewidth=2,
    legend=:right,
    grid=true,
    size=(800, 500))

if R0 > 1
    herd_immunity_threshold = (1 - 1/R0) * 100
    hline!(plt4, [herd_immunity_threshold], color=:purple, linestyle=:dash,
        label="Порог коллективного иммунитета ($(round(herd_immunity_threshold, digits=1))%)", linewidth=1.5)
end

plt5 = plot(df_ode.S, df_ode.I,
    label="Фазовая траектория",
    xlabel=L"S(t)",
    ylabel=L"I(t)",
    title="Фазовый портрет SIR модели",
    color=:blue,
    linewidth=2,
    grid=true,
    size=(800, 500),
    legend=:topright)

for i in 1:50:length(df_ode.S)-1
    plot!(plt5, [df_ode.S[i], df_ode.S[i+1]], [df_ode.I[i], df_ode.I[i+1]],
        arrow=:closed, color=:blue, alpha=0.5, label=false)
end

df_ode[!, :Re] = R0 .* df_ode.S ./ df_ode.N
plt6 = plot(df_ode.t, df_ode.Re,
    label=L"R_e(t)",
    xlabel="Время, дни",
    ylabel="Эффективное репродуктивное число",
    title="Динамика R_e",
    color=:orange,
    linewidth=2,
    grid=true,
    size=(800, 400),
    legend=:topright)

hline!(plt6, [1], color=:black, linestyle=:dash, label="R_e = 1", linewidth=1.5)

plt7 = plot(plt1, plt2, plt3, plt4, plt5, plt6,
    layout=(3, 2),
    size=(1400, 1200),
    titlefontsize=10)

savefig(plt1, plotsdir(script_name, "sir_main.png"))
savefig(plt2, plotsdir(script_name, "sir_infected.png"))
savefig(plt3, plotsdir(script_name, "sir_log_scale.png"))
savefig(plt4, plotsdir(script_name, "sir_percentages.png"))
savefig(plt5, plotsdir(script_name, "sir_phase_portrait.png"))
savefig(plt6, plotsdir(script_name, "sir_effective_R.png"))
savefig(plt7, plotsdir(script_name, "sir_panel.png"))

println("\nГрафики сохранены в: ", plotsdir(script_name))

println("\nБенчмарк решения:")
@benchmark solve(prob_ode, dt = $δt)

println("\n" * "="^60)
println("АНАЛИЗ РЕЗУЛЬТАТОВ")
println("="^60)

println("\nСтатистика эпидемии:")
println("• Общая численность популяции: N = ", round(df_ode.N[1], digits=1))
println("• Пиковое число зараженных: I_max = ", round(peak_value, digits=1))
println("• Время достижения пика: t_peak = ", round(peak_time, digits=1), " дней")
println("• Итоговое число переболевших: R(∞) = ", round(df_ode.R[end], digits=1))
println("• Доля переболевших: ", round(df_ode.R[end]/df_ode.N[1]*100, digits=1), "%")

if R0 > 1
    println("\nТеоретический анализ:")
    println("• Порог коллективного иммунитета: ", round((1 - 1/R0)*100, digits=1), "%")
    println("• Теоретический пик при S/N = 1/R0 = ", round(1/R0, digits=3))
    println("• Фактический минимум S/N = ", round(minimum(df_ode.S ./ df_ode.N), digits=3))
end

println("\n" * "="^60)
println("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО")
println("="^60)
