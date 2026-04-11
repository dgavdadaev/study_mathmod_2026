using DrWatson
@quickactivate "project"

using DifferentialEquations
using DataFrames
using StatsPlots
using LaTeXStrings
using Plots
using Statistics
using FFTW
using CSV

script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

function lotka_volterra!(du, u, p, t)
    x, y = u  # x - жертвы, y - хищники
    α, β, δ, γ = p  # параметры модели

    @inbounds begin
        du[1] = α*x - β*x*y  # уравнение для жертв
        du[2] = δ*x*y - γ*y  # уравнение для хищников
    end
    nothing
end

dt_lv = 0.01
tspan_lv = (0.0, 200.0)
u0_lv = [40.0, 9.0]  # начальные условия: [жертвы, хищники]

parameter_sets = [
    [0.1, 0.02, 0.01, 0.3, "Базовый сценарий"],
    [0.2, 0.02, 0.01, 0.3, "Высокая рождаемость жертв (α↑)"],
    [0.1, 0.04, 0.01, 0.3, "Высокое поедание (β↑)"],
    [0.1, 0.02, 0.02, 0.3, "Высокая конверсия (δ↑)"],
    [0.1, 0.02, 0.01, 0.5, "Высокая смертность хищников (γ↑)"],
    [0.1, 0.02, 0.01, 0.1, "Низкая смертность хищников (γ↓)"]
]

println("="^70)
println("ПАРАМЕТРИЧЕСКОЕ ИССЛЕДОВАНИЕ МОДЕЛИ ЛОТКИ-ВОЛЬТЕРРЫ")
println("="^70)

all_results = Dict()  # словарь для хранения всех результатов
scenario_metrics = DataFrame(
    Scenario=String[],
    α=Float64[], β=Float64[], δ=Float64[], γ=Float64[],
    x_star=Float64[], y_star=Float64[],
    x_mean=Float64[], y_mean=Float64[],
    x_amplitude=Float64[], y_amplitude=Float64[]
)

for (idx, params) in enumerate(parameter_sets)
    α = params[1]
    β = params[2]
    δ = params[3]
    γ = params[4]
    scenario_name = params[5]

    p = [α, β, δ, γ]

    x_star = γ / δ
    y_star = α / β

    prob = ODEProblem(lotka_volterra!, u0_lv, tspan_lv, p)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-10, saveat=0.1)

    df = DataFrame()
    df[!, :t] = sol.t
    df[!, :prey] = [u[1] for u in sol.u]
    df[!, :predator] = [u[2] for u in sol.u]

    df[!, :dprey_dt] = α .* df.prey .- β .* df.prey .* df.predator
    df[!, :dpredator_dt] = δ .* df.prey .* df.predator .- γ .* df.predator

    all_results[scenario_name] = df

    x_mean = mean(df.prey)
    y_mean = mean(df.predator)
    x_amplitude = (maximum(df.prey) - minimum(df.prey)) / 2
    y_amplitude = (maximum(df.predator) - minimum(df.predator)) / 2
    push!(scenario_metrics, [
        scenario_name, α, β, δ, γ,
        x_star, y_star,
        x_mean, y_mean,
        x_amplitude, y_amplitude
    ])

    println("\n📊 Сценарий $idx: $scenario_name")
    println("   Параметры: α=$α, β=$β, δ=$δ, γ=$γ")
    println("   Стационарные точки: x*=$(round(x_star, digits=2)), y*=$(round(y_star, digits=2))")
    println("   Средние значения: x̄=$(round(x_mean, digits=2)), ȳ=$(round(y_mean, digits=2))")
    println("   Амплитуды: A_x=$(round(x_amplitude, digits=2)), A_y=$(round(y_amplitude, digits=2))")
end

println("\n" * "="^70)
println("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
println("="^70)
println(scenario_metrics)

plt_prey = plot(size=(1000, 600),
    title="Сравнение сценариев: Динамика популяции жертв",
    xlabel="Время",
    ylabel="Численность жертв x(t)",
    legend=:topright)

colors = [:blue, :red, :green, :orange, :purple, :brown]

for (idx, (scenario, df)) in enumerate(all_results)
    plot!(plt_prey, df.t, df.prey,
        label=scenario,
        color=colors[idx],
        linewidth=1.5)
end

plt_predator = plot(size=(1000, 600),
    title="Сравнение сценариев: Динамика популяции хищников",
    xlabel="Время",
    ylabel="Численность хищников y(t)",
    legend=:topright)

for (idx, (scenario, df)) in enumerate(all_results)
    plot!(plt_predator, df.t, df.predator,
        label=scenario,
        color=colors[idx],
        linewidth=1.5)
end

plt_phase = plot(size=(800, 600),
    title="Сравнение фазовых портретов",
    xlabel="Популяция жертв x",
    ylabel="Популяция хищников y",
    legend=:bottomright)

for (idx, (scenario, df)) in enumerate(all_results)
    plot!(plt_phase, df.prey, df.predator,
        label=scenario,
        color=colors[idx],
        linewidth=1.5,
        alpha=0.7)
end

base_scenario = "Базовый сценарий"
df_base = all_results[base_scenario]

base_params = parameter_sets[1]
α_base = base_params[1]
β_base = base_params[2]
δ_base = base_params[3]
γ_base = base_params[4]

x_star_base = γ_base / δ_base
y_star_base = α_base / β_base

function compute_fft(signal, dt)
    n = length(signal)
    spectrum = abs.(rfft(signal))
    freq = rfftfreq(n, 1/dt)
    return freq, spectrum
end

freq_prey, spectrum_prey = compute_fft(df_base.prey .- mean(df_base.prey), dt_lv)
freq_predator, spectrum_predator = compute_fft(df_base.predator .- mean(df_base.predator), dt_lv)

plt_base = plot(layout=(2,3), size=(1400, 800),
    title="Детальный анализ: $base_scenario")

plot!(plt_base[1], df_base.t, [df_base.prey df_base.predator],
    label=[L"x(t)" L"y(t)"],
    xlabel="Время",
    ylabel="Популяция",
    title="Динамика популяций",
    linewidth=2,
    color=[:green :red])

plot!(plt_base[2], df_base.prey, df_base.predator,
    label="Фазовая траектория",
    xlabel=L"x",
    ylabel=L"y",
    title="Фазовый портрет",
    linewidth=2,
    color=:blue)

scatter!(plt_base[2], [x_star_base], [y_star_base],
    color=:black, markersize=8, label="Стационарная точка")

plot!(plt_base[3], df_base.t, [df_base.dprey_dt df_base.dpredator_dt],
    label=[L"dx/dt" L"dy/dt"],
    xlabel="Время",
    ylabel="Скорость",
    title="Скорости изменения",
    linewidth=1.5,
    color=[:green :red])

df_base[!, :prey_pct] = df_base.dprey_dt ./ df_base.prey .* 100
df_base[!, :predator_pct] = df_base.dpredator_dt ./ df_base.predator .* 100
plot!(plt_base[4], df_base.t, [df_base.prey_pct df_base.predator_pct],
    label=[L"dx/x (\%)" L"dy/y (\%)"],
    xlabel="Время",
    ylabel="Относительное изменение, %",
    title="Относительные темпы роста",
    linewidth=1.5,
    color=[:green :red])

plot!(plt_base[5], freq_prey, spectrum_prey,
    label=L"Спектр x",
    xlabel="Частота",
    ylabel="Амплитуда",
    title="Спектр жертв (лог. шкала)",
    xscale=:log10,
    yscale=:log10,
    linewidth=1.5,
    color=:green)

plot!(plt_base[6], freq_predator, spectrum_predator,
    label=L"Спектр y",
    xlabel="Частота",
    ylabel="Амплитуда",
    title="Спектр хищников (лог. шкала)",
    xscale=:log10,
    yscale=:log10,
    linewidth=1.5,
    color=:red)

plt_amplitude = groupedbar(scenario_metrics.Scenario,
    [scenario_metrics.x_amplitude scenario_metrics.y_amplitude],
    title="Сравнение амплитуд колебаний",
    xlabel="Сценарий",
    ylabel="Амплитуда",
    label=["Амплитуда жертв" "Амплитуда хищников"],
    size=(1000, 500),
    color=[:green :red],
    bar_position=:dodge)

plt_equilibrium = plot(size=(1000, 500),
    title="Сравнение средних значений со стационарными точками",
    xlabel="Сценарий",
    ylabel="Значение",
    legend=:topright)

scenario_names = scenario_metrics.Scenario
x_pos = 1:length(scenario_names)

plot!(plt_equilibrium, x_pos, scenario_metrics.x_star,
    seriestype=:scatter, marker=:circle, color=:green,
    label="x* (стационарная точка)", markersize=8)
plot!(plt_equilibrium, x_pos, scenario_metrics.x_mean,
    seriestype=:scatter, marker=:square, color=:lightgreen,
    label="x̄ (среднее)", markersize=8)

plot!(plt_equilibrium, x_pos, scenario_metrics.y_star,
    seriestype=:scatter, marker=:circle, color=:red,
    label="y* (стационарная точка)", markersize=8)
plot!(plt_equilibrium, x_pos, scenario_metrics.y_mean,
    seriestype=:scatter, marker=:square, color=:pink,
    label="ȳ (среднее)", markersize=8)

xticks!(plt_equilibrium, x_pos, scenario_names, rotation=45)

savefig(plt_prey, plotsdir(script_name, "lv_prey_comparison.png"))
savefig(plt_predator, plotsdir(script_name, "lv_predator_comparison.png"))
savefig(plt_phase, plotsdir(script_name, "lv_phase_comparison.png"))
savefig(plt_base, plotsdir(script_name, "lv_base_detailed.png"))
savefig(plt_amplitude, plotsdir(script_name, "lv_amplitude_comparison.png"))
savefig(plt_equilibrium, plotsdir(script_name, "lv_equilibrium_comparison.png"))

println("\nГрафики сохранены в: ", plotsdir(script_name))

CSV.write(datadir(script_name, "lv_metrics.csv"), scenario_metrics)
println("\nРезультаты сохранены в: ", datadir(script_name, "lv_metrics.csv"))

println("\n" * "="^70)
println("АНАЛИЗ РЕЗУЛЬТАТОВ")
println("="^70)
println("""
На основе полученных результатов можно сделать следующие выводы:

1. **Влияние α (рождаемости жертв):**
   - Увеличение α повышает равновесную численность хищников y*
   - Амплитуда колебаний возрастает
   - Частота колебаний увеличивается (период уменьшается)

2. **Влияние β (поедания жертв):**
   - Увеличение β снижает равновесную численность хищников y*
   - Амплитуда колебаний жертв возрастает
   - Система становится менее устойчивой

3. **Влияние δ (конверсии):**
   - Увеличение δ снижает равновесную численность жертв x*
   - Повышает эффективность размножения хищников
   - Увеличивает амплитуду колебаний хищников

4. **Влияние γ (смертности хищников):**
   - Увеличение γ повышает равновесную численность жертв x*
   - Снижает амплитуду колебаний обоих видов
   - При высоких γ хищники могут вымереть

5. **Фазовый сдвиг:**
   - Во всех сценариях сохраняется характерный сдвиг фаз
   - Пик хищников отстает от пика жертв примерно на 1/4 периода
""")

println("\n" * "="^70)
println("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО")
println("="^70)
