# # Параметрическое исследование трёх дифференциальных моделей

# ## Активация проекта и загрузка пакетов
using DrWatson
@quickactivate "project"

using DifferentialEquations
using DataFrames
using Plots
using JLD2
using BenchmarkTools
using CSV
using Statistics

# ## Установка каталогов
script_name = isempty(PROGRAM_FILE) ? "interactive" : splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

# ## Определение моделей

# Первая модель:
# dn/dt = (a + b*n)*(N - n)
function model_case_1!(dy, y, p, t)
    dy[1] = (p.a + p.b * y[1]) * (p.N - y[1])
end

# Вторая модель:
# dn/dt = (a + b*n)*(N - n)
function model_case_2!(dy, y, p, t)
    dy[1] = (p.a + p.b * y[1]) * (p.N - y[1])
end

# Третья модель:
# dn/dt = (a*cos(t) + b*cos(2*t)*n)*(N - n)
function model_case_3!(dy, y, p, t)
    dy[1] = (p.a * cos(t) + p.b * cos(2 * t) * y[1]) * (p.N - y[1])
end

# ## Выбор модели по типу эксперимента
function choose_model(model_type)
    if model_type == :case1
        return model_case_1!
    elseif model_type == :case2
        return model_case_2!
    elseif model_type == :case3
        return model_case_3!
    else
        error("Неизвестный тип модели: $model_type")
    end
end

# ## Правая часть уравнения для анализа производной
function rhs_value(model_type, n, p, t)
    if model_type == :case1
        return (p.a + p.b * n) * (p.N - n)
    elseif model_type == :case2
        return (p.a + p.b * n) * (p.N - n)
    elseif model_type == :case3
        return (p.a * cos(t) + p.b * cos(2 * t) * n) * (p.N - n)
    else
        error("Неизвестный тип модели: $model_type")
    end
end

# ## Базовые параметры первой модели
base_params1 = Dict(
    :N => 609.0,
    :n0 => 4.0,

    :tspan => (0.0, 10.0),
    :nt => 500,

    :model_type => :case1,

    :a => 0.54,
    :b => 0.00016,

    :solver => Tsit5(),
    :experiment_name => "base_experiment_case1"
)

# ## Базовые параметры второй модели
base_params2 = Dict(
    :N => 609.0,
    :n0 => 4.0,

    :tspan => (0.0, 0.04),
    :nt => 500,

    :model_type => :case2,

    :a => 0.000021,
    :b => 0.38,

    :solver => Tsit5(),
    :experiment_name => "base_experiment_case2"
)

# ## Базовые параметры третьей модели
base_params3 = Dict(
    :N => 609.0,
    :n0 => 4.0,

    :tspan => (0.0, 0.15),
    :nt => 500,

    :model_type => :case3,

    :a => 0.2,
    :b => 0.2,

    :solver => Tsit5(),
    :experiment_name => "base_experiment_case3"
)

base_experiments = [base_params1, base_params2, base_params3]

println("Базовые параметры экспериментов:")
for params in base_experiments
    println("\nmodel_type = ", params[:model_type])
    println(" N = ", params[:N])
    println(" n0 = ", params[:n0])
    println(" tspan = ", params[:tspan])
    println(" nt = ", params[:nt])
    println(" a = ", params[:a])
    println(" b = ", params[:b])
end

# ## Функция для запуска одного эксперимента
function run_single_experiment(params::Dict)
    N = params[:N]
    n0 = params[:n0]

    tspan = params[:tspan]
    nt = params[:nt]

    model_type = params[:model_type]
    model! = choose_model(model_type)

    a = params[:a]
    b = params[:b]

    solver = params[:solver]

    u0 = [n0]
    tgrid = collect(LinRange(tspan[1], tspan[2], nt))

    p = (a = a, b = b, N = N)

    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, solver; saveat = tgrid)

    t_vals = sol.t
    n_vals = [u[1] for u in sol.u]
    dn_vals = [rhs_value(model_type, n_vals[i], p, t_vals[i]) for i in eachindex(t_vals)]

    # --- Мини-анализ ---
    n_initial = n_vals[1]
    n_final = n_vals[end]

    n_max = maximum(n_vals)
    n_min = minimum(n_vals)
    n_mean = mean(n_vals)

    dn_final = dn_vals[end]
    dn_max = maximum(dn_vals)
    dn_min = minimum(dn_vals)
    dn_mean = mean(dn_vals)

    growth_abs = n_final - n_initial
    growth_rel = growth_abs / n_initial

    saturation_final = n_final / N
    saturation_max = n_max / N

    idx_n_max = argmax(n_vals)
    t_at_n_max = t_vals[idx_n_max]

    idx_90 = findfirst(x -> x >= 0.9 * N, n_vals)
    t_reach_90 = isnothing(idx_90) ? missing : t_vals[idx_90]

    idx_95 = findfirst(x -> x >= 0.95 * N, n_vals)
    t_reach_95 = isnothing(idx_95) ? missing : t_vals[idx_95]

    return Dict(
        "solution" => sol,
        "t_points" => t_vals,

        "n_values" => n_vals,
        "dn_values" => dn_vals,

        "n_initial" => n_initial,
        "n_final" => n_final,

        "n_max" => n_max,
        "n_min" => n_min,
        "n_mean" => n_mean,

        "dn_final" => dn_final,
        "dn_max" => dn_max,
        "dn_min" => dn_min,
        "dn_mean" => dn_mean,

        "growth_abs" => growth_abs,
        "growth_rel" => growth_rel,

        "saturation_final" => saturation_final,
        "saturation_max" => saturation_max,

        "t_at_n_max" => t_at_n_max,
        "t_reach_90" => t_reach_90,
        "t_reach_95" => t_reach_95,

        "parameters" => params
    )
end

# ## Функция печати результатов одного эксперимента
function print_experiment_summary(data, path)
    println(" n_initial: ", data["n_initial"])
    println(" n_final: ", data["n_final"])
    println(" n_min: ", data["n_min"])
    println(" n_max: ", data["n_max"])
    println(" n_mean: ", round(data["n_mean"]; digits = 4))

    println(" dn_final: ", data["dn_final"])
    println(" dn_min: ", data["dn_min"])
    println(" dn_max: ", data["dn_max"])
    println(" dn_mean: ", round(data["dn_mean"]; digits = 4))

    println(" growth_abs: ", data["growth_abs"])
    println(" growth_rel: ", round(data["growth_rel"]; digits = 4))

    println(" saturation_final: ", round(data["saturation_final"]; digits = 4))
    println(" saturation_max: ", round(data["saturation_max"]; digits = 4))

    println(" t_at_n_max: ", data["t_at_n_max"])
    println(" t_reach_90: ", data["t_reach_90"])
    println(" t_reach_95: ", data["t_reach_95"])

    println(" Файл результатов: ", path)
end

# ## Функция построения графика n(t)
function plot_time_series(data, params)
    model_type = params[:model_type]

    plt = plot(
        data["t_points"],
        data["n_values"],
        xlabel = "t",
        ylabel = "n(t)",
        label = "n(t)",
        title = "Динамика n(t): model_type=$model_type",
        lw = 2,
        legend = :bottomright,
        grid = true
    )

    hline!(
        plt,
        [params[:N]],
        label = "N",
        lw = 2,
        linestyle = :dash
    )

    return plt
end

# ## Функция построения графика производной dn/dt
function plot_derivative_series(data, params)
    model_type = params[:model_type]

    plt = plot(
        data["t_points"],
        data["dn_values"],
        xlabel = "t",
        ylabel = "dn/dt",
        label = "dn/dt",
        title = "Скорость изменения: model_type=$model_type",
        lw = 2,
        legend = :topright,
        grid = true
    )

    return plt
end

# ## Функция построения фазовой траектории dn/dt от n
function plot_phase_n_dn(data, params)
    model_type = params[:model_type]

    plt = plot(
        data["n_values"],
        data["dn_values"],
        xlabel = "n",
        ylabel = "dn/dt",
        label = "Фазовая траектория",
        title = "Фазовая траектория dn/dt(n): model_type=$model_type",
        lw = 2,
        legend = :topright,
        grid = true
    )

    return plt
end

# ## Запуск базовых экспериментов
base_results = Dict()

for params in base_experiments
    model_type = params[:model_type]

    println("\n" * "="^60)
    println("БАЗОВЫЙ ЭКСПЕРИМЕНТ: model_type=$model_type")
    println("="^60)

    data, path = produce_or_load(
        datadir(script_name, "single"),
        params,
        run_single_experiment;
        prefix = "model",
        tag = false,
        verbose = true
    )

    base_results[model_type] = Dict(
        "data" => data,
        "path" => path
    )

    print_experiment_summary(data, path)

    plt_time = plot_time_series(data, params)
    savefig(plt_time, plotsdir(script_name, "single_experiment_$(model_type).png"))

    plt_derivative = plot_derivative_series(data, params)
    savefig(plt_derivative, plotsdir(script_name, "single_experiment_$(model_type)_derivative.png"))

    plt_phase = plot_phase_n_dn(data, params)
    savefig(plt_phase, plotsdir(script_name, "single_experiment_$(model_type)_phase_n_dn.png"))

    df_base = DataFrame(
        t = data["t_points"],
        n = data["n_values"],
        dn = data["dn_values"],
        model_type = fill(string(model_type), length(data["t_points"])),
        a = fill(params[:a], length(data["t_points"])),
        b = fill(params[:b], length(data["t_points"])),
        N = fill(params[:N], length(data["t_points"]))
    )

    CSV.write(datadir(script_name, "table_single_$(model_type).csv"), df_base)
end

# ## Параметрическое сканирование

# Сетка параметров для первой модели
param_grid_case1 = Dict(
    :N => [609.0],
    :n0 => [4.0],

    :tspan => [(0.0, 10.0)],
    :nt => [500],

    :model_type => [:case1],

    :a => [0.35, 0.54, 0.70],
    :b => [0.00008, 0.00016, 0.00024],

    :solver => [Tsit5()],
    :experiment_name => ["parametric_scan_case1"]
)

# Сетка параметров для второй модели
param_grid_case2 = Dict(
    :N => [609.0],
    :n0 => [4.0],

    :tspan => [(0.0, 0.04)],
    :nt => [500],

    :model_type => [:case2],

    :a => [0.00001, 0.000021, 0.00004],
    :b => [0.25, 0.38, 0.50],

    :solver => [Tsit5()],
    :experiment_name => ["parametric_scan_case2"]
)

# Сетка параметров для третьей модели
param_grid_case3 = Dict(
    :N => [609.0],
    :n0 => [4.0],

    :tspan => [(0.0, 0.15)],
    :nt => [500],

    :model_type => [:case3],

    :a => [0.10, 0.20, 0.30],
    :b => [0.10, 0.20, 0.30],

    :solver => [Tsit5()],
    :experiment_name => ["parametric_scan_case3"]
)

all_params = vcat(
    dict_list(param_grid_case1),
    dict_list(param_grid_case2),
    dict_list(param_grid_case3)
)

println("\n" * "="^60)
println("ПАРАМЕТРИЧЕСКОЕ СКАНИРОВАНИЕ")
println("Всего комбинаций параметров: ", length(all_params))
println("="^60)

# ## Запуск всех экспериментов
all_results = []
all_dfs = []

for (i, params) in enumerate(all_params)
    println(
        "Прогресс: $i/$(length(all_params)) | ",
        "model_type=$(params[:model_type]) | ",
        "a=$(params[:a]) | b=$(params[:b])"
    )

    data, path = produce_or_load(
        datadir(script_name, "parametric_scan"),
        params,
        run_single_experiment;
        prefix = "scan",
        tag = false,
        verbose = false
    )

    result_summary = (
        model_type = string(params[:model_type]),

        N = params[:N],
        n0 = params[:n0],

        t_start = params[:tspan][1],
        t_end = params[:tspan][2],
        nt = params[:nt],

        a = params[:a],
        b = params[:b],

        n_initial = data["n_initial"],
        n_final = data["n_final"],

        n_min = data["n_min"],
        n_max = data["n_max"],
        n_mean = data["n_mean"],

        dn_final = data["dn_final"],
        dn_min = data["dn_min"],
        dn_max = data["dn_max"],
        dn_mean = data["dn_mean"],

        growth_abs = data["growth_abs"],
        growth_rel = data["growth_rel"],

        saturation_final = data["saturation_final"],
        saturation_max = data["saturation_max"],

        t_at_n_max = data["t_at_n_max"],
        t_reach_90 = data["t_reach_90"],
        t_reach_95 = data["t_reach_95"],

        filepath = path
    )

    push!(all_results, result_summary)

    df = DataFrame(
        t = data["t_points"],
        n = data["n_values"],
        dn = data["dn_values"],

        model_type = fill(string(params[:model_type]), length(data["t_points"])),

        N = fill(params[:N], length(data["t_points"])),
        n0 = fill(params[:n0], length(data["t_points"])),

        a = fill(params[:a], length(data["t_points"])),
        b = fill(params[:b], length(data["t_points"])),

        t_start = fill(params[:tspan][1], length(data["t_points"])),
        t_end = fill(params[:tspan][2], length(data["t_points"]))
    )

    push!(all_dfs, df)
end

# ## Анализ результатов сканирования
results_df = DataFrame(all_results)
full_timeseries_df = vcat(all_dfs...)

println("\nСводная таблица результатов:")
println(first(results_df, 10))

CSV.write(datadir(script_name, "results_summary.csv"), results_df)
CSV.write(datadir(script_name, "timeseries_full.csv"), full_timeseries_df)

# ## Сравнительные графики n(t) для каждой модели
model_types = unique(results_df.model_type)

for model_type_string in model_types
    params_subset = filter(params -> string(params[:model_type]) == model_type_string, all_params)

    plt = plot(size = (950, 520), dpi = 150)

    for params in params_subset
        data, _ = produce_or_load(
            datadir(script_name, "parametric_scan"),
            params,
            run_single_experiment;
            prefix = "scan",
            tag = false,
            verbose = false
        )

        label_text = "a=$(params[:a]), b=$(params[:b])"

        plot!(
            plt,
            data["t_points"],
            data["n_values"],
            label = label_text,
            lw = 2,
            alpha = 0.8
        )
    end

    plot!(
        plt,
        xlabel = "t",
        ylabel = "n(t)",
        title = "Сканирование: траектории n(t), model_type=$model_type_string",
        legend = :outerright,
        grid = true
    )

    savefig(plt, plotsdir(script_name, "parametric_scan_n_comparison_$(model_type_string).png"))
end

# ## Сравнительные графики dn/dt для каждой модели
for model_type_string in model_types
    params_subset = filter(params -> string(params[:model_type]) == model_type_string, all_params)

    plt = plot(size = (950, 520), dpi = 150)

    for params in params_subset
        data, _ = produce_or_load(
            datadir(script_name, "parametric_scan"),
            params,
            run_single_experiment;
            prefix = "scan",
            tag = false,
            verbose = false
        )

        label_text = "a=$(params[:a]), b=$(params[:b])"

        plot!(
            plt,
            data["t_points"],
            data["dn_values"],
            label = label_text,
            lw = 2,
            alpha = 0.8
        )
    end

    plot!(
        plt,
        xlabel = "t",
        ylabel = "dn/dt",
        title = "Сканирование: скорость изменения, model_type=$model_type_string",
        legend = :outerright,
        grid = true
    )

    savefig(plt, plotsdir(script_name, "parametric_scan_dn_comparison_$(model_type_string).png"))
end

# ## Сравнительные фазовые траектории dn/dt(n)
for model_type_string in model_types
    params_subset = filter(params -> string(params[:model_type]) == model_type_string, all_params)

    plt = plot(size = (950, 520), dpi = 150)

    for params in params_subset
        data, _ = produce_or_load(
            datadir(script_name, "parametric_scan"),
            params,
            run_single_experiment;
            prefix = "scan",
            tag = false,
            verbose = false
        )

        label_text = "a=$(params[:a]), b=$(params[:b])"

        plot!(
            plt,
            data["n_values"],
            data["dn_values"],
            label = label_text,
            lw = 2,
            alpha = 0.8
        )
    end

    plot!(
        plt,
        xlabel = "n",
        ylabel = "dn/dt",
        title = "Сканирование: фазовые траектории dn/dt(n), model_type=$model_type_string",
        legend = :outerright,
        grid = true
    )

    savefig(plt, plotsdir(script_name, "parametric_scan_phase_n_dn_$(model_type_string).png"))
end

# ## График зависимости n_final от параметра a
p_a_final = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_a_final,
            sub.a,
            sub.n_final,
            seriestype = :scatter,
            label = "$model_type_string: n_final от a"
        )
    end
end

plot!(
    p_a_final,
    xlabel = "a",
    ylabel = "n_final",
    title = "Зависимость итогового значения n от параметра a",
    legend = :bottomright,
    grid = true
)

savefig(p_a_final, plotsdir(script_name, "n_final_vs_a.png"))

# ## График зависимости n_final от параметра b
p_b_final = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_b_final,
            sub.b,
            sub.n_final,
            seriestype = :scatter,
            label = "$model_type_string: n_final от b"
        )
    end
end

plot!(
    p_b_final,
    xlabel = "b",
    ylabel = "n_final",
    title = "Зависимость итогового значения n от параметра b",
    legend = :bottomright,
    grid = true
)

savefig(p_b_final, plotsdir(script_name, "n_final_vs_b.png"))

# ## График зависимости n_max от параметра a
p_a_max = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_a_max,
            sub.a,
            sub.n_max,
            seriestype = :scatter,
            label = "$model_type_string: n_max от a"
        )
    end
end

plot!(
    p_a_max,
    xlabel = "a",
    ylabel = "n_max",
    title = "Зависимость максимального значения n от параметра a",
    legend = :bottomright,
    grid = true
)

savefig(p_a_max, plotsdir(script_name, "n_max_vs_a.png"))

# ## График зависимости n_max от параметра b
p_b_max = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_b_max,
            sub.b,
            sub.n_max,
            seriestype = :scatter,
            label = "$model_type_string: n_max от b"
        )
    end
end

plot!(
    p_b_max,
    xlabel = "b",
    ylabel = "n_max",
    title = "Зависимость максимального значения n от параметра b",
    legend = :bottomright,
    grid = true
)

savefig(p_b_max, plotsdir(script_name, "n_max_vs_b.png"))

# ## График зависимости saturation_final от параметра a
p_a_saturation = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_a_saturation,
            sub.a,
            sub.saturation_final,
            seriestype = :scatter,
            label = "$model_type_string: saturation_final от a"
        )
    end
end

plot!(
    p_a_saturation,
    xlabel = "a",
    ylabel = "n_final / N",
    title = "Зависимость финального насыщения от параметра a",
    legend = :bottomright,
    grid = true
)

savefig(p_a_saturation, plotsdir(script_name, "saturation_final_vs_a.png"))

# ## График зависимости saturation_final от параметра b
p_b_saturation = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_b_saturation,
            sub.b,
            sub.saturation_final,
            seriestype = :scatter,
            label = "$model_type_string: saturation_final от b"
        )
    end
end

plot!(
    p_b_saturation,
    xlabel = "b",
    ylabel = "n_final / N",
    title = "Зависимость финального насыщения от параметра b",
    legend = :bottomright,
    grid = true
)

savefig(p_b_saturation, plotsdir(script_name, "saturation_final_vs_b.png"))

# ## Бенчмаркинг
println("\n" * "="^60)
println("БЕНЧМАРКИНГ ДЛЯ РАЗНЫХ ПАРАМЕТРОВ")
println("="^60)

benchmark_results = []

for params in all_params
    function benchmark_run()
        N = params[:N]
        n0 = params[:n0]

        u0 = [n0]
        p = (a = params[:a], b = params[:b], N = N)

        model! = choose_model(params[:model_type])

        prob = ODEProblem(model!, u0, params[:tspan], p)

        return solve(
            prob,
            params[:solver];
            saveat = LinRange(params[:tspan][1], params[:tspan][2], params[:nt])
        )
    end

    println(
        "\nБенчмарк для model_type=$(params[:model_type]), ",
        "a=$(params[:a]), b=$(params[:b]):"
    )

    bmark = @benchmark $benchmark_run() samples = 80 evals = 1
    tsec = median(bmark).time / 1e9

    println(" Медианное время: ", round(tsec; digits = 6), " сек")

    push!(
        benchmark_results,
        (
            model_type = string(params[:model_type]),
            a = params[:a],
            b = params[:b],
            time = tsec
        )
    )
end

bench_df = DataFrame(benchmark_results)
CSV.write(datadir(script_name, "benchmark_results.csv"), bench_df)

# ## График времени вычисления от параметра a
p_time_a = plot(size = (900, 520), dpi = 150)

for model_type_string in unique(bench_df.model_type)
    sub = bench_df[bench_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_time_a,
            sub.a,
            sub.time,
            seriestype = :scatter,
            label = "$model_type_string: время от a"
        )
    end
end

plot!(
    p_time_a,
    xlabel = "a",
    ylabel = "Время вычисления, сек",
    title = "Зависимость времени решения ODE от параметра a",
    legend = :topright,
    grid = true
)

savefig(p_time_a, plotsdir(script_name, "computation_time_vs_a.png"))

# ## График времени вычисления от параметра b
p_time_b = plot(size = (900, 520), dpi = 150)

for model_type_string in unique(bench_df.model_type)
    sub = bench_df[bench_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_time_b,
            sub.b,
            sub.time,
            seriestype = :scatter,
            label = "$model_type_string: время от b"
        )
    end
end

plot!(
    p_time_b,
    xlabel = "b",
    ylabel = "Время вычисления, сек",
    title = "Зависимость времени решения ODE от параметра b",
    legend = :topright,
    grid = true
)

savefig(p_time_b, plotsdir(script_name, "computation_time_vs_b.png"))

# ## Сохранение всех результатов
@save datadir(script_name, "all_results.jld2") base_experiments base_results param_grid_case1 param_grid_case2 param_grid_case3 all_params results_df bench_df full_timeseries_df

@save datadir(script_name, "all_plots.jld2") p_a_final p_b_final p_a_max p_b_max p_a_saturation p_b_saturation p_time_a p_time_b

println("\n" * "="^60)
println("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
println("="^60)

println("\nРезультаты сохранены в:")
println(" • data/$(script_name)/single/ - базовые эксперименты")
println(" • data/$(script_name)/parametric_scan/ - параметрическое сканирование")
println(" • data/$(script_name)/table_single_case1.csv - таблица для первой модели")
println(" • data/$(script_name)/table_single_case2.csv - таблица для второй модели")
println(" • data/$(script_name)/table_single_case3.csv - таблица для третьей модели")
println(" • data/$(script_name)/results_summary.csv - сводная таблица")
println(" • data/$(script_name)/timeseries_full.csv - полные временные ряды")
println(" • data/$(script_name)/benchmark_results.csv - результаты бенчмаркинга")
println(" • data/$(script_name)/all_results.jld2 - сводные данные")
println(" • data/$(script_name)/all_plots.jld2 - объекты графиков")
println(" • plots/$(script_name)/ - все графики")