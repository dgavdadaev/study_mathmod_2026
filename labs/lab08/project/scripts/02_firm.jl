# # Параметрическое исследование модели конкуренции двух фирм
#
# **Цель:** исследовать две модели конкуренции фирм:
#
# 1. Случай рыночной конкуренции без социально-психологического влияния.
# 2. Случай конкуренции с дополнительным социально-психологическим фактором.
#
# Для каждой модели:
# - решить систему дифференциальных уравнений;
# - построить графики M₁(t) и M₂(t);
# - построить графики производных dM₁/dt и dM₂/dt;
# - построить фазовую траекторию M₂(M₁);
# - выполнить параметрическое сканирование;
# - сохранить таблицы, графики и JLD2-файлы;
# - выполнить бенчмаркинг.

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

# ## Расчёт коэффициентов модели

function calculate_coefficients(params::Dict)
    p_cr = params[:p_cr]
    N = params[:N]
    q = params[:q]

    tau_1 = params[:tau_1]
    tau_2 = params[:tau_2]

    p_tilde_1 = params[:p_tilde_1]
    p_tilde_2 = params[:p_tilde_2]

    a_1 = p_cr / (tau_1^2 * p_tilde_1^2 * N * q)

    a_2 = p_cr / (tau_2^2 * p_tilde_2^2 * N * q)

    b = p_cr / (
        tau_1^2 * p_tilde_1^2 *
        tau_2^2 * p_tilde_2^2 *
        N * q
    )

    c_1 = (p_cr - p_tilde_1) / (tau_1 * p_tilde_1)

    c_2 = (p_cr - p_tilde_2) / (tau_2 * p_tilde_2)

    return (
        a_1 = a_1,
        a_2 = a_2,
        b = b,
        c_1 = c_1,
        c_2 = c_2,
        d = params[:d],

        p_cr = p_cr,
        N = N,
        q = q,
        tau_1 = tau_1,
        tau_2 = tau_2,
        p_tilde_1 = p_tilde_1,
        p_tilde_2 = p_tilde_2
    )
end

# ## Определение моделей

# Первая модель:
#
# dM₁/dt = M₁ - (b / c₁)M₁M₂ - (a₁ / c₁)M₁²
#
# dM₂/dt = (c₂ / c₁)M₂ - (b / c₁)M₁M₂ - (a₂ / c₁)M₂²

function model_case_1!(dy, y, p, t)
    M1 = y[1]
    M2 = y[2]

    dy[1] = M1 -
            (p.b / p.c_1) * M1 * M2 -
            (p.a_1 / p.c_1) * M1^2

    dy[2] = (p.c_2 / p.c_1) * M2 -
            (p.b / p.c_1) * M1 * M2 -
            (p.a_2 / p.c_1) * M2^2
end

# Вторая модель:
#
# dM₁/dt = M₁ - ((b / c₁) + d)M₁M₂ - (a₁ / c₁)M₁²
#
# dM₂/dt = (c₂ / c₁)M₂ - (b / c₁)M₁M₂ - (a₂ / c₁)M₂²
#
# Добавка d учитывает дополнительный социально-психологический фактор.

function model_case_2!(dy, y, p, t)
    M1 = y[1]
    M2 = y[2]

    dy[1] = M1 -
            ((p.b / p.c_1) + p.d) * M1 * M2 -
            (p.a_1 / p.c_1) * M1^2

    dy[2] = (p.c_2 / p.c_1) * M2 -
            (p.b / p.c_1) * M1 * M2 -
            (p.a_2 / p.c_1) * M2^2
end

# ## Выбор модели

function choose_model(model_type)
    if model_type == :case1
        return model_case_1!
    elseif model_type == :case2
        return model_case_2!
    else
        error("Неизвестный тип модели: $model_type")
    end
end

# ## Правая часть системы для анализа производных

function rhs_values(model_type, M1, M2, p, t)
    if model_type == :case1
        dM1 = M1 -
              (p.b / p.c_1) * M1 * M2 -
              (p.a_1 / p.c_1) * M1^2

        dM2 = (p.c_2 / p.c_1) * M2 -
              (p.b / p.c_1) * M1 * M2 -
              (p.a_2 / p.c_1) * M2^2

        return dM1, dM2

    elseif model_type == :case2
        dM1 = M1 -
              ((p.b / p.c_1) + p.d) * M1 * M2 -
              (p.a_1 / p.c_1) * M1^2

        dM2 = (p.c_2 / p.c_1) * M2 -
              (p.b / p.c_1) * M1 * M2 -
              (p.a_2 / p.c_1) * M2^2

        return dM1, dM2

    else
        error("Неизвестный тип модели: $model_type")
    end
end

# ## Базовые параметры первой модели

base_params1 = Dict(
    :p_cr => 26.0,
    :N => 33.0,
    :q => 1.0,

    :tau_1 => 25.0,
    :tau_2 => 14.0,

    :p_tilde_1 => 5.5,
    :p_tilde_2 => 11.0,

    :d => 0.00033,

    :M1_0 => 3.3,
    :M2_0 => 2.2,

    :tspan => (0.0, 20.0),
    :nt => 500,

    :model_type => :case1,

    :solver => Tsit5(),
    :experiment_name => "base_experiment_case1"
)

# ## Базовые параметры второй модели

base_params2 = Dict(
    :p_cr => 26.0,
    :N => 33.0,
    :q => 1.0,

    :tau_1 => 25.0,
    :tau_2 => 14.0,

    :p_tilde_1 => 5.5,
    :p_tilde_2 => 11.0,

    :d => 0.00033,

    :M1_0 => 3.3,
    :M2_0 => 2.2,

    :tspan => (0.0, 20.0),
    :nt => 500,

    :model_type => :case2,

    :solver => Tsit5(),
    :experiment_name => "base_experiment_case2"
)

base_experiments = [base_params1, base_params2]

println("Базовые параметры экспериментов:")

for params in base_experiments
    println("\nmodel_type = ", params[:model_type])
    println(" p_cr = ", params[:p_cr])
    println(" N = ", params[:N])
    println(" q = ", params[:q])
    println(" tau_1 = ", params[:tau_1])
    println(" tau_2 = ", params[:tau_2])
    println(" p_tilde_1 = ", params[:p_tilde_1])
    println(" p_tilde_2 = ", params[:p_tilde_2])
    println(" d = ", params[:d])
    println(" M1_0 = ", params[:M1_0])
    println(" M2_0 = ", params[:M2_0])
    println(" tspan = ", params[:tspan])
    println(" nt = ", params[:nt])
end

# ## Функция для запуска одного эксперимента

function run_single_experiment(params::Dict)
    M1_0 = params[:M1_0]
    M2_0 = params[:M2_0]

    tspan = params[:tspan]
    nt = params[:nt]

    model_type = params[:model_type]
    model! = choose_model(model_type)

    solver = params[:solver]

    u0 = [M1_0, M2_0]
    tgrid = collect(LinRange(tspan[1], tspan[2], nt))

    p = calculate_coefficients(params)

    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, solver; saveat = tgrid)

    t_vals = sol.t

    M1_vals = [u[1] for u in sol.u]
    M2_vals = [u[2] for u in sol.u]

    dM1_vals = Float64[]
    dM2_vals = Float64[]

    for i in eachindex(t_vals)
        dM1, dM2 = rhs_values(model_type, M1_vals[i], M2_vals[i], p, t_vals[i])

        push!(dM1_vals, dM1)
        push!(dM2_vals, dM2)
    end

    total_vals = M1_vals .+ M2_vals
    difference_vals = M1_vals .- M2_vals
    ratio_vals = M1_vals ./ M2_vals

    # --- Мини-анализ M₁ ---

    M1_initial = M1_vals[1]
    M1_final = M1_vals[end]

    M1_min = minimum(M1_vals)
    M1_max = maximum(M1_vals)
    M1_mean = mean(M1_vals)

    M1_growth_abs = M1_final - M1_initial
    M1_growth_rel = M1_growth_abs / M1_initial

    idx_M1_max = argmax(M1_vals)
    t_at_M1_max = t_vals[idx_M1_max]

    # --- Мини-анализ M₂ ---

    M2_initial = M2_vals[1]
    M2_final = M2_vals[end]

    M2_min = minimum(M2_vals)
    M2_max = maximum(M2_vals)
    M2_mean = mean(M2_vals)

    M2_growth_abs = M2_final - M2_initial
    M2_growth_rel = M2_growth_abs / M2_initial

    idx_M2_max = argmax(M2_vals)
    t_at_M2_max = t_vals[idx_M2_max]

    # --- Анализ производных ---

    dM1_final = dM1_vals[end]
    dM1_min = minimum(dM1_vals)
    dM1_max = maximum(dM1_vals)
    dM1_mean = mean(dM1_vals)

    dM2_final = dM2_vals[end]
    dM2_min = minimum(dM2_vals)
    dM2_max = maximum(dM2_vals)
    dM2_mean = mean(dM2_vals)

    # --- Анализ суммарного объёма и разницы ---

    total_initial = total_vals[1]
    total_final = total_vals[end]
    total_min = minimum(total_vals)
    total_max = maximum(total_vals)
    total_mean = mean(total_vals)

    difference_initial = difference_vals[1]
    difference_final = difference_vals[end]
    difference_min = minimum(difference_vals)
    difference_max = maximum(difference_vals)
    difference_mean = mean(difference_vals)

    ratio_initial = ratio_vals[1]
    ratio_final = ratio_vals[end]
    ratio_min = minimum(ratio_vals)
    ratio_max = maximum(ratio_vals)
    ratio_mean = mean(ratio_vals)

    final_leader = M1_final > M2_final ? "firm_1" : M2_final > M1_final ? "firm_2" : "equal"

    idx_equal = findfirst(x -> abs(x) <= 1e-3, difference_vals)
    t_near_equal = isnothing(idx_equal) ? missing : t_vals[idx_equal]

    return Dict(
        "solution" => sol,

        "t_points" => t_vals,

        "M1_values" => M1_vals,
        "M2_values" => M2_vals,

        "dM1_values" => dM1_vals,
        "dM2_values" => dM2_vals,

        "total_values" => total_vals,
        "difference_values" => difference_vals,
        "ratio_values" => ratio_vals,

        "M1_initial" => M1_initial,
        "M1_final" => M1_final,
        "M1_min" => M1_min,
        "M1_max" => M1_max,
        "M1_mean" => M1_mean,
        "M1_growth_abs" => M1_growth_abs,
        "M1_growth_rel" => M1_growth_rel,
        "t_at_M1_max" => t_at_M1_max,

        "M2_initial" => M2_initial,
        "M2_final" => M2_final,
        "M2_min" => M2_min,
        "M2_max" => M2_max,
        "M2_mean" => M2_mean,
        "M2_growth_abs" => M2_growth_abs,
        "M2_growth_rel" => M2_growth_rel,
        "t_at_M2_max" => t_at_M2_max,

        "dM1_final" => dM1_final,
        "dM1_min" => dM1_min,
        "dM1_max" => dM1_max,
        "dM1_mean" => dM1_mean,

        "dM2_final" => dM2_final,
        "dM2_min" => dM2_min,
        "dM2_max" => dM2_max,
        "dM2_mean" => dM2_mean,

        "total_initial" => total_initial,
        "total_final" => total_final,
        "total_min" => total_min,
        "total_max" => total_max,
        "total_mean" => total_mean,

        "difference_initial" => difference_initial,
        "difference_final" => difference_final,
        "difference_min" => difference_min,
        "difference_max" => difference_max,
        "difference_mean" => difference_mean,

        "ratio_initial" => ratio_initial,
        "ratio_final" => ratio_final,
        "ratio_min" => ratio_min,
        "ratio_max" => ratio_max,
        "ratio_mean" => ratio_mean,

        "final_leader" => final_leader,
        "t_near_equal" => t_near_equal,

        "coefficients" => p,
        "parameters" => params
    )
end

# ## Функция печати результатов одного эксперимента

function print_experiment_summary(data, path)
    println(" M1_initial: ", data["M1_initial"])
    println(" M1_final: ", data["M1_final"])
    println(" M1_min: ", data["M1_min"])
    println(" M1_max: ", data["M1_max"])
    println(" M1_mean: ", round(data["M1_mean"]; digits = 4))
    println(" M1_growth_abs: ", data["M1_growth_abs"])
    println(" M1_growth_rel: ", round(data["M1_growth_rel"]; digits = 4))
    println(" t_at_M1_max: ", data["t_at_M1_max"])

    println()

    println(" M2_initial: ", data["M2_initial"])
    println(" M2_final: ", data["M2_final"])
    println(" M2_min: ", data["M2_min"])
    println(" M2_max: ", data["M2_max"])
    println(" M2_mean: ", round(data["M2_mean"]; digits = 4))
    println(" M2_growth_abs: ", data["M2_growth_abs"])
    println(" M2_growth_rel: ", round(data["M2_growth_rel"]; digits = 4))
    println(" t_at_M2_max: ", data["t_at_M2_max"])

    println()

    println(" dM1_final: ", data["dM1_final"])
    println(" dM1_min: ", data["dM1_min"])
    println(" dM1_max: ", data["dM1_max"])
    println(" dM1_mean: ", round(data["dM1_mean"]; digits = 4))

    println()

    println(" dM2_final: ", data["dM2_final"])
    println(" dM2_min: ", data["dM2_min"])
    println(" dM2_max: ", data["dM2_max"])
    println(" dM2_mean: ", round(data["dM2_mean"]; digits = 4))

    println()

    println(" total_initial: ", data["total_initial"])
    println(" total_final: ", data["total_final"])
    println(" total_min: ", data["total_min"])
    println(" total_max: ", data["total_max"])
    println(" total_mean: ", round(data["total_mean"]; digits = 4))

    println()

    println(" difference_initial: ", data["difference_initial"])
    println(" difference_final: ", data["difference_final"])
    println(" difference_min: ", data["difference_min"])
    println(" difference_max: ", data["difference_max"])
    println(" difference_mean: ", round(data["difference_mean"]; digits = 4))

    println()

    println(" ratio_initial: ", data["ratio_initial"])
    println(" ratio_final: ", data["ratio_final"])
    println(" ratio_min: ", data["ratio_min"])
    println(" ratio_max: ", data["ratio_max"])
    println(" ratio_mean: ", round(data["ratio_mean"]; digits = 4))

    println()

    println(" final_leader: ", data["final_leader"])
    println(" t_near_equal: ", data["t_near_equal"])

    println()

    println(" Коэффициенты:")
    println(" a_1 = ", data["coefficients"].a_1)
    println(" a_2 = ", data["coefficients"].a_2)
    println(" b = ", data["coefficients"].b)
    println(" c_1 = ", data["coefficients"].c_1)
    println(" c_2 = ", data["coefficients"].c_2)
    println(" d = ", data["coefficients"].d)

    println()

    println(" Файл результатов: ", path)
end

# ## График M₁(t) и M₂(t)

function plot_time_series(data, params)
    model_type = params[:model_type]

    plt = plot(
        data["t_points"],
        data["M1_values"],
        xlabel = "t",
        ylabel = "M(t)",
        label = "M₁(t)",
        title = "Динамика объёмов продаж: model_type=$model_type",
        lw = 2,
        legend = :topright,
        grid = true
    )

    plot!(
        plt,
        data["t_points"],
        data["M2_values"],
        label = "M₂(t)",
        lw = 2
    )

    return plt
end

# ## График производных dM₁/dt и dM₂/dt

function plot_derivative_series(data, params)
    model_type = params[:model_type]

    plt = plot(
        data["t_points"],
        data["dM1_values"],
        xlabel = "t",
        ylabel = "dM/dt",
        label = "dM₁/dt",
        title = "Скорость изменения объёмов продаж: model_type=$model_type",
        lw = 2,
        legend = :topright,
        grid = true
    )

    plot!(
        plt,
        data["t_points"],
        data["dM2_values"],
        label = "dM₂/dt",
        lw = 2
    )

    return plt
end

# ## Фазовая траектория M₂(M₁)

function plot_phase_M1_M2(data, params)
    model_type = params[:model_type]

    plt = plot(
        data["M1_values"],
        data["M2_values"],
        xlabel = "M₁",
        ylabel = "M₂",
        label = "M₂(M₁)",
        title = "Фазовая траектория M₂(M₁): model_type=$model_type",
        lw = 2,
        legend = :topright,
        grid = true
    )

    return plt
end

# ## График разности M₁(t) - M₂(t)

function plot_difference_series(data, params)
    model_type = params[:model_type]

    plt = plot(
        data["t_points"],
        data["difference_values"],
        xlabel = "t",
        ylabel = "M₁(t) - M₂(t)",
        label = "M₁ - M₂",
        title = "Разность объёмов продаж: model_type=$model_type",
        lw = 2,
        legend = :topright,
        grid = true
    )

    hline!(
        plt,
        [0.0],
        label = "Равенство объёмов",
        lw = 2,
        linestyle = :dash
    )

    return plt
end

# ## Запуск базовых экспериментов

base_results = Dict()

for params in base_experiments
    model_type = params[:model_type]

    println("\n" * "="^70)
    println("БАЗОВЫЙ ЭКСПЕРИМЕНТ: model_type=$model_type")
    println("="^70)

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

    plt_phase = plot_phase_M1_M2(data, params)
    savefig(plt_phase, plotsdir(script_name, "single_experiment_$(model_type)_phase_M1_M2.png"))

    plt_difference = plot_difference_series(data, params)
    savefig(plt_difference, plotsdir(script_name, "single_experiment_$(model_type)_difference.png"))

    df_base = DataFrame(
        t = data["t_points"],

        M1 = data["M1_values"],
        M2 = data["M2_values"],

        dM1 = data["dM1_values"],
        dM2 = data["dM2_values"],

        total = data["total_values"],
        difference = data["difference_values"],
        ratio = data["ratio_values"],

        model_type = fill(string(model_type), length(data["t_points"])),

        p_cr = fill(params[:p_cr], length(data["t_points"])),
        N = fill(params[:N], length(data["t_points"])),
        q = fill(params[:q], length(data["t_points"])),

        tau_1 = fill(params[:tau_1], length(data["t_points"])),
        tau_2 = fill(params[:tau_2], length(data["t_points"])),

        p_tilde_1 = fill(params[:p_tilde_1], length(data["t_points"])),
        p_tilde_2 = fill(params[:p_tilde_2], length(data["t_points"])),

        d = fill(params[:d], length(data["t_points"]))
    )

    CSV.write(datadir(script_name, "table_single_$(model_type).csv"), df_base)
end

# ## Параметрическое сканирование

# В сканировании меняются:
# - p_tilde_1;
# - p_tilde_2;
# - d для второй модели.
#
# tau_1 и tau_2 оставлены базовыми, чтобы число экспериментов
# не становилось чрезмерно большим.

# ## Сетка параметров для первой модели

param_grid_case1 = Dict(
    :p_cr => [26.0],
    :N => [33.0],
    :q => [1.0],

    :tau_1 => [25.0],
    :tau_2 => [14.0],

    :p_tilde_1 => [5.0, 5.5, 6.0],
    :p_tilde_2 => [10.0, 11.0, 12.0],

    :d => [0.00033],

    :M1_0 => [3.3],
    :M2_0 => [2.2],

    :tspan => [(0.0, 20.0)],
    :nt => [500],

    :model_type => [:case1],

    :solver => [Tsit5()],
    :experiment_name => ["parametric_scan_case1"]
)

# ## Сетка параметров для второй модели

param_grid_case2 = Dict(
    :p_cr => [26.0],
    :N => [33.0],
    :q => [1.0],

    :tau_1 => [25.0],
    :tau_2 => [14.0],

    :p_tilde_1 => [5.0, 5.5, 6.0],
    :p_tilde_2 => [10.0, 11.0, 12.0],

    :d => [0.00010, 0.00033, 0.00066],

    :M1_0 => [3.3],
    :M2_0 => [2.2],

    :tspan => [(0.0, 20.0)],
    :nt => [500],

    :model_type => [:case2],

    :solver => [Tsit5()],
    :experiment_name => ["parametric_scan_case2"]
)

all_params = vcat(
    dict_list(param_grid_case1),
    dict_list(param_grid_case2)
)

println("\n" * "="^70)
println("ПАРАМЕТРИЧЕСКОЕ СКАНИРОВАНИЕ")
println("Всего комбинаций параметров: ", length(all_params))
println("="^70)

# ## Запуск всех экспериментов

all_results = []
all_dfs = []

for (i, params) in enumerate(all_params)
    println(
        "Прогресс: $i/$(length(all_params)) | ",
        "model_type=$(params[:model_type]) | ",
        "p_tilde_1=$(params[:p_tilde_1]) | ",
        "p_tilde_2=$(params[:p_tilde_2]) | ",
        "d=$(params[:d])"
    )

    data, path = produce_or_load(
        datadir(script_name, "parametric_scan"),
        params,
        run_single_experiment;
        prefix = "scan",
        tag = false,
        verbose = false
    )

    coeffs = data["coefficients"]

    result_summary = (
        model_type = string(params[:model_type]),

        p_cr = params[:p_cr],
        N = params[:N],
        q = params[:q],

        tau_1 = params[:tau_1],
        tau_2 = params[:tau_2],

        p_tilde_1 = params[:p_tilde_1],
        p_tilde_2 = params[:p_tilde_2],

        d = params[:d],

        a_1 = coeffs.a_1,
        a_2 = coeffs.a_2,
        b = coeffs.b,
        c_1 = coeffs.c_1,
        c_2 = coeffs.c_2,

        M1_0 = params[:M1_0],
        M2_0 = params[:M2_0],

        t_start = params[:tspan][1],
        t_end = params[:tspan][2],
        nt = params[:nt],

        M1_initial = data["M1_initial"],
        M1_final = data["M1_final"],
        M1_min = data["M1_min"],
        M1_max = data["M1_max"],
        M1_mean = data["M1_mean"],
        M1_growth_abs = data["M1_growth_abs"],
        M1_growth_rel = data["M1_growth_rel"],
        t_at_M1_max = data["t_at_M1_max"],

        M2_initial = data["M2_initial"],
        M2_final = data["M2_final"],
        M2_min = data["M2_min"],
        M2_max = data["M2_max"],
        M2_mean = data["M2_mean"],
        M2_growth_abs = data["M2_growth_abs"],
        M2_growth_rel = data["M2_growth_rel"],
        t_at_M2_max = data["t_at_M2_max"],

        dM1_final = data["dM1_final"],
        dM1_min = data["dM1_min"],
        dM1_max = data["dM1_max"],
        dM1_mean = data["dM1_mean"],

        dM2_final = data["dM2_final"],
        dM2_min = data["dM2_min"],
        dM2_max = data["dM2_max"],
        dM2_mean = data["dM2_mean"],

        total_final = data["total_final"],
        total_min = data["total_min"],
        total_max = data["total_max"],
        total_mean = data["total_mean"],

        difference_final = data["difference_final"],
        difference_min = data["difference_min"],
        difference_max = data["difference_max"],
        difference_mean = data["difference_mean"],

        ratio_final = data["ratio_final"],
        ratio_min = data["ratio_min"],
        ratio_max = data["ratio_max"],
        ratio_mean = data["ratio_mean"],

        final_leader = data["final_leader"],
        t_near_equal = data["t_near_equal"],

        filepath = path
    )

    push!(all_results, result_summary)

    df = DataFrame(
        t = data["t_points"],

        M1 = data["M1_values"],
        M2 = data["M2_values"],

        dM1 = data["dM1_values"],
        dM2 = data["dM2_values"],

        total = data["total_values"],
        difference = data["difference_values"],
        ratio = data["ratio_values"],

        model_type = fill(string(params[:model_type]), length(data["t_points"])),

        p_cr = fill(params[:p_cr], length(data["t_points"])),
        N = fill(params[:N], length(data["t_points"])),
        q = fill(params[:q], length(data["t_points"])),

        tau_1 = fill(params[:tau_1], length(data["t_points"])),
        tau_2 = fill(params[:tau_2], length(data["t_points"])),

        p_tilde_1 = fill(params[:p_tilde_1], length(data["t_points"])),
        p_tilde_2 = fill(params[:p_tilde_2], length(data["t_points"])),

        d = fill(params[:d], length(data["t_points"])),

        a_1 = fill(coeffs.a_1, length(data["t_points"])),
        a_2 = fill(coeffs.a_2, length(data["t_points"])),
        b = fill(coeffs.b, length(data["t_points"])),
        c_1 = fill(coeffs.c_1, length(data["t_points"])),
        c_2 = fill(coeffs.c_2, length(data["t_points"]))
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

# ## Сравнительные графики M₁(t) для каждой модели

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

        label_text = "p1=$(params[:p_tilde_1]), p2=$(params[:p_tilde_2]), d=$(params[:d])"

        plot!(
            plt,
            data["t_points"],
            data["M1_values"],
            label = label_text,
            lw = 2,
            alpha = 0.8
        )
    end

    plot!(
        plt,
        xlabel = "t",
        ylabel = "M₁(t)",
        title = "Сканирование: траектории M₁(t), model_type=$model_type_string",
        legend = :outerright,
        grid = true
    )

    savefig(plt, plotsdir(script_name, "parametric_scan_M1_comparison_$(model_type_string).png"))
end

# ## Сравнительные графики M₂(t) для каждой модели

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

        label_text = "p1=$(params[:p_tilde_1]), p2=$(params[:p_tilde_2]), d=$(params[:d])"

        plot!(
            plt,
            data["t_points"],
            data["M2_values"],
            label = label_text,
            lw = 2,
            alpha = 0.8
        )
    end

    plot!(
        plt,
        xlabel = "t",
        ylabel = "M₂(t)",
        title = "Сканирование: траектории M₂(t), model_type=$model_type_string",
        legend = :outerright,
        grid = true
    )

    savefig(plt, plotsdir(script_name, "parametric_scan_M2_comparison_$(model_type_string).png"))
end

# ## Сравнительные графики разности M₁(t) - M₂(t)

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

        label_text = "p1=$(params[:p_tilde_1]), p2=$(params[:p_tilde_2]), d=$(params[:d])"

        plot!(
            plt,
            data["t_points"],
            data["difference_values"],
            label = label_text,
            lw = 2,
            alpha = 0.8
        )
    end

    hline!(
        plt,
        [0.0],
        label = "M₁ = M₂",
        lw = 2,
        linestyle = :dash
    )

    plot!(
        plt,
        xlabel = "t",
        ylabel = "M₁(t) - M₂(t)",
        title = "Сканирование: разность объёмов продаж, model_type=$model_type_string",
        legend = :outerright,
        grid = true
    )

    savefig(plt, plotsdir(script_name, "parametric_scan_difference_comparison_$(model_type_string).png"))
end

# ## Сравнительные фазовые траектории M₂(M₁)

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

        label_text = "p1=$(params[:p_tilde_1]), p2=$(params[:p_tilde_2]), d=$(params[:d])"

        plot!(
            plt,
            data["M1_values"],
            data["M2_values"],
            label = label_text,
            lw = 2,
            alpha = 0.8
        )
    end

    plot!(
        plt,
        xlabel = "M₁",
        ylabel = "M₂",
        title = "Сканирование: фазовые траектории M₂(M₁), model_type=$model_type_string",
        legend = :outerright,
        grid = true
    )

    savefig(plt, plotsdir(script_name, "parametric_scan_phase_M1_M2_$(model_type_string).png"))
end

# ## График зависимости M1_final от p_tilde_1

p_M1_final_p1 = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_M1_final_p1,
            sub.p_tilde_1,
            sub.M1_final,
            seriestype = :scatter,
            label = "$model_type_string: M1_final"
        )
    end
end

plot!(
    p_M1_final_p1,
    xlabel = "p̃₁",
    ylabel = "M1_final",
    title = "Зависимость итогового объёма M₁ от p̃₁",
    legend = :bottomright,
    grid = true
)

savefig(p_M1_final_p1, plotsdir(script_name, "M1_final_vs_p_tilde_1.png"))

# ## График зависимости M2_final от p_tilde_2

p_M2_final_p2 = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_M2_final_p2,
            sub.p_tilde_2,
            sub.M2_final,
            seriestype = :scatter,
            label = "$model_type_string: M2_final"
        )
    end
end

plot!(
    p_M2_final_p2,
    xlabel = "p̃₂",
    ylabel = "M2_final",
    title = "Зависимость итогового объёма M₂ от p̃₂",
    legend = :bottomright,
    grid = true
)

savefig(p_M2_final_p2, plotsdir(script_name, "M2_final_vs_p_tilde_2.png"))

# ## График зависимости difference_final от p_tilde_1

p_difference_final_p1 = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_difference_final_p1,
            sub.p_tilde_1,
            sub.difference_final,
            seriestype = :scatter,
            label = "$model_type_string: M1_final - M2_final"
        )
    end
end

hline!(
    p_difference_final_p1,
    [0.0],
    label = "Равенство итоговых объёмов",
    lw = 2,
    linestyle = :dash
)

plot!(
    p_difference_final_p1,
    xlabel = "p̃₁",
    ylabel = "M1_final - M2_final",
    title = "Влияние p̃₁ на итоговую разность объёмов продаж",
    legend = :bottomright,
    grid = true
)

savefig(p_difference_final_p1, plotsdir(script_name, "difference_final_vs_p_tilde_1.png"))

# ## График зависимости difference_final от p_tilde_2

p_difference_final_p2 = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_difference_final_p2,
            sub.p_tilde_2,
            sub.difference_final,
            seriestype = :scatter,
            label = "$model_type_string: M1_final - M2_final"
        )
    end
end

hline!(
    p_difference_final_p2,
    [0.0],
    label = "Равенство итоговых объёмов",
    lw = 2,
    linestyle = :dash
)

plot!(
    p_difference_final_p2,
    xlabel = "p̃₂",
    ylabel = "M1_final - M2_final",
    title = "Влияние p̃₂ на итоговую разность объёмов продаж",
    legend = :bottomright,
    grid = true
)

savefig(p_difference_final_p2, plotsdir(script_name, "difference_final_vs_p_tilde_2.png"))

# ## График зависимости difference_final от d

p_difference_final_d = plot(size = (900, 520), dpi = 150)

for model_type_string in model_types
    sub = results_df[results_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_difference_final_d,
            sub.d,
            sub.difference_final,
            seriestype = :scatter,
            label = "$model_type_string: M1_final - M2_final"
        )
    end
end

hline!(
    p_difference_final_d,
    [0.0],
    label = "Равенство итоговых объёмов",
    lw = 2,
    linestyle = :dash
)

plot!(
    p_difference_final_d,
    xlabel = "d",
    ylabel = "M1_final - M2_final",
    title = "Влияние социально-психологического фактора d",
    legend = :bottomright,
    grid = true
)

savefig(p_difference_final_d, plotsdir(script_name, "difference_final_vs_d.png"))

# ## Бенчмаркинг

println("\n" * "="^70)
println("БЕНЧМАРКИНГ ДЛЯ РАЗНЫХ ПАРАМЕТРОВ")
println("="^70)

benchmark_results = []

for params in all_params
    function benchmark_run()
        M1_0 = params[:M1_0]
        M2_0 = params[:M2_0]

        u0 = [M1_0, M2_0]

        p = calculate_coefficients(params)
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
        "p_tilde_1=$(params[:p_tilde_1]), ",
        "p_tilde_2=$(params[:p_tilde_2]), ",
        "d=$(params[:d]):"
    )

    bmark = @benchmark $benchmark_run() samples = 80 evals = 1
    tsec = median(bmark).time / 1e9

    println(" Медианное время: ", round(tsec; digits = 6), " сек")

    push!(
        benchmark_results,
        (
            model_type = string(params[:model_type]),

            p_tilde_1 = params[:p_tilde_1],
            p_tilde_2 = params[:p_tilde_2],
            d = params[:d],

            time = tsec
        )
    )
end

bench_df = DataFrame(benchmark_results)

CSV.write(datadir(script_name, "benchmark_results.csv"), bench_df)

# ## График времени вычисления от p_tilde_1

p_time_p1 = plot(size = (900, 520), dpi = 150)

for model_type_string in unique(bench_df.model_type)
    sub = bench_df[bench_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_time_p1,
            sub.p_tilde_1,
            sub.time,
            seriestype = :scatter,
            label = "$model_type_string: время от p̃₁"
        )
    end
end

plot!(
    p_time_p1,
    xlabel = "p̃₁",
    ylabel = "Время вычисления, сек",
    title = "Зависимость времени решения ODE от p̃₁",
    legend = :topright,
    grid = true
)

savefig(p_time_p1, plotsdir(script_name, "computation_time_vs_p_tilde_1.png"))

# ## График времени вычисления от p_tilde_2

p_time_p2 = plot(size = (900, 520), dpi = 150)

for model_type_string in unique(bench_df.model_type)
    sub = bench_df[bench_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_time_p2,
            sub.p_tilde_2,
            sub.time,
            seriestype = :scatter,
            label = "$model_type_string: время от p̃₂"
        )
    end
end

plot!(
    p_time_p2,
    xlabel = "p̃₂",
    ylabel = "Время вычисления, сек",
    title = "Зависимость времени решения ODE от p̃₂",
    legend = :topright,
    grid = true
)

savefig(p_time_p2, plotsdir(script_name, "computation_time_vs_p_tilde_2.png"))

# ## График времени вычисления от d

p_time_d = plot(size = (900, 520), dpi = 150)

for model_type_string in unique(bench_df.model_type)
    sub = bench_df[bench_df.model_type .== model_type_string, :]

    if nrow(sub) > 0
        plot!(
            p_time_d,
            sub.d,
            sub.time,
            seriestype = :scatter,
            label = "$model_type_string: время от d"
        )
    end
end

plot!(
    p_time_d,
    xlabel = "d",
    ylabel = "Время вычисления, сек",
    title = "Зависимость времени решения ODE от d",
    legend = :topright,
    grid = true
)

savefig(p_time_d, plotsdir(script_name, "computation_time_vs_d.png"))

# ## Сохранение всех результатов

@save datadir(script_name, "all_results.jld2") base_experiments base_results param_grid_case1 param_grid_case2 all_params results_df bench_df full_timeseries_df

@save datadir(script_name, "all_plots.jld2") p_M1_final_p1 p_M2_final_p2 p_difference_final_p1 p_difference_final_p2 p_difference_final_d p_time_p1 p_time_p2 p_time_d

println("\n" * "="^70)
println("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
println("="^70)

println("\nРезультаты сохранены в:")
println(" • data/$(script_name)/single/ - базовые эксперименты")
println(" • data/$(script_name)/parametric_scan/ - параметрическое сканирование")
println(" • data/$(script_name)/table_single_case1.csv - таблица для первой модели")
println(" • data/$(script_name)/table_single_case2.csv - таблица для второй модели")
println(" • data/$(script_name)/results_summary.csv - сводная таблица")
println(" • data/$(script_name)/timeseries_full.csv - полные временные ряды")
println(" • data/$(script_name)/benchmark_results.csv - результаты бенчмаркинга")
println(" • data/$(script_name)/all_results.jld2 - сводные данные")
println(" • data/$(script_name)/all_plots.jld2 - объекты графиков")
println(" • plots/$(script_name)/ - все графики")