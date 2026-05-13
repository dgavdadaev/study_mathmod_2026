# # Численное решение трёх задач Коши и визуализация
# **Цель:** решить три варианта дифференциального уравнения,
# построить графики n(t),
# сохранить изображения и таблицы с результатами.

# ## Инициализация проекта и загрузка пакетов
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

# ## Общие начальные условия
N = 609.0
n0 = 4.0
u0 = [n0]

# ## Первая модель
# Уравнение:
# dn/dt = (a + b*n)*(N - n)

a1 = 0.54
b1 = 0.00016

t0_1 = 0.0
tmax_1 = 10.0
tspan_1 = (t0_1, tmax_1)

p1 = (a = a1, b = b1, N = N)

function model_case_1!(dy, y, p, t)
    dy[1] = (p.a + p.b * y[1]) * (p.N - y[1])
end

# ## Вторая модель
# Уравнение:
# dn/dt = (a + b*n)*(N - n)

a2 = 0.000021
b2 = 0.38

t0_2 = 0.0
tmax_2 = 0.04
tspan_2 = (t0_2, tmax_2)

p2 = (a = a2, b = b2, N = N)

function model_case_2!(dy, y, p, t)
    dy[1] = (p.a + p.b * y[1]) * (p.N - y[1])
end

# ## Третья модель
# Уравнение:
# dn/dt = (a*cos(t) + b*cos(2*t)*n)*(N - n)

a3 = 0.2
b3 = 0.2

t0_3 = 0.0
tmax_3 = 0.15
tspan_3 = (t0_3, tmax_3)

p3 = (a = a3, b = b3, N = N)

function model_case_3!(dy, y, p, t)
    dy[1] = (p.a * cos(t) + p.b * cos(2 * t) * y[1]) * (p.N - y[1])
end

# ## Утилита: один прогон модели
function run_case(case_name; model!, u0, tspan, p, nt = 500, image_name = case_name)
    # --- Сетка по времени ---
    tgrid = collect(LinRange(tspan[1], tspan[2], nt))

    # --- Решение ОДУ ---
    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = tgrid)

    # --- Таблица с результатами ---
    df = DataFrame(
        t = sol.t,
        n = [u[1] for u in sol.u]
    )

    # --- График n(t) ---
    plt = plot(
        sol,
        xlabel = "t",
        ylabel = "n(t)",
        label = "n(t)",
        lw = 2,
        title = "Численное решение — $case_name",
        legend = :topright
    )

    # --- Сохранение графика ---
    savefig(plt, plotsdir(script_name, "$(image_name).png"))

    # --- Сохранение таблицы ---
    CSV.write(datadir(script_name, "table_$(case_name).csv"), df)

    # --- Сохранение данных в JLD2 ---
    @save datadir(script_name, "data_$(case_name).jld2") df p u0 tspan nt

    return (
        sol = sol,
        df = df,
        plt = plt
    )
end

# ## Запуск первой модели
res_1 = run_case(
    "case_1";
    model! = model_case_1!,
    u0 = u0,
    tspan = tspan_1,
    p = p1,
    nt = 500,
    image_name = "04"
)

# ## Запуск второй модели
res_2 = run_case(
    "case_2";
    model! = model_case_2!,
    u0 = u0,
    tspan = tspan_2,
    p = p2,
    nt = 500,
    image_name = "05"
)

# ## Запуск третьей модели
res_3 = run_case(
    "case_3";
    model! = model_case_3!,
    u0 = u0,
    tspan = tspan_3,
    p = p3,
    nt = 500,
    image_name = "06"
)

# ## Вывод первых строк таблиц
println("Первые 5 строк таблицы результатов для первой модели:")
println(first(res_1.df, 5))

println()
println("Первые 5 строк таблицы результатов для второй модели:")
println(first(res_2.df, 5))

println()
println("Первые 5 строк таблицы результатов для третьей модели:")
println(first(res_3.df, 5))

# ## Показать график в интерактивной среде
res_1.plt