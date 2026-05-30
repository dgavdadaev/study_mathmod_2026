# # Численное решение модели конкуренции двух фирм
# **Цель:** решить два варианта системы дифференциальных уравнений,
# описывающей динамику объёмов продаж двух фирм,
# построить графики M₁(t) и M₂(t),
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

# ## Общие параметры модели

p_cr = 26.0
N = 33.0
q = 1.0

tau_1 = 25.0
tau_2 = 14.0

p_tilde_1 = 5.5
p_tilde_2 = 11.0

d = 0.00033

# ## Начальные условия

M1_0 = 3.3
M2_0 = 2.2

u0 = [M1_0, M2_0]

# ## Временной интервал
# Используется нормированное время:
# t = c₁Θ

t0 = 0.0
tmax = 20.0
tspan = (t0, tmax)

nt = 500

# ## Вычисление коэффициентов модели

a_1 = p_cr / (tau_1^2 * p_tilde_1^2 * N * q)

a_2 = p_cr / (tau_2^2 * p_tilde_2^2 * N * q)

b = p_cr / (
    tau_1^2 * p_tilde_1^2 *
    tau_2^2 * p_tilde_2^2 *
    N * q
)

c_1 = (p_cr - p_tilde_1) / (tau_1 * p_tilde_1)

c_2 = (p_cr - p_tilde_2) / (tau_2 * p_tilde_2)

# ## Набор параметров для передачи в модели

p = (
    a_1 = a_1,
    a_2 = a_2,
    b = b,
    c_1 = c_1,
    c_2 = c_2,
    d = d
)

# ## Первая модель
# Рассматривается конкуренция двух фирм только рыночными методами.
#
# Система уравнений:
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

# ## Вторая модель
# Помимо экономического фактора учитывается социально-психологическое
# влияние, которое меняет коэффициент взаимодействия фирм в первом уравнении.
#
# Система уравнений:
#
# dM₁/dt = M₁ - ((b / c₁) + d)M₁M₂ - (a₁ / c₁)M₁²
#
# dM₂/dt = (c₂ / c₁)M₂ - (b / c₁)M₁M₂ - (a₂ / c₁)M₂²

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

# ## Утилита: один прогон модели

function run_case(case_name; model!, u0, tspan, p, nt = 500, image_name = case_name)
    # --- Сетка по времени ---
    tgrid = collect(LinRange(tspan[1], tspan[2], nt))

    # --- Решение системы ОДУ ---
    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = tgrid)

    # --- Таблица с результатами ---
    df = DataFrame(
        t = sol.t,
        M1 = [u[1] for u in sol.u],
        M2 = [u[2] for u in sol.u]
    )

    # --- График M₁(t) и M₂(t) ---
    plt = plot(
        df.t,
        df.M1,
        xlabel = "t",
        ylabel = "M(t)",
        label = "M₁(t)",
        lw = 2,
        title = "Динамика объёмов продаж — $case_name",
        legend = :topright
    )

    plot!(
        plt,
        df.t,
        df.M2,
        label = "M₂(t)",
        lw = 2
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
    tspan = tspan,
    p = p,
    nt = nt,
    image_name = "01"
)

# ## Запуск второй модели

res_2 = run_case(
    "case_2";
    model! = model_case_2!,
    u0 = u0,
    tspan = tspan,
    p = p,
    nt = nt,
    image_name = "02"
)

# ## Вывод рассчитанных коэффициентов

println("Коэффициенты модели:")
println("a_1 = ", a_1)
println("a_2 = ", a_2)
println("b   = ", b)
println("c_1 = ", c_1)
println("c_2 = ", c_2)
println("d   = ", d)

println()

# ## Вывод первых строк таблиц

println("Первые 5 строк таблицы результатов для первой модели:")
println(first(res_1.df, 5))

println()

println("Первые 5 строк таблицы результатов для второй модели:")
println(first(res_2.df, 5))

# ## Показать график в интерактивной среде

res_1.plt