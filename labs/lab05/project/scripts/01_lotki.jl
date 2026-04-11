# # Модель хищник-жертва: численное решение и визуализация
# **Цель:** решить систему ОДУ Лотки–Вольтерры,
# построить графики изменения популяций во времени,
# построить фазовый портрет,
# сохранить изображения и таблицу с результатами.

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

# ## Начальные условия и временной интервал
x0 = 8.0
y0 = 11.0

u0 = [x0, y0]

t0 = 0.0
tmax = 150.0
tspan = (t0, tmax)

# ## Параметры модели
a = 0.25
b = 0.025
c = 0.45
d = 0.045

p = (a = a, b = b, c = c, d = d)

# ## Определение модели
# x(t) — численность жертв
# y(t) — численность хищников
#
# dx/dt = -a*x + b*x*y
# dy/dt =  c*y - d*x*y
function predator_prey!(dy, y, p, t)
    dy[1] = -p.a * y[1] + p.b * y[1] * y[2]
    dy[2] =  p.c * y[2] - p.d * y[1] * y[2]
end

# ## Утилита: один прогон (решение, таблица, графики, сохранение)
function run_case(case_name; model!, u0, tspan, p, nt=1000)
    # --- Сетка по времени ---
    tgrid = collect(LinRange(tspan[1], tspan[2], nt))

    # --- Решение ОДУ ---
    prob = ODEProblem(model!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=tgrid)

    # --- Таблица с результатами ---
    df = DataFrame(
        t = sol.t,
        x = first.(sol.u),
        y = last.(sol.u)
    )

    # --- График x(t) и y(t) ---
    plt_time = plot(
        sol,
        xlabel = "t",
        ylabel = "Численность",
        label = ["x(t) — жертвы" "y(t) — хищники"],
        lw = 2,
        title = "Динамика популяций — $case_name",
        legend = :topright
    )

    # --- Фазовый портрет ---
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

    # --- Сохранение графиков ---
    savefig(plt_time, plotsdir(script_name, "$(case_name)_time.png"))
    savefig(plt_phase, plotsdir(script_name, "$(case_name)_phase.png"))

    # --- Сохранение таблицы ---
    CSV.write(datadir(script_name, "table_$(case_name).csv"), df)

    # --- Сохранение данных в JLD2 ---
    @save datadir(script_name, "data_$(case_name).jld2") df p u0 tspan nt

    return (sol = sol, df = df, plt_time = plt_time, plt_phase = plt_phase)
end

# ## Запуск модели
res = run_case(
    "predator_prey";
    model! = predator_prey!,
    u0 = u0,
    tspan = tspan,
    p = p,
    nt = 1000
)

# ## Вывод первых строк таблицы
println("Первые 5 строк таблицы результатов:")
println(first(res.df, 5))

# ## Показать график в интерактивной среде
res.plt_time