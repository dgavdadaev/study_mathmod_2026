# # Модель SIR: Эпидемиологическое моделирование
# 
# **Цель работы:** Исследовать динамику распространения инфекционного заболевания
# с помощью классической компартментальной модели SIR.
# 
# ## Теоретическое введение
# 
# Модель SIR (Susceptible-Infectious-Recovered) делит популяцию на три группы:
# - **S (Susceptible)** - восприимчивые к инфекции
# - **I (Infectious)** - инфицированные и заразные
# - **R (Recovered)** - выздоровевшие (с иммунитетом)
# 
# ### Система дифференциальных уравнений
# 
# $$
# \begin{cases}
# \frac{dS}{dt} = -\beta \cdot c \cdot \frac{I}{N} \cdot S \\[1em]
# \frac{dI}{dt} = \beta \cdot c \cdot \frac{I}{N} \cdot S - \gamma I \\[1em]
# \frac{dR}{dt} = \gamma I
# \end{cases}
# $$
# 
# где:
# - $\beta$ - вероятность передачи инфекции при контакте
# - $c$ - среднее число контактов в единицу времени
# - $\gamma$ - скорость выздоровления ($1/\gamma$ - средняя продолжительность болезни)
# - $N = S + I + R$ - общая численность популяции
# 
# ### Ключевые показатели
# 
# **Базовое репродуктивное число:**
# $$R_0 = \frac{c \cdot \beta}{\gamma}$$
# 
# **Эффективное репродуктивное число:**
# $$R_e(t) = R_0 \cdot \frac{S(t)}{N}$$

# ## Инициализация проекта и загрузка пакетов

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

# ## Настройка директорий для сохранения результатов
# 
# Создаем директории для графиков и данных, используя имя текущего скрипта.

script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

# ## Определение функции модели SIR
# 
# Функция описывает правые части системы дифференциальных уравнений.
# Используем макрос `@inbounds` для повышения производительности.

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

# ## Задание параметров модели
# 
# ### Параметры интегрирования
# - `δt = 0.1` - шаг интегрирования
# - `tmax = 40.0` - максимальное время моделирования (дни)
# - `tspan = (0.0, 40.0)` - временной интервал

δt = 0.1
tmax = 40.0
tspan = (0.0, tmax)

# ### Начальные условия
# Популяция состоит из 1000 человек:
# - $S_0 = 990$ восприимчивых (99%)
# - $I_0 = 10$ инфицированных (1%)  
# - $R_0 = 0$ выздоровевших (0%)

u0 = [990.0, 10.0, 0.0]  # S, I, R

# ### Параметры модели
# - $\beta = 0.05$ - вероятность заражения при контакте (5%)
# - $c = 10.0$ - среднее число контактов в день
# - $\gamma = 0.25$ - скорость выздоровления (средняя длительность болезни = 4 дня)

p = [0.05, 10.0, 0.25]    # β, c, γ

# ## Расчет базового репродуктивного числа
# 
# $$R_0 = \frac{c \cdot \beta}{\gamma} = \frac{10 \cdot 0.05}{0.25} = 2$$
# 
# $R_0 = 2$ означает, что один больной в полностью восприимчивой популяции
# заразит в среднем 2 человек. Это указывает на развитие эпидемии.

R0 = (p[2] * p[1]) / p[3]

# Вывод информации о модели
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

# ## Численное решение системы ОДУ
# 
# Создаем задачу Коши и решаем её.

prob_ode = ODEProblem(sir_ode!, u0, tspan, p)
sol_ode = solve(prob_ode, dt = δt)

# ## Подготовка данных для анализа
# 
# Преобразуем решение в DataFrame для удобной обработки и визуализации.

df_ode = DataFrame(Tables.table(sol_ode'))
rename!(df_ode, ["S", "I", "R"])
df_ode[!, :t] = sol_ode.t
df_ode[!, :N] = df_ode.S + df_ode.I + df_ode.R

# ## Визуализация результатов
# 
# ### 1. Динамика всех трех групп
# 
# На графике показано изменение численности каждой группы во времени.
# Видно, как восприимчивые (S) уменьшаются, инфицированные (I) проходят через пик,
# а выздоровевшие (R) накапливаются.

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

# ### 2. Динамика числа инфицированных с выделением пика
# 
# Этот график фокусируется на группе I. Пик эпидемии - критический момент,
# когда нагрузка на систему здравоохранения максимальна.

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

# ### 3. Экспоненциальный рост в логарифмическом масштабе
# 
# На начальном этапе эпидемии рост числа зараженных должен быть экспоненциальным,
# что на логарифмическом графике выглядит как прямая линия.
# Отклонение от прямой указывает на замедление роста из-за уменьшения
# числа восприимчивых.

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

# ### 4. Доли популяции в процентах
# 
# Этот график показывает относительный вклад каждой группы.
# Добавлена линия порога коллективного иммунитета - доли населения,
# которая должна иметь иммунитет для предотвращения эпидемии.

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

# ### 5. Фазовый портрет (I vs S)
# 
# Фазовый портрет показывает соотношение между восприимчивыми и инфицированными.
# Траектория движется по часовой стрелке от начальной точки к конечной.
# Стрелки указывают направление движения во времени.

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

# ### 6. Динамика эффективного репродуктивного числа
# 
# $R_e(t)$ показывает, сколько человек заражает один больной в текущих условиях.
# Когда $R_e$ падает ниже 1, эпидемия идет на спад.

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

# ### 7. Компактная панель всех графиков
# 
# Собираем все графики в одну панель для удобного сравнения.

plt7 = plot(plt1, plt2, plt3, plt4, plt5, plt6, 
    layout=(3, 2), 
    size=(1400, 1200), 
    titlefontsize=10)

# ## Сохранение графиков
# 
# Все созданные графики сохраняются в директорию `plots/script_name/`.

savefig(plt1, plotsdir(script_name, "sir_main.png"))
savefig(plt2, plotsdir(script_name, "sir_infected.png"))
savefig(plt3, plotsdir(script_name, "sir_log_scale.png"))
savefig(plt4, plotsdir(script_name, "sir_percentages.png"))
savefig(plt5, plotsdir(script_name, "sir_phase_portrait.png"))
savefig(plt6, plotsdir(script_name, "sir_effective_R.png"))
savefig(plt7, plotsdir(script_name, "sir_panel.png"))

println("\nГрафики сохранены в: ", plotsdir(script_name))

# ## Оценка производительности
# 
# Бенчмарк показывает время выполнения решения задачи.

println("\nБенчмарк решения:")
@benchmark solve(prob_ode, dt = $δt)

# ## Анализ результатов
# 
# Ключевые показатели эпидемии:

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
