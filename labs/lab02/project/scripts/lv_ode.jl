# # Модель Лотки-Вольтерры: Хищник-жертва
# 
# **Цель работы:** Исследовать динамику взаимодействия двух видов - хищников и жертв -
# с помощью классической модели Лотки-Вольтерры.
# 
# ## Теоретическое введение
# 
# Модель Лотки-Вольтерры (также известная как модель "хищник-жертва") была независимо
# предложена Альфредом Лоткой (1925) и Витторио Вольтеррой (1926). Она описывает
# циклические колебания численности двух взаимодействующих популяций.
# 
# ### Система дифференциальных уравнений
# 
# $$
# \begin{cases}
# \frac{dx}{dt} = \alpha x - \beta x y \\[1em]
# \frac{dy}{dt} = \delta x y - \gamma y
# \end{cases}
# $$
# 
# где:
# - $x$ - численность популяции жертв (например, зайцы)
# - $y$ - численность популяции хищников (например, лисы)
# - $\alpha$ - коэффициент естественного прироста жертв (рождаемость)
# - $\beta$ - коэффициент поедания жертв хищниками (вероятность встречи)
# - $\delta$ - коэффициент конверсии (эффективность превращения съеденных жертв в потомство хищников)
# - $\gamma$ - коэффициент естественной смертности хищников
# 
# ### Ключевые характеристики
# 
# **Стационарные точки (положения равновесия):**
# 
# $$
# x^* = \frac{\gamma}{\delta}, \quad y^* = \frac{\alpha}{\beta}
# $$
# 
# **Период малых колебаний вблизи равновесия:**
# 
# $$
# T \approx \frac{2\pi}{\sqrt{\alpha \gamma}}
# $$
# 
# **Инвариант (сохраняющаяся величина):**
# 
# $$
# V(x, y) = \delta x - \gamma \ln x + \beta y - \alpha \ln y = \text{const}
# $$

# ## Инициализация проекта и загрузка пакетов

using DrWatson
@quickactivate "project"

using DifferentialEquations
using DataFrames
using StatsPlots
using LaTeXStrings
using Plots
using Statistics
using FFTW

# ## Настройка директорий для сохранения результатов
# 
# Создаем директории для графиков и данных, используя имя текущего скрипта.

script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

# ## Определение функции модели Лотки-Вольтерры
# 
# Функция описывает правые части системы дифференциальных уравнений.
# Используем макрос `@inbounds` для повышения производительности.

function lotka_volterra!(du, u, p, t)
    x, y = u  # x - жертвы, y - хищники
    α, β, δ, γ = p  # параметры модели
    
    @inbounds begin
        du[1] = α*x - β*x*y  # уравнение для жертв: dx/dt = αx - βxy
        du[2] = δ*x*y - γ*y  # уравнение для хищников: dy/dt = δxy - γy
    end
    nothing
end

# ## Задание параметров модели
# 
# ### Параметры интегрирования
# - `dt_lv = 0.01` - шаг интегрирования
# - `tspan_lv = (0.0, 200.0)` - временной интервал моделирования

dt_lv = 0.01
tspan_lv = (0.0, 200.0)

# ### Начальные условия
# - $x_0 = 40.0$ - начальная популяция жертв
# - $y_0 = 9.0$ - начальная популяция хищников

u0_lv = [40.0, 9.0]

# ### Параметры модели (классические значения из литературы)
# - $\alpha = 0.1$ - скорость размножения жертв
# - $\beta = 0.02$ - скорость поедания жертв хищниками
# - $\delta = 0.01$ - коэффициент конверсии пищи в потомство хищников
# - $\gamma = 0.3$ - смертность хищников

p_lv = [0.1,   # α
        0.02,  # β
        0.01,  # δ
        0.3]   # γ

# ## Численное решение системы ОДУ
# 
# Создаем задачу Коши и решаем её методом Цытовича 5-го порядка
# с высокой точностью.

prob_lv = ODEProblem(lotka_volterra!, u0_lv, tspan_lv, p_lv)
sol_lv = solve(prob_lv, 
    Tsit5(),          # Метод 5-го порядка
    reltol=1e-8,      # Относительная точность
    abstol=1e-10,     # Абсолютная точность
    saveat=0.1)       # Сохраняем каждые 0.1 единицы времени

# ## Подготовка данных для анализа
# 
# Преобразуем решение в DataFrame для удобной обработки и визуализации.

df_lv = DataFrame()
df_lv[!, :t] = sol_lv.t
df_lv[!, :prey] = [u[1] for u in sol_lv.u]      # жертвы
df_lv[!, :predator] = [u[2] for u in sol_lv.u]  # хищники

# ### Расчет производных для анализа скоростей изменения

df_lv[!, :dprey_dt] = p_lv[1] .* df_lv.prey .- p_lv[2] .* df_lv.prey .* df_lv.predator
df_lv[!, :dpredator_dt] = p_lv[3] .* df_lv.prey .* df_lv.predator .- p_lv[4] .* df_lv.predator

# ## Анализ модели
# 
# ### Вывод информации о параметрах

println("="^60)
println("Модель Лотки-Вольтерры (хищник-жертва)")
println("="^60)
println("\nПараметры модели:")
println("α (скорость размножения жертв) = ", p_lv[1])
println("β (скорость поедания жертв) = ", p_lv[2])
println("δ (коэффициент конверсии) = ", p_lv[3])
println("γ (смертность хищников) = ", p_lv[4])
println("\nНачальные условия:")
println("Жертвы (x₀) = ", u0_lv[1])
println("Хищники (y₀) = ", u0_lv[2])

# ### Стационарные точки (положения равновесия)
# 
# $$
# x^* = \frac{\gamma}{\delta}, \quad y^* = \frac{\alpha}{\beta}
# $$

x_star = p_lv[4] / p_lv[3]  # стационарная точка для жертв
y_star = p_lv[1] / p_lv[2]  # стационарная точка для хищников

println("\nСтационарные точки (положения равновесия):")
println("x* = γ/δ = ", round(x_star, digits=3))
println("y* = α/β = ", round(y_star, digits=3))

# ## Визуализация результатов
# 
# ### 1. Динамика популяций во времени
# 
# На графике показаны колебания численности жертв (зеленый) и хищников (красный).
# Видно, что пик численности хищников отстает от пика жертв - характерная особенность
# модели Лотки-Вольтерры.

plt1 = plot(df_lv.t, [df_lv.prey df_lv.predator], 
    label=[L"Жертвы (x)" L"Хищники (y)"], 
    xlabel="Время", 
    ylabel="Популяция", 
    title="Модель Лотки-Вольтерры: Динамика популяций", 
    linewidth=2, 
    legend=:topright, 
    grid=true, 
    size=(900, 500), 
    color=[:green :red])

# Добавление стационарных уровней
hline!(plt1, [x_star], color=:green, linestyle=:dash, alpha=0.5, label="x* (равновесие жертв)")
hline!(plt1, [y_star], color=:red, linestyle=:dash, alpha=0.5, label="y* (равновесие хищников)")

# ### 2. Фазовый портрет (хищники vs жертвы)
# 
# Фазовый портрет показывает замкнутую траекторию вокруг стационарной точки.
# Стрелки указывают направление движения во времени. Замкнутость траектории
# свидетельствует о циклическом характере колебаний.

plt2 = plot(df_lv.prey, df_lv.predator, 
    label="Фазовая траектория", 
    xlabel="Популяция жертв (x)", 
    ylabel="Популяция хищников (y)", 
    title="Фазовый портрет системы", 
    color=:blue, 
    linewidth=1.5, 
    grid=true, 
    size=(800, 600), 
    legend=:topright)

# Добавление стрелок направления на фазовом портрете
step = 50  # шаг для отображения стрелок
for i in 1:step:length(df_lv.prey)-step
    plot!(plt2, [df_lv.prey[i], df_lv.prey[i+step]], [df_lv.predator[i], df_lv.predator[i+step]], 
        arrow=:closed, color=:blue, alpha=0.3, label=false)
end

# Добавление стационарной точки
scatter!(plt2, [x_star], [y_star], color=:black, markersize=8, label="Стационарная точка (x*, y*)")

# ### Изоклины (линии нулевого роста)
# 
# Изоклина для жертв (dx/dt = 0): $y = \alpha/\beta$ (горизонтальная линия)
# Изоклина для хищников (dy/dt = 0): $x = \gamma/\delta$ (вертикальная линия)

x_range = LinRange(0, maximum(df_lv.prey)*1.1, 100)
y_nulcline = p_lv[1] ./ (p_lv[2] .* x_range)  # изоклина для жертв (не используется в классическом виде)
# В классической модели изоклина жертв - горизонтальная линия y = α/β
# Изоклина хищников - вертикальная линия x = γ/δ

# Добавляем вертикальную изоклину хищников
vline!(plt2, [x_star], color=:blue, linestyle=:dash, linewidth=1.5, label="Изоклина dy/dt = 0")

# ### 3. Скорости изменения популяций
# 
# График производных показывает, как быстро меняется численность каждого вида
# в каждый момент времени.

plt3 = plot(df_lv.t, [df_lv.dprey_dt df_lv.dpredator_dt], 
    label=[L"dx/dt" L"dy/dt"], 
    xlabel="Время", 
    ylabel="Скорость изменения", 
    title="Скорости изменения популяций", 
    linewidth=1.5, 
    legend=:topright, 
    grid=true, 
    size=(900, 400), 
    color=[:green :red])

# ### 4. Относительные темпы роста (в процентах)
# 
# Относительные изменения показывают, на сколько процентов меняется популяция
# в единицу времени.

df_lv[!, :prey_pct_change] = df_lv.dprey_dt ./ df_lv.prey .* 100
df_lv[!, :predator_pct_change] = df_lv.dpredator_dt ./ df_lv.predator .* 100

plt4 = plot(df_lv.t, [df_lv.prey_pct_change df_lv.predator_pct_change], 
    label=[L"dx/dt / x (\%)" L"dy/dt / y (\%)"], 
    xlabel="Время", 
    ylabel="Относительное изменение, %", 
    title="Относительные темпы роста", 
    linewidth=1.5, 
    legend=:topright, 
    grid=true, 
    size=(900, 400), 
    color=[:green :red])

# ### 5. Спектральный анализ (быстрое преобразование Фурье)
# 
# Спектральный анализ позволяет определить основные частоты колебаний
# и соответствующие им периоды.

function compute_fft(signal, dt)
    n = length(signal)
    # Используем rfft для вещественных сигналов (возвращает только положительные частоты)
    spectrum = abs.(rfft(signal))
    # Соответствующие частоты для rfft
    freq = rfftfreq(n, 1/dt)
    return freq, spectrum
end

# Вычисление периодов колебаний
freq_prey, spectrum_prey = compute_fft(df_lv.prey .- mean(df_lv.prey), dt_lv)
freq_predator, spectrum_predator = compute_fft(df_lv.predator .- mean(df_lv.predator), dt_lv)

plt5 = plot(freq_prey, [spectrum_prey spectrum_predator], 
    label=[L"Жертвы (x)" L"Хищники (y)"], 
    xlabel="Частота", 
    ylabel="Амплитуда", 
    title="Спектральный анализ (Фурье)", 
    linewidth=1.5, 
    xscale=:log10, 
    yscale=:log10, 
    legend=:topright, 
    grid=true, 
    size=(800, 400), 
    color=[:green :red])

# Нахождение доминирующих частот
if length(spectrum_prey) > 0
    idx_prey = argmax(spectrum_prey[2:end]) + 1  # пропускаем нулевую частоту
    dominant_freq_prey = freq_prey[idx_prey]
    period_prey = 1/dominant_freq_prey
    println("\nДоминирующая частота колебаний жертв: ", round(dominant_freq_prey, digits=4), " Гц")
    println("Период колебаний жертв: ", round(period_prey, digits=2), " единиц времени")
end

# ### 6. Компактная панель всех графиков
# 
# Собираем все графики в одну панель для удобного сравнения.

plt6 = plot(layout=(3, 2), size=(1200, 900))

plot!(plt6[1], df_lv.t, df_lv.prey, label=L"x(t)", color=:green, linewidth=2, 
    title="Популяция жертв", grid=true)
plot!(plt6[2], df_lv.t, df_lv.predator, label=L"y(t)", color=:red, linewidth=2, 
    title="Популяция хищников", grid=true)
plot!(plt6[3], df_lv.prey, df_lv.predator, label=false, color=:blue, linewidth=1.5, 
    title="Фазовый портрет", xlabel=L"x", ylabel=L"y", grid=true)
scatter!(plt6[3], [x_star], [y_star], color=:black, markersize=5, label="x*, y*")
plot!(plt6[4], df_lv.t, [df_lv.dprey_dt df_lv.dpredator_dt], 
    label=[L"dx/dt" L"dy/dt"], color=[:green :red], linewidth=1.5, 
    title="Скорости изменения", grid=true, legend=:topright)
plot!(plt6[5], freq_prey, spectrum_prey, label=L"x", color=:green, linewidth=1.5, 
    title="Спектр жертв", xscale=:log10, yscale=:log10, grid=true)
plot!(plt6[6], df_lv.t, [df_lv.prey_pct_change df_lv.predator_pct_change], 
    label=[L"dx/x" L"dy/y"], color=[:green :red], linewidth=1.5, 
    title="Относительные изменения", grid=true, legend=:topright)

# ## Анализ результатов

println("\n" * "="^60)
println("Анализ результатов")
println("="^60)
println("\nОсновные статистики:")
println("Жертвы: min = ", round(minimum(df_lv.prey), digits=2), 
    ", max = ", round(maximum(df_lv.prey), digits=2), 
    ", mean = ", round(mean(df_lv.prey), digits=2))
println("Хищники: min = ", round(minimum(df_lv.predator), digits=2), 
    ", max = ", round(maximum(df_lv.predator), digits=2), 
    ", mean = ", round(mean(df_lv.predator), digits=2))

# ### Анализ фазового сдвига
# 
# В модели Лотки-Вольтерры пик численности хищников должен отставать
# от пика жертв примерно на четверть периода.

function find_first_peak(signal, time)
    for i in 2:length(signal)-1
        if signal[i] > signal[i-1] && signal[i] > signal[i+1]
            return time[i], signal[i]
        end
    end
    return NaN, NaN
end

peak_time_prey, peak_value_prey = find_first_peak(df_lv.prey, df_lv.t)
peak_time_predator, peak_value_predator = find_first_peak(df_lv.predator, df_lv.t)

if !isnan(peak_time_prey) && !isnan(peak_time_predator)
    phase_shift = peak_time_predator - peak_time_prey
    println("\nАнализ колебаний:")
    println("Первый пик жертв: время = ", round(peak_time_prey, digits=2), 
        ", значение = ", round(peak_value_prey, digits=2))
    println("Первый пик хищников: время = ", round(peak_time_predator, digits=2), 
        ", значение = ", round(peak_value_predator, digits=2))
    println("Сдвиг фаз (хищники отстают): ", round(phase_shift, digits=2))
end

# ## Сохранение графиков
# 
# Все созданные графики сохраняются в директорию `plots/script_name/`.

savefig(plt1, plotsdir(script_name, "lv_dynamics.png"))
savefig(plt2, plotsdir(script_name, "lv_phase_portrait.png"))
savefig(plt3, plotsdir(script_name, "lv_derivatives.png"))
savefig(plt4, plotsdir(script_name, "lv_relative_changes.png"))
savefig(plt5, plotsdir(script_name, "lv_spectrum.png"))
savefig(plt6, plotsdir(script_name, "lv_panel.png"))

println("\nГрафики сохранены в: ", plotsdir(script_name))

# ## Анализ чувствительности к параметрам (опционально)
# 
# Этот раздел позволяет исследовать, как изменение ключевых параметров
# влияет на поведение системы.

println("\n\n" * "="^60)
println("Анализ чувствительности")
println("="^60)

function analyze_parameter_sensitivity(param_index, values, param_name)
    println("\nАнализ чувствительности к параметру: ", param_name)
    results = []
    
    for val in values
        p_test = copy(p_lv)
        p_test[param_index] = val
        prob_test = ODEProblem(lotka_volterra!, u0_lv, tspan_lv, p_test)
        sol_test = solve(prob_test, dt = dt_lv)
        prey_end = sol_test.u[end][1]
        predator_end = sol_test.u[end][2]
        
        push!(results, (param=val, prey=prey_end, predator=predator_end))
        println("  $(param_name)=$(val): $(round(prey_end, digits=2)), $(round(predator_end, digits=2))")
    end
    return results
end

# Для выполнения анализа раскомментируйте следующие строки:
if false  # Установите в true для выполнения анализа
    println("\n1. Влияние скорости размножения жертв (α):")
    analyze_parameter_sensitivity(1, [0.05, 0.1, 0.2, 0.3], "α")
    
    println("\n2. Влияние смертности хищников (γ):")
    analyze_parameter_sensitivity(4, [0.1, 0.3, 0.5, 0.7], "γ")
end

println("\n" * "="^60)
println("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО")
println("="^60)
