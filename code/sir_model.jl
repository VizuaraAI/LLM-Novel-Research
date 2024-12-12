
using DifferentialEquations, Plots

function sir_ode!(du, u, p, t)
    S, I, R = u
    beta, gamma = p
    du[1] = -beta*S*I
    du[2] = beta*S*I - gamma*I
    du[3] = gamma*I
end

S0 = 0.99
I0 = 0.01
R0 = 0.0
u0 = [S0, I0, R0]

beta = 0.5
gamma = 0.1
p = [beta, gamma]

tspan = (0.0, 60.0)

prob = ODEProblem(sir_ode!, u0, tspan, p)

sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

plot(sol, xlabel="Time", ylabel="Proportion", title="SIR Model", lw=2)
savefig("sir_model_plot.png")
