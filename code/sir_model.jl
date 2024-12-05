
using DifferentialEquations

# Define the SIR parameters
β = 0.1
γ = 0.05

# Define the SIR model
function sir_model!(du, u, p, t)
    S, I, R = u
    du[1] = -β*S*I
    du[2] = β*S*I - γ*I
    du[3] = γ*I
end

# Initial conditions
u0 = [0.99, 0.01, 0.0]

# Time span
tspan = (0.0, 200.0)

# Solve the SIR model
prob = ODEProblem(sir_model!, u0, tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

# Plot the solution
using Plots
p = plot(sol)

# Save the plot
savefig(p, "sir_model_plot.png")
