
    # Define the SIR model parameters and initial conditions
    b = 0.1
    g = 0.05
    S0 = 0.99
    I0 = 0.01
    R0 = 0.0
    tspan = (0.0, 200.0)

    # Define the SIR model ODEs
    function sir_ode!(du, u, p, t)
        S, I, R = u
        b, g = p
        du[1] = -b*S*I
        du[2] = b*S*I - g*I
        du[3] = g*I
    end

    # Solve the SIR model ODEs numerically
    using DifferentialEquations

    u0 = [S0, I0, R0]
    p = [b, g]
    prob = ODEProblem(sir_ode!, u0, tspan, p)
    sol = solve(prob)

    # Plot the solution
    using Plots

    plot(sol, xlabel="Time", ylabel="Proportions", title="SIR Model")
    