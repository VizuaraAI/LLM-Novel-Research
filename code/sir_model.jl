
    using DifferentialEquations, Plots

    function sir_model!(du, u, p, t)
        S, I, R = u
        b, g = p
        du[1] = -b*S*I
        du[2] = b*S*I - g*I
        du[3] = g*I
    end

    u0 = [0.99, 0.01, 0.0]
    p = [0.1, 0.05]
    tspan = (0.0, 200.0)

    prob = ODEProblem(sir_model!, u0, tspan, p)
    sol = solve(prob)

    plot(sol, xlabel="Time", ylabel="Proportion", title="SIR Model", lw=2)
    savefig("sir_model_plot.png")
    