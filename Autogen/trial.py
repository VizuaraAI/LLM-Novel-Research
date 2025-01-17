from julia import Main

# Execute the entire Julia script as a multi-line string
result = Main.eval("""
using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, Plots

rng = MersenneTwister(99)
u0 = [2.0, 0.0]
datasize = 30
tspan = (0.0, 1.5)
tsteps = range(tspan[1], tspan[2], length=datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(); saveat=tsteps))

dudt2 = Lux.Chain(x -> x .^ 3, Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat=tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

function callback(state, l; doplot=false)
    println(l)
    if doplot
        pred = predict_neuralode(state.u)
        plt = scatter(tsteps, ode_data[1, :]; label="data")
        scatter!(plt, tsteps, pred[1, :]; label="prediction")
        display(plt)
    end
    return false
end

pinit = ComponentArray(p)
callback((; u=pinit), loss_neuralode(pinit), doplot=true)

adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05); callback=callback, maxiters=300)        

optprob2 = remake(optprob; u0=result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01); callback, allow_f_increases=false)    

callback((; u=result_neuralode2.u), loss_neuralode(result_neuralode2.u), doplot=true)
savefig("C:\\Users\\Raymundoneo\\Documents\\LLM-Novel-research project\\LLM-Novel-Research\\code\\sirmodel.png")
""")
print(result)
