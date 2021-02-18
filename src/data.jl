import Random

function linreg_toy_data()
    Random.seed!(1231245)
    σ = .5
    N = 50
    X = collect(range(-6, 6; length=N)) .+ randn(N)
    y = X .+ randn(N)
    y[10:30] .= y[10:30] .+ 4
    y[30:40] .= sin.(y[30:40])
    Array(reshape(X, 1, :)), y
end
