using Parameters, Optim, LineSearches, Distributions, LinearAlgebra
# importing the Pkg module
using Pkg
Pkg.add("CSV")

# installing the DataFrame
Pkg.add("DataFrames")


number_of_repeats = 50
LR = 0.1
data_file = "C:\\Users\\Owner\\Desktop\\MastersProject\\tests\\test_data\\test1.txt"

gd = GradientDescent(linesearch=Static(), alphaguess=LR)

function generate_poly(coeffs_squares, coeffs_first_dgr)

    return function f(x ::Vector{Float64})
        squares = sum(x .^ 2 .* coeffs_squares)
        linear = sum(coeffs_first_dgr .* x)
        return linear + squares
        end


end

function generate_grad(coeffs_squares, coeffs_first_dgr)

        return function g!(G, x ::Vector{Float64})

                for i in 1:length(x)
                    G[i] = 2*coeffs_squares[i]*x[i] + coeffs_first_dgr[i]
                end

                return G

               end
end


for i in 1:number_of_repeats
    coeffs1 = rand(Uniform(-1.0,1.0),3)
    coeffs_first_dgr = rand(Uniform(-1.0,1.0),3)
    f = generate_poly(coeffs1, coeffs_first_dgr)
    g! = generate_grad(coeffs1, coeffs_first_dgr)
    x0 = rand(Uniform(-1.0,1.0),3)
    open(data_file, "w") do file
        write(file,coeffs1)
        write(file, "\n")
        write(file, coeffs_first_dgr)
        write(file, "\n")
        write(file, x0)
        write(file, "\n")
    end
    res = optimize(f,
               g!,
               x0,
               gd,
               Optim.Options(g_tol = 0,
                             x_tol = 1e-3,
                             iterations = 1000,
                             store_trace = true,
                             show_trace = true,
                             extended_trace= true,
                             show_warnings = true))
    open(data_file, "w") do file
        writedlm(file, Optim.x_trace(res), ", ")
        write(file, "\n - \n")
    end
end