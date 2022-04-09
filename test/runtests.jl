push!(LOAD_PATH, "../src")

include("util.jl")

using Test
using Zygote, TensorOperations
using UnitaryMatrices

loss_ad(m, x) = abs(sum(pure_evolve(m, x)))
loss_dm_ad(m, x) = abs(sum(dm_evolve(m, x)))

println("-----------test rotational matrcies-----------------")

function test_rx(::Type{T}, n::Int, start_pos::Int) where {T<:Number}
	m = RotationMatrix{T}(n, start_pos)
	mm = Matrix(m)
	x = randn(ComplexF64, n, 3)
	return maximum(abs.(pure_evolve(m, x) - mm * x)) < 1.0e-6
end

function test_rx_ad(::Type{T}, n::Int, start_pos::Int) where {T <: Number}
	m = RotationMatrix{T}(n, start_pos)
	x = randn(T, n, 3)

	loss_fd(θs, y) = abs(sum(pure_evolve(RotationMatrix{T}(θs, n, start_pos), y)))

	grad1 = Zygote.gradient(loss_ad, m, x)
	grad2 = fdm_gradient(loss_fd, m.θs, x)

	return max( maximum(abs.(grad1[1] - grad2[1])), maximum(abs.(grad1[2] - grad2[2])) ) < 1.0e-6
	# return grad1, grad2
end


function test_rx_dm_evolve(::Type{T}, n::Int, start_pos::Int) where {T <: Number}
	m = RotationMatrix{T}(n, start_pos)
	mm = Matrix(m)
	x = randn(ComplexF64, n, n)
	# x = x' * x
	return maximum(abs.( dm_evolve(m, x) - mm * x * mm')) < 1.0e-6
end


function test_rx_dm_ad(::Type{T}, n::Int, start_pos::Int) where {T <: Number}
	m = RotationMatrix{T}(n, start_pos)
	x = randn(T, n, n)
	x = x' + x

	loss_dm_fd(θs, y) = abs(sum(dm_evolve(RotationMatrix{T}(θs, n, start_pos), y)))

	grad1 = Zygote.gradient(loss_dm_ad, m, x)
	grad2 = fdm_gradient(loss_dm_fd, m.θs, x)

	return max( maximum(abs.(grad1[1] - grad2[1])), maximum(abs.(grad1[2] - grad2[2])) ) < 1.0e-6
end

@testset "rotational matrcies operations" begin
	for T in (Float64, ComplexF64)
		for start_pos in (0, 1)
			@test test_rx(T, 5, start_pos)
			@test test_rx_ad(T, 5, start_pos)
			@test test_rx_dm_evolve(T, 5, start_pos)
			@test test_rx_dm_ad(T, 5, start_pos)
		end
	end
end

println("-----------test unitary matrcies-----------------")


function test_ux(n::Int)
	m = UnitaryMatrix(n)
	mm = Matrix(m)
	x = randn(ComplexF64, n, 3)
	return maximum(abs.(pure_evolve(m, x) - mm * x)) < 1.0e-6
end

function test_ux_ad(n::Int) 
	m = UnitaryMatrix(n)
	x = randn(eltype(m), n, 3)

	loss_fd(θs, y) = abs(sum(pure_evolve(UnitaryMatrix(θs, n), y)))

	grad1 = Zygote.gradient(loss_ad, m, x)
	grad2 = fdm_gradient(loss_fd, parameters(m), x)

	return max( maximum(abs.(grad1[1] - grad2[1])), maximum(abs.(grad1[2] - grad2[2])) ) < 1.0e-6
	# return grad1, grad2
end

function test_ux_dm_evolve(n::Int) 
	m = UnitaryMatrix(n)
	mm = Matrix(m)
	x = randn(ComplexF64, n, n)
	return maximum(abs.( dm_evolve(m, x) - mm * x * mm')) < 1.0e-6
end

function test_ux_dm_evolve_3(n::Int) 
	m = UnitaryMatrix(n)
	mm = Matrix(m)
	x = randn(ComplexF64, n, n, 4)
	@tensor tmp[1,5,4] := mm[1,2] * x[2,3,4] * conj(mm[5,3])
	return maximum(abs.( dm_evolve(m, x) - tmp)) < 1.0e-6
end

function test_ux_dm_ad(n::Int) 
	m = UnitaryMatrix(n)
	x = randn(eltype(m), n, n)
	x = x' + x

	loss_dm_fd(θs, y) = abs(sum(dm_evolve(UnitaryMatrix(θs, n), y)))

	grad1 = Zygote.gradient(loss_dm_ad, m, x)
	grad2 = fdm_gradient(loss_dm_fd, parameters(m), x)

	return max( maximum(abs.(grad1[1] - grad2[1])), maximum(abs.(grad1[2] - grad2[2])) ) < 1.0e-6
end

function test_ux_dm_ad_3(n::Int) 
	m = UnitaryMatrix(n)
	x = randn(eltype(m), 2*n, 2*n)
	x = x' + x
	x = reshape(permutedims(reshape(x, n, 2, n, 2), (1,3,2,4)), n, n, 4)

	loss_dm_fd(θs, y) = abs(sum(dm_evolve(UnitaryMatrix(θs, n), y)))

	grad1 = Zygote.gradient(loss_dm_ad, m, x)
	grad2 = fdm_gradient(loss_dm_fd, parameters(m), x)

	return max( maximum(abs.(grad1[1] - grad2[1])), maximum(abs.(grad1[2] - grad2[2])) ) < 1.0e-6
end

@testset "unitary matrcies operations" begin
	for n in (4, 5)
		@test test_ux(n)
		@test test_ux_ad(n)
		@test test_ux_dm_evolve(n)
		@test test_ux_dm_evolve_3(n)
		@test test_ux_dm_ad(n)
		@test test_ux_dm_ad_3(n)
	end
end


println("-----------test orthogonal matrcies-----------------")


function test_ox(n::Int)
	m = OrthogonalMatrix(n)
	mm = Matrix(m)
	x = randn(ComplexF64, n, 3)
	return maximum(abs.(pure_evolve(m, x) - mm * x)) < 1.0e-6
end

function test_ox_ad(n::Int) 
	m = OrthogonalMatrix(n)
	x = randn(eltype(m), n, 3)

	loss_fd(θs, y) = abs(sum(pure_evolve(OrthogonalMatrix(θs, n), y)))

	grad1 = Zygote.gradient(loss_ad, m, x)
	grad2 = fdm_gradient(loss_fd, parameters(m), x)

	return max( maximum(abs.(grad1[1] - grad2[1])), maximum(abs.(grad1[2] - grad2[2])) ) < 1.0e-6
	# return grad1, grad2
end

function test_ox_dm_evolve(n::Int) 
	m = OrthogonalMatrix(n)
	mm = Matrix(m)
	x = randn(ComplexF64, n, n)
	return maximum(abs.( dm_evolve(m, x) - mm * x * mm')) < 1.0e-6
end

function test_ox_dm_evolve_3(n::Int) 
	m = OrthogonalMatrix(n)
	mm = Matrix(m)
	x = randn(ComplexF64, n, n, 4)
	@tensor tmp[1,5,4] := mm[1,2] * x[2,3,4] * conj(mm[5,3])
	return maximum(abs.( dm_evolve(m, x) - tmp)) < 1.0e-6
end

function test_ox_dm_ad(n::Int) 
	m = OrthogonalMatrix(n)
	x = randn(eltype(m), n, n)
	x = x' + x

	loss_dm_fd(θs, y) = abs(sum(dm_evolve(OrthogonalMatrix(θs, n), y)))

	grad1 = Zygote.gradient(loss_dm_ad, m, x)
	grad2 = fdm_gradient(loss_dm_fd, parameters(m), x)

	return max( maximum(abs.(grad1[1] - grad2[1])), maximum(abs.(grad1[2] - grad2[2])) ) < 1.0e-6
end

function test_ox_dm_ad_3(n::Int) 
	m = OrthogonalMatrix(n)
	x = randn(eltype(m), 2*n, 2*n)
	x = x' + x
	x = reshape(permutedims(reshape(x, n, 2, n, 2), (1,3,2,4)), n, n, 4)

	loss_dm_fd(θs, y) = abs(sum(dm_evolve(OrthogonalMatrix(θs, n), y)))

	grad1 = Zygote.gradient(loss_dm_ad, m, x)
	grad2 = fdm_gradient(loss_dm_fd, parameters(m), x)

	return max( maximum(abs.(grad1[1] - grad2[1])), maximum(abs.(grad1[2] - grad2[2])) ) < 1.0e-6
end

@testset "orthogonal matrcies operations" begin
	for n in (4, 5)
		@test test_ox(n)
		@test test_ox_ad(n)
		@test test_ox_dm_evolve(n)
		@test test_ox_dm_evolve_3(n)
		@test test_ox_dm_ad(n)
		@test test_ox_dm_ad_3(n)
	end
end


