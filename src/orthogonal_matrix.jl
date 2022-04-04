

struct OrthogonalMatrix{T <: Real} <: AbstractMatrix{T}
	rotations::Vector{RotationMatrix{T, T}}
	n::Int
end


function OrthogonalMatrix(θs::Vector{<:Real}, n::Int)
	(length(θs) == div(n*(n-1), 2) ) || throw("number of parameters should be n(n-1)/2.")
	start_pos = 0
	RT = eltype(θs)
	rotations = Vector{RotationMatrix{RT, RT}}()
	current_pos = 0
	for i in 1:n
		L = _nparas(RT, n, start_pos)
		mj = RotationMatrix{RT}(θs[current_pos+1:current_pos+L], n, start_pos)
		push!(rotations,  mj)
		current_pos += L
		start_pos = 1 - start_pos
	end
	# println("current pos is $current_pos")
	@assert current_pos == length(θs)
	return OrthogonalMatrix(rotations, n)
end

OrthogonalMatrix(::Type{T}, n::Int) where {T<:Real} = OrthogonalMatrix(rand(T, div(n*(n-1), 2) ) .* 2π, n)
OrthogonalMatrix(n::Int) = OrthogonalMatrix(Float64, n)

Base.size(x::OrthogonalMatrix) = (x.n, x.n)
Base.size(x::OrthogonalMatrix, i::Int) = (@assert ((i==1) || (i==2)); x.n)
Base.Matrix(m::OrthogonalMatrix) = pure_evolve(m, Matrix(LinearAlgebra.I, size(m)))

function parameters(m::OrthogonalMatrix)
	paras = eltype(m)[]
	for r in m.rotations
		append!(paras, parameters(r))
	end
	return paras
end

nrotations(m::OrthogonalMatrix) = length(m.rotations)


function pure_evolve_util(m::OrthogonalMatrix, x::AbstractMatrix)
	T = promote_type(eltype(m), eltype(x))
	rwork = Vector{real(T)}(undef, size(m, 1))
	y = Matrix{T}(x)
	for r in m.rotations
		y = _rx_real!(r.θs, y, y, r.start_pos, rwork)
	end
	return y, rwork
end
pure_evolve(m::OrthogonalMatrix, x::AbstractMatrix) = pure_evolve_util(m, x)[1]

function dm_evolve_util(m::OrthogonalMatrix, x::AbstractMatrix)
	T = promote_type(eltype(m), eltype(x))
	rwork = Vector{real(T)}(undef, size(m, 1))
	y = Matrix{T}(x)
	for r in m.rotations
		y = _rxrd_real!(r.θs, y, y, r.start_pos, rwork)
	end
	return y, rwork	
end
dm_evolve(m::OrthogonalMatrix, x::AbstractMatrix) = dm_evolve_util(m, x)[1]


Zygote.@adjoint pure_evolve(m::OrthogonalMatrix, x::AbstractMatrix) = begin
	y, rwork = pure_evolve_util(m, x)
	return y, Δ -> begin
		Δ, ∇θ, x1 = pure_back_propagate(Δ, m, y, rwork)
		return ∇θ, Δ
	end
end

Zygote.@adjoint dm_evolve(m::OrthogonalMatrix, x::AbstractMatrix) = begin
	y, rwork = dm_evolve_util(m, x)
	return y, Δ -> begin
		Δ, ∇θ, x1 = dm_back_propagate(Δ, m, y, rwork)
		return ∇θ, Δ
	end
end


function pure_back_propagate(Δ::AbstractMatrix, m::OrthogonalMatrix, y::AbstractMatrix, rwork::Vector{<:Real})
	RT = real(eltype(m))
	∇θs = Vector{RT}[]
	Δ = convert(typeof(y), Δ)
	for r in Iterators.reverse(m.rotations)
		Δ, ∇θ, y = pure_back_propagate(Δ, r, y, rwork)
		push!(∇θs, ∇θ)
	end
	∇θs_all = RT[]
	for item in Iterators.reverse(∇θs)
		append!(∇θs_all, item)
	end
	return Δ, ∇θs_all, y
end

function dm_back_propagate(Δ::AbstractMatrix, m::OrthogonalMatrix, y::AbstractMatrix, rwork::Vector{<:Real})
	RT = real(eltype(m))
	∇θs = Vector{RT}[]
	Δ = convert(typeof(y), Δ)
	for r in Iterators.reverse(m.rotations)
		Δ, ∇θ, y = dm_back_propagate(Δ, r, y, rwork)
		push!(∇θs, ∇θ)
	end
	∇θs_all = RT[]
	for item in Iterators.reverse(∇θs)
		append!(∇θs_all, item)
	end
	return Δ, ∇θs_all, y
end










