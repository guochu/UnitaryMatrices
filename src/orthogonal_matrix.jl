

struct OrthogonalMatrix{T <: Real} <: AbstractUnitaryMatrix{T}
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
Base.eltype(::Type{OrthogonalMatrix{T}}) where {T} = T
Base.eltype(m::OrthogonalMatrix) = eltype(typeof(m))


Base.:*(m::OrthogonalMatrix, x::AbstractVecOrMat) = pure_evolve(m, x)

function parameters(m::OrthogonalMatrix)
	paras = eltype(m)[]
	for r in m.rotations
		append!(paras, parameters(r))
	end
	return paras
end
nparameters(m::OrthogonalMatrix) = div(m.n*(m.n-1), 2)
nrotations(m::OrthogonalMatrix) = length(m.rotations)


function pure_evolve_util(m::OrthogonalMatrix, x::AbstractVecOrMat)
	T = promote_type(eltype(m), eltype(x))
	rwork = Vector{real(T)}(undef, size(m, 1))
	y = _typed_copy(T, x) 
	for r in m.rotations
		y = _rx_real!(r.θs, y, y, r.start_pos, rwork)
	end
	return y, rwork
end
pure_evolve(m::OrthogonalMatrix, x::AbstractVecOrMat) = pure_evolve_util(m, x)[1]

function dm_evolve_util(m::OrthogonalMatrix, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}})
	T = promote_type(eltype(m), eltype(x))
	rwork = Vector{real(T)}(undef, size(m, 1))
	y = Array{T}(x)
	for r in m.rotations
		y = _rxrd_real!(r.θs, y, y, r.start_pos, rwork)
	end
	return y, rwork	
end
dm_evolve(m::OrthogonalMatrix, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}}) = dm_evolve_util(m, x)[1]

Zygote.@adjoint OrthogonalMatrix(θs::Vector{<:Real}, n::Int) = OrthogonalMatrix(θs, n), z -> (z, nothing)

Zygote.@adjoint pure_evolve(m::OrthogonalMatrix, x::AbstractVecOrMat) = begin
	y, rwork = pure_evolve_util(m, x)
	return y, Δ -> begin
		Δ, ∇θ, x1 = pure_back_propagate(Δ, m, copy(y), rwork)
		return ∇θ, Δ
	end
end

Zygote.@adjoint dm_evolve(m::OrthogonalMatrix, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}}) = begin
	y, rwork = dm_evolve_util(m, x)
	return y, Δ -> begin
		Δ, ∇θ, x1 = dm_back_propagate(Δ, m, copy(y), rwork)
		return ∇θ, Δ
	end
end


function pure_back_propagate(Δ::AbstractVecOrMat, m::OrthogonalMatrix, y::AbstractVecOrMat, rwork::Vector{<:Real})
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

function dm_back_propagate(Δ::Union{AbstractMatrix, AbstractArray{<:Number, 3}}, m::OrthogonalMatrix, y::Union{AbstractMatrix, AbstractArray{<:Number, 3}}, rwork::Vector{<:Real})
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


