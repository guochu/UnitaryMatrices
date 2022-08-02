


struct UnitaryMatrix{T <: Complex, RT <: Real} <: AbstractUnitaryMatrix{T}
	rotations::Vector{RotationMatrix{T, RT}}
	diagonals::Vector{RT}
end


function UnitaryMatrix(θs::Vector{<:Real}, n::Int)
	(length(θs) == n^2) || throw("number of parameters should be n^2.")
	start_pos = 0
	RT = eltype(θs)
	T = Complex{RT}
	rotations = Vector{RotationMatrix{T, RT}}()
	current_pos = 0
	for i in 1:n
		L = _nparas(T, n, start_pos)
		mj = RotationMatrix{T}(θs[current_pos+1:current_pos+L], n, start_pos)
		push!(rotations,  mj)
		current_pos += L
		start_pos = 1 - start_pos
	end
	# println("current pos is $current_pos")
	@assert current_pos+n == length(θs)
	return UnitaryMatrix(rotations, θs[current_pos+1:current_pos+n])
end


UnitaryMatrix(::Type{T}, n::Int) where {T<:Real} = UnitaryMatrix(rand(T, n*n) .* 2π, n)
UnitaryMatrix(n::Int) = UnitaryMatrix(Float64, n)


Base.size(x::UnitaryMatrix) = (length(x.diagonals), length(x.diagonals))
Base.size(x::UnitaryMatrix, i::Int) = (@assert ((i==1) || (i==2)); length(x.diagonals))
Base.Matrix(m::UnitaryMatrix) = pure_evolve(m, Matrix(LinearAlgebra.I, size(m)))
Base.eltype(::Type{UnitaryMatrix{T, RT}}) where {T, RT} = T
Base.eltype(m::UnitaryMatrix) = eltype(typeof(m))


Base.:*(m::UnitaryMatrix, x::AbstractVecOrMat) = pure_evolve(m, x)

function parameters(m::UnitaryMatrix)
	paras = real(eltype(m))[]
	for r in m.rotations
		append!(paras, parameters(r))
	end
	append!(paras, m.diagonals)
	return paras
end
nparameters(m::UnitaryMatrix) = (length(m.diagonals))^2

nrotations(m::UnitaryMatrix) = length(m.rotations)

function pure_evolve_util(m::UnitaryMatrix, x::AbstractVecOrMat)
	# rwork, cwork = compute_workspace(m)
	T = promote_type(eltype(m), eltype(x))
	rwork = Vector{real(T)}(undef, size(m, 1))
	cwork = Vector{T}(undef, size(m, 1))
	y = _typed_copy(T, x)
	for r in m.rotations
		y = _rx_cpx!(r.θs, y, y, r.start_pos, rwork, cwork)
	end
	for i in 1:size(m, 1)
		cwork[i] = exp(im*m.diagonals[i])
	end
	pure_apply_diagonals!(cwork, y, y)
	return y, rwork, cwork
end
pure_evolve(m::UnitaryMatrix, x::AbstractVecOrMat) = pure_evolve_util(m, x)[1]

function dm_evolve_util(m::UnitaryMatrix, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}})
	# rwork, cwork = compute_workspace(m)
	T = promote_type(eltype(m), eltype(x))
	rwork = Vector{real(T)}(undef, size(m, 1))
	cwork = Vector{T}(undef, size(m, 1))
	y = Array{T}(x)
	for r in m.rotations
		y = _rxrd_cpx!(r.θs, y, y, r.start_pos, rwork, cwork)
	end
	for i in 1:size(m, 1)
		cwork[i] = exp(im*m.diagonals[i])
	end
	dm_apply_diagonals!(cwork, y, y)
	return y, rwork, cwork	
end
dm_evolve(m::UnitaryMatrix, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}}) = dm_evolve_util(m, x)[1]

Zygote.@adjoint UnitaryMatrix(θs::Vector{<:Real}, n::Int) = UnitaryMatrix(θs, n), z -> (z, nothing)

Zygote.@adjoint pure_evolve(m::UnitaryMatrix, x::AbstractVecOrMat) = begin
	y, rwork, cwork = pure_evolve_util(m, x)
	return y, Δ -> begin
		# rwork, cwork = compute_workspace(m)
		Δ, ∇θ, x1 = pure_back_propagate(Δ, m, copy(y), rwork, cwork)
		return ∇θ, Δ
	end
end

Zygote.@adjoint dm_evolve(m::UnitaryMatrix, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}}) = begin
	y, rwork, cwork = dm_evolve_util(m, x)
	return y, Δ -> begin
		# rwork, cwork = compute_workspace(m)
		Δ, ∇θ, x1 = dm_back_propagate(Δ, m, copy(y), rwork, cwork)
		return ∇θ, Δ
	end
end



function pure_back_propagate(Δ::AbstractVecOrMat, m::UnitaryMatrix, y::AbstractVecOrMat, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	RT = real(eltype(m))
	∇θs = Vector{RT}[]
	Δ = convert(typeof(y), Δ)
	Δ, ∇Ds, y = pure_diagonal_back_propagate(Δ, m.diagonals, y, cwork)
	push!(∇θs, ∇Ds)
	for r in Iterators.reverse(m.rotations)
		Δ, ∇θ, y = pure_back_propagate(Δ, r, y, rwork, cwork)
		push!(∇θs, ∇θ)
	end
	∇θs_all = RT[]
	for item in Iterators.reverse(∇θs)
		append!(∇θs_all, item)
	end
	return Δ, ∇θs_all, y
end

function dm_back_propagate(Δ::Union{AbstractMatrix, AbstractArray{<:Number, 3}}, m::UnitaryMatrix, y::Union{AbstractMatrix, AbstractArray{<:Number, 3}}, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	RT = real(eltype(m))
	∇θs = Vector{RT}[]
	Δ = convert(typeof(y), Δ)
	Δ, ∇Ds, y = dm_diagonal_back_propagate(Δ, m.diagonals, y, cwork)
	push!(∇θs, ∇Ds)
	for r in Iterators.reverse(m.rotations)
		Δ, ∇θ, y = dm_back_propagate(Δ, r, y, rwork, cwork)
		push!(∇θs, ∇θ)
	end
	∇θs_all = RT[]
	for item in Iterators.reverse(∇θs)
		append!(∇θs_all, item)
	end
	return Δ, ∇θs_all, y
end

