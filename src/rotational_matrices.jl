



"""
	struct RotationMatrix{T <: Real} <: AbstractMatrix{Complex{T}}
	n : matrix size
	start_pos is 0 or 1
"""
struct RotationMatrix{T, RT} <: AbstractMatrix{T}
	θs::Vector{RT}
	# ϕs::Vector{RT}
	n::Int
	start_pos::Int

end

function RotationMatrix{T}(θs::Vector{<:Real}, n::Int, start_pos::Int) where {T<:Complex}
	RT = real(T)
	@assert ((start_pos==0) || (start_pos==1))
	@assert length(θs) == _nparas(T, n, start_pos)
	RotationMatrix{T, RT}(convert(Vector{RT}, θs), n, start_pos)
end

function RotationMatrix{T}(θs::Vector{<:Real}, n::Int, start_pos::Int) where {T <: Real}
	@assert ((start_pos==0) || (start_pos==1))
	@assert length(θs) == _nparas(T, n, start_pos)
	RotationMatrix{T, T}(convert(Vector{T}, θs), n, start_pos)
end
RotationMatrix{T}(n::Int, start_pos::Int) where {T<:Number} = RotationMatrix{T}(rand(_nparas(T, n, start_pos)), n, start_pos)

function _nparas(::Type{T}, n::Int, start_pos::Int) where {T<:Number}
	nh = (start_pos == 0) ? div(n, 2) : div(n+1, 2)-1
	return (T <: Real) ? nh : 2*nh
end

Base.size(x::RotationMatrix) = (x.n, x.n)
Base.size(x::RotationMatrix, i::Int) = (@assert ((i==1) || (i==2)); x.n)
Base.Matrix(m::RotationMatrix) = pure_evolve(m, Matrix(LinearAlgebra.I, size(m)))

parameters(m::RotationMatrix) = m.θs

function pure_evolve_util(m::RotationMatrix{<:Real}, x::AbstractMatrix)
	rwork = compute_workspace(m)
	T = promote_type(eltype(m), eltype(x))
	y = _rx_real!(m.θs, x, Matrix{T}(x), m.start_pos, rwork)
	return y, rwork
end

function pure_evolve_util(m::RotationMatrix{<:Complex}, x::AbstractMatrix)
	rwork, cwork = compute_workspace(m)
	T = promote_type(eltype(m), eltype(x))
	y = _rx_cpx!(m.θs, x, Matrix{T}(x), m.start_pos, rwork, cwork)
	return y, rwork, cwork
end
pure_evolve(m::RotationMatrix, x::AbstractMatrix) = pure_evolve_util(m, x)[1]


Zygote.@adjoint pure_evolve(m::RotationMatrix{<:Real}, x::AbstractMatrix) = begin
	y, rwork = pure_evolve_util(m, x)
	return y, Δ -> begin
		# rwork = compute_workspace(m)
		Δ, ∇θ, x1 = pure_back_propagate(Δ, m, y, rwork)
		return ∇θ, Δ
	end
end
Zygote.@adjoint pure_evolve(m::RotationMatrix{<:Complex}, x::AbstractMatrix) = begin
	y, rwork, cwork = pure_evolve_util(m, x)
	return y, Δ -> begin
		# rwork, cwork = compute_workspace(m)
		Δ, ∇θ, x1 = pure_back_propagate(Δ, m, y, rwork, cwork)
		return ∇θ, Δ
	end
end

function dm_evolve_util(m::RotationMatrix{<:Real}, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}})
	rwork = compute_workspace(m)
	T = promote_type(eltype(m), eltype(x))
	y = _rxrd_real!(m.θs, x, Array{T}(x), m.start_pos, rwork)
	return y, rwork
end

function dm_evolve_util(m::RotationMatrix{<:Complex}, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}})
	rwork, cwork = compute_workspace(m)
	T = promote_type(eltype(m), eltype(x))
	y = _rxrd_cpx!(m.θs, x, Array{T}(x), m.start_pos, rwork, cwork)
	return y, rwork, cwork
end
dm_evolve(m::RotationMatrix, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}}) = dm_evolve_util(m, x)[1]

Zygote.@adjoint dm_evolve(m::RotationMatrix{<:Real}, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}}) = begin
	y, rwork = dm_evolve_util(m, x)
	return y, Δ -> begin
		# rwork = compute_workspace(m)
		Δ, ∇θ, x1 = dm_back_propagate(Δ, m, y, rwork)
		return ∇θ, Δ
	end
end

Zygote.@adjoint dm_evolve(m::RotationMatrix{<:Complex}, x::Union{AbstractMatrix, AbstractArray{<:Number, 3}}) = begin
	y, rwork, cwork = dm_evolve_util(m, x)
	return y, Δ -> begin
		# rwork, cwork = compute_workspace(m)
		Δ, ∇θ, x1 = dm_back_propagate(Δ, m, y, rwork, cwork)
		return ∇θ, Δ
	end
end

function pure_back_propagate(Δ::AbstractMatrix, m::RotationMatrix{<:Real}, y::AbstractMatrix, rwork::Vector{<:Real})
	∇θ, Δ = _∇rx_real!(Δ, m.θs, y, m.start_pos, rwork)
	return Δ, ∇θ, y
end

function pure_back_propagate(Δ::AbstractMatrix, m::RotationMatrix{<:Complex}, y::AbstractMatrix, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	∇θ, Δ = _∇rx_cpx!(Δ, m.θs, y, m.start_pos, rwork, cwork)
	return Δ, ∇θ, y
end

function dm_back_propagate(Δ::AbstractMatrix, m::RotationMatrix{<:Real}, y::AbstractMatrix, rwork::Vector{<:Real})
	∇θ, Δ = _∇rxrd_real!(Δ, m.θs, y, m.start_pos, rwork)
	return Δ, ∇θ, y
end
function dm_back_propagate(Δ::AbstractArray{<:Number, 3}, m::RotationMatrix{<:Real}, y::AbstractArray{<:Number, 3}, rwork::Vector{<:Real})
	∇θ, Δ = _∇rxrd_real!(Δ, m.θs, y, m.start_pos, rwork)
	return Δ, ∇θ, y
end

function dm_back_propagate(Δ::AbstractMatrix, m::RotationMatrix{<:Complex}, y::AbstractMatrix, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	∇θ, Δ = _∇rxrd_cpx!(Δ, m.θs, y, m.start_pos, rwork, cwork)
	return Δ, ∇θ, y
end
function dm_back_propagate(Δ::AbstractArray{<:Number, 3}, m::RotationMatrix{<:Complex}, y::AbstractArray{<:Number, 3}, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	∇θ, Δ = _∇rxrd_cpx!(Δ, m.θs, y, m.start_pos, rwork, cwork)
	return Δ, ∇θ, y
end


function compute_workspace(m::RotationMatrix{<:Complex})
	rwork = similar(m.θs, length(m.θs))
	cwork = Vector{eltype(m)}(undef, div(length(m.θs), 2) )
	return rwork, cwork
end
compute_workspace(m::RotationMatrix{<:Real}) = similar(m.θs, 2*length(m.θs))



# phase = exp(iϕ)
@inline _r2x_real(costheta, sintheta, x1, x2) = costheta * x1 - sintheta*x2, sintheta * x1 + costheta * x2
@inline function _r2x_cpx(phase, costheta, sintheta, x1, x2) 
	phase_sintheta = phase * sintheta
	return costheta * x1 - phase_sintheta * x2, conj(phase_sintheta) * x1 + costheta * x2
end


@inline  _∇r2x_real(x1, x2, rΔ1, rΔ2) = conj(x1) * rΔ2 - conj(x2) * rΔ1
@inline function _∇r2x_cpx(phase, costheta, sintheta, x1, x2, rΔ1, rΔ2)
	∇θ = phase * conj(x1) * rΔ2-conj(phase) * conj(x2) * rΔ1
	sincostheta = phase * sintheta * costheta
	sinsintheta = sintheta^2
	∇ϕ = im * (conj(x1) * (sinsintheta * rΔ1 + sincostheta * rΔ2) + conj(x2) * (conj(sincostheta)*rΔ1- sinsintheta*rΔ2))
	return real(∇θ), real(∇ϕ)
end 

function _rx_real!(θs, x::AbstractMatrix, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real})
	L = length(θs)
	@assert length(rwork) >= 2*L
	rwork2 = view(rwork, L+1:2*L)
	for i in 1:L
		sintheta, costheta = sincos(θs[i])
		rwork[i] = sintheta
		rwork2[i] = costheta
	end

	for j in 1:size(x, 2)
		@inbounds for i in 1:L
			sintheta, costheta = rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			# phase = exp(im*ϕs[i])
			# sintheta, costheta = sincos(θs[i])
			x1, x2 = x[pos, j], x[pos+1, j]
			y[pos, j], y[pos+1, j] = _r2x_real(costheta, sintheta, x1, x2)
		end
	end
	return y
end

function _rx_cpx!(θs, x::AbstractMatrix, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	L = div(length(θs), 2)
	@assert length(rwork) >= 2*L
	@assert length(cwork) >= L
	rwork2 = view(rwork, L+1:2*L)
	@inbounds for i in 1:L
		cwork[i] = exp(im*θs[L+i])
		sintheta, costheta = sincos(θs[i])
		rwork[i] = sintheta
		rwork2[i] = costheta
	end

	for j in 1:size(x, 2)
		@inbounds for i in 1:L
			phase, sintheta, costheta = cwork[i], rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			# phase = exp(im*ϕs[i])
			# sintheta, costheta = sincos(θs[i])
			x1, x2 = x[pos, j], x[pos+1, j]
			y[pos, j], y[pos+1, j] = _r2x_cpx(phase, costheta, sintheta, x1, x2)
		end
	end
	return y
end

function init_rwork!(θs, rwork::Vector{<:Real})
	L = length(θs)
	@assert length(rwork) >= 2*L
	rwork2 = view(rwork, L+1:2*L)
	@inbounds for i in 1:L
		sintheta, costheta = sincos(θs[i])
		rwork[i] = sintheta
		rwork2[i] = costheta
	end	
end


function _rxrd_real_util!(θs, x::AbstractMatrix, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real})
	L = length(θs)
	rwork2 = view(rwork, L+1:2*L)

	for j in 1:size(x, 2)
		@inbounds for i in 1:L
			sintheta, costheta = rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			x1, x2 = x[pos, j], x[pos+1, j]
			y[pos, j], y[pos+1, j] = _r2x_real(costheta, sintheta, x1, x2)
		end
	end
	@inbounds for i in 1:L
		sintheta, costheta = rwork[i], rwork2[i]
		pos = 2*i-1+start_pos
		for j in 1:size(x, 1)
			x1, x2 = y[j, pos], y[j, pos+1]		
			y[j, pos], y[j, pos+1] = _r2x_real(costheta, sintheta, x1, x2)
		end		
	end
	return y
end

function _rxrd_real!(θs, x::AbstractMatrix, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real})
	init_rwork!(θs, rwork)
	return _rxrd_real_util!(θs, x, y, start_pos, rwork)
end

"""
	The first two dimensions of y correspond to a sub-density matrix
"""
function _rxrd_real!(θs, x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3}, start_pos::Int, rwork::Vector{<:Real})
	init_rwork!(θs, rwork)
	for k in 1:size(x, 3)
		_rxrd_real_util!(θs, view(x, :, :, k), view(y, :, :, k), start_pos, rwork)
	end
	return y
end


function init_cwork!(θs, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	L = div(length(θs), 2)
	@assert length(rwork) >= 2*L
	@assert length(cwork) >= L
	rwork2 = view(rwork, L+1:2*L)
	@inbounds for i in 1:L
		cwork[i] = exp(im*θs[L+i])
		sintheta, costheta = sincos(θs[i])
		rwork[i] = sintheta
		rwork2[i] = costheta
	end
end

function _rxrd_cpx_util!(θs, x::AbstractMatrix, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	L = div(length(θs), 2)
	rwork2 = view(rwork, L+1:2*L)

	for j in 1:size(x, 2)
		@inbounds for i in 1:L
			phase, sintheta, costheta = cwork[i], rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			x1, x2 = x[pos, j], x[pos+1, j]
			y[pos, j], y[pos+1, j] = _r2x_cpx(phase, costheta, sintheta, x1, x2)
		end
	end
	@inbounds for i in 1:L
		phase, sintheta, costheta = conj(cwork[i]), rwork[i], rwork2[i]
		pos = 2*i-1+start_pos
		for j in 1:size(x, 1)
			x1, x2 = y[j, pos], y[j, pos+1]		
			y[j, pos], y[j, pos+1] = _r2x_cpx(phase, costheta, sintheta, x1, x2)
		end		
	end
	return y
end

function _rxrd_cpx!(θs, x::AbstractMatrix, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	init_cwork!(θs, rwork, cwork)
	return _rxrd_cpx_util!(θs, x, y, start_pos, rwork, cwork)
end
function _rxrd_cpx!(θs, x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3}, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	init_cwork!(θs, rwork, cwork)
	for k in 1:size(x, 3)
		_rxrd_cpx_util!(θs, view(x, :, :, k), view(y,:,:,k), start_pos, rwork, cwork)
	end
	return y
end


function _∇rx_real_util!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, ∇θ)
	L = length(θs)
	rwork2 = view(rwork, L+1:2*L)

	# Δ = convert(typeof(y), Δ)
	# ∇θ = zero(θs)
	for j in 1:size(y, 2)
		@inbounds for i in 1:L
			sintheta, costheta = -rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			x1, x2 = y[pos, j], y[pos+1, j]
			# compute y ← x = U† y
			y[pos, j], y[pos+1, j] = _r2x_real(costheta, sintheta, x1, x2)
			x1, x2 = Δ[pos, j], Δ[pos+1, j]
			# compute Δ ← ∇x = U† Δ 
			Δ[pos, j], Δ[pos+1, j] = _r2x_real(costheta, sintheta, x1, x2)
			# compute ∇U
			∇θ[i] += _∇r2x_real(y[pos, j], y[pos+1, j], Δ[pos, j], Δ[pos+1, j])
		end
	end	
	return ∇θ, Δ
end

"""
	_∇rx_real!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real})
	y = Ux, Δ is the gradient from the previous layer
"""
function _∇rx_real!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real})
	init_rwork!(θs, rwork)
	Δ = convert(typeof(y), Δ)
	return _∇rx_real_util!(Δ, θs, y, start_pos, rwork, zero(θs))
end

function _∇rx_cpx_util!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex}, ∇θ)
	L = div(length(θs), 2)
	rwork2 = view(rwork, L+1:2*L)

	# Δ = convert(typeof(y), Δ)
	# ∇θ = zero(θs)
	for j in 1:size(y, 2)
		@inbounds for i in 1:L
			phase, sintheta, costheta = cwork[i], rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			x1, x2 = y[pos, j], y[pos+1, j]
			# compute y ← x = U† y
			y[pos, j], y[pos+1, j] = _r2x_cpx(phase, costheta, -sintheta, x1, x2)
			x1, x2 = Δ[pos, j], Δ[pos+1, j]
			# compute Δ ← ∇x = U† Δ 
			Δ[pos, j], Δ[pos+1, j] = _r2x_cpx(phase, costheta, -sintheta, x1, x2)
			# compute ∇U
			a, b = _∇r2x_cpx(phase, costheta, sintheta, y[pos, j], y[pos+1, j], Δ[pos, j], Δ[pos+1, j])
			∇θ[i] += a 
			∇θ[L+i] += b
		end
	end	
	return ∇θ, Δ
end

"""
	_∇rx_cpx!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	y = Ux, Δ is the gradient from the previous layer
"""
function _∇rx_cpx!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	init_cwork!(θs, rwork, cwork)
	Δ = convert(typeof(y), Δ)
	return _∇rx_cpx_util!(Δ, θs, y, start_pos, rwork, cwork, zero(θs))
end


function _∇rxrd_real_util!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, ∇θ)
	L = length(θs)
	rwork2 = view(rwork, L+1:2*L)

	# Δ = convert(typeof(y), Δ)
	# ∇θ = zero(θs)
	for j in 1:size(y, 2)
		@inbounds for i in 1:L
			sintheta, costheta = -rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			x1, x2 = y[pos, j], y[pos+1, j]
			# compute y ← x = U† y
			y[pos, j], y[pos+1, j] = _r2x_real(costheta, sintheta, x1, x2)
			x1, x2 = Δ[pos, j], Δ[pos+1, j]
			# compute Δ ← ∇x = U† Δ 
			Δ[pos, j], Δ[pos+1, j] = _r2x_real(costheta, sintheta, x1, x2)
		end
	end	
	@inbounds for i in 1:L
		sintheta, costheta = -rwork[i], rwork2[i]
		pos = 2*i-1+start_pos
		for j in 1:size(y, 1)
			x1, x2 = y[j, pos], y[j, pos+1]
			y[j, pos], y[j, pos+1] = _r2x_real(costheta, sintheta, x1, x2)

			x1, x2 = Δ[j, pos], Δ[j, pos+1]
			Δ[j, pos], Δ[j, pos+1] = _r2x_real(costheta, sintheta, x1, x2)
		end
	end
	for j in 1:size(y, 2)
		@inbounds for i in 1:L
			pos = 2*i-1+start_pos
			# compute ∇U
			∇θ[i] += 2 * _∇r2x_real(y[pos, j], y[pos+1, j], Δ[pos, j], Δ[pos+1, j])
		end
	end
	return ∇θ, Δ
end


"""
	_∇rxrd_real!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real})
	y = Ux, Δ is the gradient from the previous layer
"""
function _∇rxrd_real!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real})
	init_rwork!(θs, rwork)
	Δ = convert(typeof(y), Δ)
	return _∇rxrd_real_util!(Δ, θs, y, start_pos, rwork, zero(θs))
end
function _∇rxrd_real!(Δ::AbstractArray{<:Number, 3}, θs, y::AbstractArray{<:Number, 3}, start_pos::Int, rwork::Vector{<:Real})
	init_rwork!(θs, rwork)
	Δ = convert(typeof(y), Δ)
	∇θ = zero(θs)
	for k in 1:size(y, 3)
		_∇rxrd_real_util!(view(Δ, :, :, k), θs, view(y, :, :, k), start_pos, rwork, ∇θ)
	end
	return ∇θ, Δ
end

function _∇rxrd_cpx_util!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex}, ∇θ)
	L = div(length(θs), 2)
	rwork2 = view(rwork, L+1:2*L)

	# Δ = convert(typeof(y), Δ)
	# ∇θ = zero(θs)
	for j in 1:size(y, 2)
		@inbounds for i in 1:L
			phase, sintheta, costheta = cwork[i], -rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			x1, x2 = y[pos, j], y[pos+1, j]
			# compute y ← x = U† y
			y[pos, j], y[pos+1, j] = _r2x_cpx(phase, costheta, sintheta, x1, x2)
			x1, x2 = Δ[pos, j], Δ[pos+1, j]
			# compute Δ ← ∇x = U† Δ 
			Δ[pos, j], Δ[pos+1, j] = _r2x_cpx(phase, costheta, sintheta, x1, x2)
			# # compute ∇U
			# a, b = _∇r2x_cpx(phase, costheta, sintheta, y[pos, j], y[pos+1, j], Δ[pos, j], Δ[pos+1, j])
			# ∇θ[i] += a 
			# ∇θ[L+i] += b
		end
	end	
	@inbounds for i in 1:L
		phase, sintheta, costheta = conj(cwork[i]), -rwork[i], rwork2[i]
		pos = 2*i-1+start_pos
		for j in 1:size(y, 1)
			x1, x2 = y[j, pos], y[j, pos+1]
			y[j, pos], y[j, pos+1] = _r2x_cpx(phase, costheta, sintheta, x1, x2)

			x1, x2 = Δ[j, pos], Δ[j, pos+1]
			Δ[j, pos], Δ[j, pos+1] = _r2x_cpx(phase, costheta, sintheta, x1, x2)
		end
	end
	for j in 1:size(y, 2)
		@inbounds for i in 1:L
			phase, sintheta, costheta = cwork[i], rwork[i], rwork2[i]
			pos = 2*i-1+start_pos
			# compute ∇U
			a, b = _∇r2x_cpx(phase, costheta, sintheta, y[pos, j], y[pos+1, j], Δ[pos, j], Δ[pos+1, j])
			∇θ[i] += 2*a 
			∇θ[L+i] += 2*b			
		end
	end
	return ∇θ, Δ
end

"""
	_∇rxrd_cpx!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	y = Ux, Δ is the gradient from the previous layer
"""
function _∇rxrd_cpx!(Δ::AbstractMatrix, θs, y::AbstractMatrix, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	init_cwork!(θs, rwork, cwork)
	Δ = convert(typeof(y), Δ)
	return _∇rxrd_cpx_util!(Δ,θs, y, start_pos, rwork, cwork, zero(θs))
end

function _∇rxrd_cpx!(Δ::AbstractArray{<:Number, 3}, θs, y::AbstractArray{<:Number, 3}, start_pos::Int, rwork::Vector{<:Real}, cwork::Vector{<:Complex})
	init_cwork!(θs, rwork, cwork)
	Δ = convert(typeof(y), Δ)
	∇θ = zero(θs)
	for k in 1:size(y, 3)
		_∇rxrd_cpx_util!(view(Δ,:,:,k),θs, view(y,:,:,k), start_pos, rwork, cwork, ∇θ)
	end
	return ∇θ, Δ
end




