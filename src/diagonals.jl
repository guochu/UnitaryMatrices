

function pure_diagonal_back_propagate(Δ::AbstractVecOrMat, Ds::Vector{<:Real}, y::AbstractVecOrMat, cwork::Vector{<:Complex})
	∇Ds, Δ = ∇pure_apply_diagonals!(Δ, Ds, y, cwork)
	return Δ, ∇Ds, y
end

function dm_diagonal_back_propagate(Δ::AbstractMatrix, Ds::Vector{<:Real}, y::AbstractMatrix, cwork::Vector{<:Complex})
	∇Ds, Δ = ∇dm_apply_diagonals!(Δ, Ds, y, cwork)
	return Δ, ∇Ds, y
end
function dm_diagonal_back_propagate(Δ::AbstractArray{<:Number, 3}, Ds::Vector{<:Real}, y::AbstractArray{<:Number, 3}, cwork::Vector{<:Complex})
	∇Ds, Δ = ∇dm_apply_diagonals!(Δ, Ds, y, cwork)
	return Δ, ∇Ds, y
end



function pure_apply_diagonals!(Ds::Vector{<:Complex}, x::AbstractVecOrMat, y::AbstractVecOrMat)
	# @assert length(Ds) == size(x, 1)
	for j in 1:size(x, 2)
		for i in 1:size(x, 1)
			y[i, j] = Ds[i] * x[i, j]
		end
	end
	return y
end

function dm_apply_diagonals!(Ds::Vector{<:Complex}, x::AbstractMatrix, y::AbstractMatrix)
	# @assert (length(Ds) == size(x, 1)) && (size(x, 1) == size(x, 2))
	for j in 1:size(x, 2)
		for i in 1:size(x, 1)
			y[i, j] = Ds[i] * x[i, j] * conj(Ds[j])
		end
	end
	return y
end
function dm_apply_diagonals!(Ds::Vector{<:Complex}, x::AbstractArray{<:Number, 3}, y::AbstractArray{<:Number, 3})
	# @assert (length(Ds) == size(x, 1)) && (size(x, 1) == size(x, 2))
	for k in 1:size(x, 3)
		dm_apply_diagonals!(Ds, view(x, :,:,k), view(y,:,:,k))
	end
	return y
end

function init_diagonal_cwork!(Ds, cwork::Vector{<:Complex})
	@assert length(cwork) >= length(Ds)
	L = length(Ds)
	for i in 1:L
		cwork[i] = exp(-im * Ds[i])
	end
end

function ∇pure_apply_diagonals_util!(Δ::AbstractVecOrMat, Ds::Vector{<:Real}, y::AbstractVecOrMat, cwork::Vector{<:Complex})
	L = length(Ds)
	Δ = convert(typeof(y), Δ)
	pure_apply_diagonals!(cwork, y, y)
	pure_apply_diagonals!(cwork, Δ, Δ)
	∇Ds = zero(Ds)
	for j in 1:size(y, 2)
		for i in 1:L
			∇Ds[i] += real(-im * conj(y[i, j]) * Δ[i, j])
		end
	end
	return ∇Ds, Δ
end
function ∇pure_apply_diagonals!(Δ::AbstractVecOrMat, Ds::Vector{<:Real}, y::AbstractVecOrMat, cwork::Vector{<:Complex})
	init_diagonal_cwork!(Ds, cwork)
	return ∇pure_apply_diagonals_util!(Δ, Ds, y, cwork)
end
function ∇pure_apply_diagonals!(Δ::AbstractArray{<:Number, 3}, Ds::Vector{<:Real}, y::AbstractArray{<:Number, 3}, cwork::Vector{<:Complex})
	init_diagonal_cwork!(Ds, cwork)
	Δ = convert(typeof(y), Δ)
	for k in 1:size(y, 3)
		∇pure_apply_diagonals_util!(view(Δ,:,:,k), Ds, view(y,:,:,k), cwork)
	end
	return ∇Ds, Δ
end

function ∇dm_apply_diagonals_util!(Δ::AbstractMatrix, Ds::Vector{<:Real}, y::AbstractMatrix, cwork::Vector{<:Complex}, ∇Ds)
	L = length(Ds)
	Δ = convert(typeof(y), Δ)	
	dm_apply_diagonals!(cwork, y, y)
	dm_apply_diagonals!(cwork, Δ, Δ)
	# ∇Ds = zero(Ds)
	for j in 1:size(y, 2)
		for i in 1:L
			∇Ds[i] += 2 * real(-im * conj(y[i, j]) * Δ[i, j])
		end
	end	
	return ∇Ds, Δ
end
function ∇dm_apply_diagonals!(Δ::AbstractMatrix, Ds::Vector{<:Real}, y::AbstractMatrix, cwork::Vector{<:Complex})
	init_diagonal_cwork!(Ds, cwork)
	return ∇dm_apply_diagonals_util!(Δ, Ds, y, cwork, zero(Ds))
end
function ∇dm_apply_diagonals!(Δ::AbstractArray{<:Number, 3}, Ds::Vector{<:Real}, y::AbstractArray{<:Number, 3}, cwork::Vector{<:Complex})
	init_diagonal_cwork!(Ds, cwork)
	Δ = convert(typeof(y), Δ)
	∇Ds = zero(Ds)
	for k in 1:size(y, 3)
		∇dm_apply_diagonals_util!(view(Δ,:,:,k), Ds, view(y,:,:,k), cwork, ∇Ds)
	end
	return ∇Ds, Δ
end




