module UnitaryMatrices


using Zygote
using LinearAlgebra

export RotationMatrix, pure_evolve, dm_evolve
export UnitaryMatrix, OrthogonalMatrix, parameters, nparameters


# const AbstractMatVec = Union{AbstractMatrix, AbstractVector}

abstract type AbstractUnitaryMatrix{T} end

# defnition of rotational matrices
include("rotational_matrices.jl")
include("diagonals.jl")

# definition of parametric unitary matrix
include("unitary_matrix.jl")
include("orthogonal_matrix.jl")



end