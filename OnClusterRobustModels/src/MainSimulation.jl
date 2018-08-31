#=
    Author: José Bayoán Santiago Calderón (@Nosferican)
=#

## Objective

#=
    This simulation will explore the behavior of the pooling estimator with heterogeneity in treatment effects.
=#

## Set up the environment for this application

using Pkg: activate
activate(joinpath(@__DIR__, ".."))
using Distributions: Normal, TDist
using LinearAlgebra: cholesky!, diag, inv, Hermitian
using LinearAlgebra.BLAS: ger!
using Plots: plot, histogram
using StatPlots: StatPlots, @df
using Random: seed!
using Statistics: mean, quantile
using StatsBase: mean_and_std, sample, shuffle, weights, summarystats
using DataFrames: DataFrame, stack

import Base: rand

## Data Generation Process

#=
The data generation process follows

y = α + x₁ ( β₁ + γⱼ ) + x₂ ( β₂ ) + u

where y and u are standard normal distributed variables, α an intercept term, x₁ and x₂ exogenous explanatory variables and β₁, β₂, and γᵢ capture the relation between the the features and the outcome variable. x₁ is a heterogenous in treatment discrete dimension since the partial effect is cluster determined. x₂ is a homogenous in treatment dimension since the partial effect is constant for all observations.
=#

### Generating the population and the average treatment effect

#=WLOG:

- The population is of size: 1e5
- There are five clusters for the x₁ dimension each with equal probability
- x₁ and x₂ are standard normal distributed
- β₁ = 1e0 and β₂ = 5e-1
=#

seed!(0)
const X = hcat(ones(Int(1e5)), rand(Normal(), Int(1e5), 2))
const subgroups = sort(rand(-0.5:0.25:0.5, Int(1e5)))
const y = vec(sum(X[:,2:end] .* reduce(vcat, hcat.(1 .+ subgroups, 0.5)),
                  dims = 2) + rand(Normal(), length(subgroups)))
const ID = mapreduce(idx -> repeat([idx], inner = Int(1e2)), vcat, 1:Int(1e3))
const β = (X \ y)[2:end]

### Samping Design

#=
The sampling design describes the way observations from the population are observed. The two designs we consider are random (i.e., every observations has equal sampling probability) and clustered (i.e., probability of observations being sampled are cluster dependent).

WLOG:

- The random sampling design has a probability of 1e-1.
- The clustered sampling design have distinct probabilities drawed without replacement from `1e-1:5e-2:3e-1`
=#

#=
NOTE:
Stratified sampling design when there is heterogeneity in treatment effects results in clustered sampling design.
=#

"""
    SamplingDesign

Abstract type for sampling designs.
"""
abstract type SamplingDesign end
"""
    RandomSampling

Struct for random sampling (i.e., every observation has equal probability of being observed). The sampling probability is given by `p`.
"""
struct RandomSampling <: SamplingDesign
    p::Float64
    function RandomSampling(p)
        zero(p) ≤ p ≤ one(p) || throw(ArgumentError("p ∉ [0, 1]"))
        return new(p)
    end
end
"""
    ClusteredSampling

Struct for clustered sampling (i.e., the sampling probability of an observation is cluster dependent). The sampling probability for each cluster is given by `p`.
"""
struct ClusteredSampling{K} <: SamplingDesign
    p::Dict{K,Float64}
    function ClusteredSampling(p::Dict{K,Float64}) where {K<:Any}
        all(elem -> zero(elem) ≤ elem ≤ one(elem), values(p)) ||
        throw(ArgumentError("p contains non valid probability elements"))
        return new{K}(p)
    end
end
"""
    rand(design::RandomSampling, sample_size::Integer)

Samples using `RandomSampling`.
"""
function rand(design::RandomSampling, sample_size::Integer)
    p = getfield(design, :p)
    return sample([true, false], weights([p, one(p) - p]), sample_size)
end
"""
    rand(design::ClusteredSampling, subgroups::AbstractVector)

Samples using `ClusteredSampling`.
"""
function rand(design::ClusteredSampling, subgroups::AbstractVector)
    prob_vector = getfield(design, :p)
    groups = Set(subgroups)
    keys(prob_vector) ⊆ groups || throw(ArgumentError("sampling design must specify probabilities for each cluster in subgroups"))
    obs = BitVector(undef, length(subgroups))
    for (k, v) ∈ prob_vector
        tmp = subgroups .== k
        obs[tmp] = sample([true, false], weights([v, one(v) - v]), sum(tmp))
    end
    return obs
end


## Trial

#=
- Every trial will sample from the population using a sampling design
- It will fit the model using the various estimators
- Report the distribution of parameter estimates (i.e., first moments)
- Report empirical coverage rates (i.e., second moments) for hypothesis testing
=#

gram(obj::AbstractVecOrMat) = transpose(obj) * obj
gram(obj::AbstractVecOrMat{<:Real}) = Hermitian(transpose(obj) * obj)
ewh(X::AbstractMatrix, û::AbstractVector) = gram(abs.(û) .* X) # X' * (û * û.' .* I) * X
function lz(X::AbstractMatrix, û::AbstractVector, clusters::AbstractVector{<:AbstractVector{<:Integer}})
    output = zeros(size(X, 2), size(X, 2))

    for cluster ∈ clusters
        output += gram(transpose(û[cluster]) * X[cluster,:]) # X[cluster,:].' * û[cluster] * û[cluster].' * X[cluster,:]
    end
    return output
end
outerprod(obj) = ger!(one(Float64), obj, obj, zeros(length(obj), length(obj))) # obj * obj.'
function solve(linearpredictor::AbstractMatrix, response::AbstractVector)
    IM = inv(cholesky!(gram(linearpredictor)))
    β = IM * transpose(linearpredictor) * response
    β, IM
end
"""
    sampling(linear_predictor::AbstractMatrix{<:Real},
             random_component::AbstractVector{<:Real};
             design::SamplingDesign = RandomSampling(1e-1),
             subgroups::Union{AbstractVector, Nothing} = nothing)

Return: (β̂::AbstractVector{<:Real},
         Ω̂::AbstractMatrix{<:Real},
         obs::AbstractVector{<:Bool})

Samples from a population defined by a linear predictor and random component. If using a `ClusteredSampling`, one must provide a one dimensional discrete cluster structure.
"""
function sampling(linear_predictor::AbstractMatrix{<:Real},
                  random_component::AbstractVector{<:Real};
                  design::SamplingDesign = RandomSampling(1e-1),
                  subgroups::Union{AbstractVector, Nothing} = subgroups)
    if isa(design, ClusteredSampling)
        subgroups isa AbstractVector ||
        throw(ArgumentError("a ClusteredSampling requires a one dimensional cluster structure given by subgroups"))
    end
    m, n = size(linear_predictor)
    m == length(random_component) || throw(DimensionMismatch("linear predictor and random componen must have same number of observations"))
    if design isa RandomSampling
        obs = rand(design, length(random_component))
    elseif design isa ClusteredSampling
        obs = rand(design, subgroups)
    end
    lp = linear_predictor[obs,:]
    response = random_component[obs]
    return (solve(lp, response)..., obs)
end

## Sampling

βrandom = mapreduce(t -> sampling(X, y)[1][2:end],
                    hcat,
                    1:Int(1e2))
mean(βrandom, dims = 2)

βclustered = mapreduce(t -> sampling(X,
                                     y,
                                     design = zip(-5e-1:2.5e-1:5e-1,
                                                  shuffle(1e-1:5e-2:3e-1)) |>
                                              Dict |>
                                              ClusteredSampling,
                                     subgroups = subgroups)[1][2:end],
                       hcat,
                       1:Int(1e2))
mean(βclustered, dims = 2)

### Summary

println(mean_and_std(βrandom, 2))
println(mean_and_std(βclustered, 2))

function trial(linear_predictor::AbstractMatrix{<:Real},
               random_component::AbstractVector{<:Real},
               β::AbstractVector;
               sampling_option::SamplingDesign = RandomSampling(1e-1),
               subgroups::Union{AbstractVector, Nothing} = subgroups)
    β̂, IM, obs = sampling(linear_predictor,
                          random_component,
                          design = sampling_option,
                          subgroups = subgroups)
    lp = X[obs,:]
    response = y[obs]
    m = length(response)
    û = response - lp * β̂

    rdf = reduce(-, size(lp))
    EWH = m / rdf * IM * ewh(lp, û) * IM
    tstar = quantile(TDist(rdf), 1 - 5e-2 / 2)
    movement = tstar * sqrt.(diag(EWH)[2:end])

    LB = β̂[2:end] - movement

    UB = β̂[2:end] + movement
    EWH = LB .≤ β .≤ UB

    g = length(unique(ID[obs]))
    LZ = g / (g - 1) * (m - 1) / rdf * IM * lz(lp, û, map(group -> findall(group .== ID[obs]), unique(ID[obs]))) * IM
    tstar = quantile(TDist(g - 1), 1 - 5e-2 / 2)
    movement = tstar * sqrt.(diag(LZ)[2:end])

    LB = β̂[2:end] - movement

    UB = β̂[2:end] + movement
    LZ = LB .≤ β .≤ UB

    g = length(unique(subgroups[obs]))
    JBSC = g / (g - 1) * (m - 1) / rdf * IM * lz(lp, û, map(group -> findall(group .== subgroups[obs]), unique(subgroups[obs]))) * IM
    tstar = quantile(TDist(rdf), 1 - 5e-2 / 2)
    movement = tstar * sqrt.(diag(JBSC)[2:end])

    LB = β̂[2:end] - movement

    UB = β̂[2:end] + movement
    JBSC = LB .≤ β .≤ UB

    output = vcat(EWH, LZ, JBSC)
end

seed!(0)
results_random = mapreduce(itr -> trial(X, y, vec(mean(βrandom, dims = 2))),
                           hcat,
                           1:Int(1e1))
mean(results_random, dims = 2)

seed!(0)
results_cluster = mapreduce(itr -> trial(X,
                                         y,
                                         vec(mean(βclustered, dims = 2)),
                                         sampling_option = zip(-5e-1:2.5e-1:5e-1,
                                             shuffle(1e-1:5e-2:3e-1)) |>
                                             Dict |>
                                             ClusteredSampling,
                                         subgroups = subgroups),
                            hcat,
                            1:Int(1e1))
mean(results_cluster, dims = 2)

mapslices(summarystats, βrandom, dims = 2)
mapslices(summarystats, βclustered, dims = 2)

toexp = transpose(vcat(βrandom, βclustered))
df = stack(DataFrame(toexp), 1:4)
df[:Condition] = ifelse.(map(elem -> elem ∈ [:x1,:x4], df[:variable]),
                         "Random",
                         "Clustered")
p = @df df density(:value,
                   group  = (:Condition),
                   fill   = (.25, 0),
                   legend = :topright,
                   xlab = "Parameter Estimates",
                   ylab = "Density",
                   guidefont = font(10, "Times New Roman"),
                   xtickfont = font(8, "Times New Roman"),
                   ytickfont = font(8, "Times New Roman"))

df = DataFrame()
df[:y] = y
df[:X1] = X[:,2]
df[:X2] = X[:,3]
df[:G] = get.(levelsmap(subgroups), subgroups, 0)
categorical!(df, :G)
head(df)

mf = ModelFrame(@formula(y ~ X1 & G + X2), df)
mm = ModelMatrix(mf).m
β = mm \ y
hcat(coefnames(mf), β)


βm = zeros(Int(1e4), 6)
for t ∈ 1:size(βstratified, 1)
    βm[t,:] = trial_m("Clustered")
end
mean(βm, 1)

std(βm, 1)

β = β[2:end]

seed!(0)
results_cluster_mod = zeros(Int(1e3), 18)
for t ∈ 1:size(results_cluster_mod, 1)
    results_cluster_mod[t,:] = trial_mod("Clustered", β)
end
mean(results_cluster_mod, 1)

mean(results_cluster_mod, 1)[1:6]

for each ∈ 1:6:18
    println(mean(results_cluster_mod[:,each:(each + 5)], 1))
end
