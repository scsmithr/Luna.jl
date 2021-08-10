using Flux
using Flux: glorot_uniform

"""
    Attention

A traditional attention mechanism.
"""
struct Attention{F, M<:AbstractMatrix}
    Wq::M
    Wk::M # TODO: eliminate
    Wv::M
    ω::F
end

Flux.functor(a::Attention) = (a.Wq, a.Wk, a.Wv), ps -> Attention(ps..., a.ω)

function Attention(d::Int, ω::F, init = glorot_uniform) where F
    return Attention(
        init(d, d),
        init(d, d),
        init(d, d),
        ω)
end

function (a::Attention)(X::M, C::M) where {T, M<:AbstractMatrix{T}}
    dk = size(a.Wq, 1)
    Q = X * a.Wq
    K = C * a.Wk
    V = C * a.Wv
    # TODO: causal
    A = a.ω( (Q * K')/sqrt(dk) )
    return A * V
end

struct FFN
    in::Dense
    out::Dense
end

Flux.@functor FFN

function FFN(d::Int, h::Int)
    return FFN(Dense(d, h),
               Dense(h, d))
end

(ffn::FFN)(X::AbstractMatrix) = ffn.out(ffn.in(X))

struct LunaAttention
    pack::Attention
    unpack::Attention
    packnorm::LayerNorm
    unpacknorm::LayerNorm
    ffn::FFN
    ffnorm::LayerNorm
end

Flux.@functor LunaAttention

function LunaAttention(d::Int, h::Int)
    return LunaAttention(Attention(d, identity), # TODO: elu
                         Attention(d, softmax),
                         LayerNorm(d),
                         LayerNorm(d),
                         FFN(d, h),
                         LayerNorm(d))
end

function (a::LunaAttention)(X::M, P::M, C::M) where {T, M<:AbstractMatrix{T}}
    Yp = a.pack(P, C)
    Yx = a.unpack(X, Yp)
    Pa = a.packnorm(Yp + P)
    Xa = a.unpacknorm(Yx + X)
    X_out = a.ffnorm( a.ffn(Xa)+Xa )
    return (X_out, Pa)
end
