using Flux
using Flux: glorot_uniform

"""
    Attention

A traditional scaled dot product attention mechanism.
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

elum(X::AbstractArray) = elu.(X)

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

"""
    LunaAttention

A Luna attention layer composed to two Attention layers with linear efficiency.
"""
struct LunaAttention
    pack::Attention
    unpack::Attention
end

Flux.@functor LunaAttention

LunaAttention(d::Int) = LunaAttention(Attention(d, elum),
                                      Attention(d, softmax))

function (la::LunaAttention)(X::M, P::M, C::M) where {T, M<:AbstractMatrix{T}}
    Yp = la.pack(P, C)
    Yx = la.unpack(X, Yp)
    return (Yp, Yx)
end

struct LunaEncoderBlock
    attn::LunaAttention
    linnorm::LayerNorm
    attnnorm::LayerNorm
    ffn::FFN
    ffnorm::LayerNorm
end

Flux.@functor LunaEncoderBlock

function LunaEncoderBlock(d::Int, h::Int)
    return LunaEncoderBlock(LunaAttention(d),
                            LayerNorm(d),
                            LayerNorm(d),
                            FFN(d, h),
                            LayerNorm(d))
end

function (e::LunaEncoderBlock)(X::M, P::M) where {T, M<:AbstractMatrix{T}}
    C = X
    (Yp, Yx) = e.attn(X, P, C)
    Pa = e.linnorm(Yp + P)
    Xa = e.attnnorm(Yx + X)
    X_out = e.ffnorm(e.ffn(Xa) + Xa)
    return (X_out, Pa)
end

struct LunaDecoderBlock
    maskattn::Attention # TODO: Proper mask
    masknorm::LayerNorm
    attn::LunaAttention
    linnorm::LayerNorm
    attnnorm::LayerNorm
    ffn::FFN
    ffnorm::LayerNorm
end

Flux.@functor LunaDecoderBlock

function LunaDecoderBlock(d::Int, h::Int)
    return LunaDecoderBlock(Attention(d, softmax),
                            LayerNorm(d),
                            LunaAttention(d),
                            LayerNorm(d),
                            LayerNorm(d),
                            FFN(d, h),
                            LayerNorm(d))
end

function (d::LunaDecoderBlock)(X::M, P::M, E::M) where {T, M<:AbstractMatrix{T}}
    C = X
    X_m = d.maskattn(X, C)
    X_n = d.masknorm(X_m)
    (Yp, Yx) = d.attn(X_n, P, E)
    Pa = d.linnorm(Yp + P)
    Xa = d.linnorm(Yx + X_n)
    X_out = d.ffnorm(e.ffn(Xa) + Xa)
end
