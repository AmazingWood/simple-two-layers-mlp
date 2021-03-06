#include "OpInterfaces.hpp"
namespace RLDNN
{
template <typename Precision, size_t Rank, Device dev=Device::CPU>
class RelULayer
{
private:
    Tensor<bool, Rank> mask;
public:
    RelULayer() = default;
    ~RelULayer() = default;
    Tensor<Precision, Rank> forward(const OpInOutType<Precision, Rank> &args)
    {
        const Tensor<Precision, Rank> &x = args.at("x");
        this->mask = x > 0.f;
        Tensor<Precision, Rank> zeros(x.dimensions());
        zeros.setZero();
        Tensor<Precision, Rank> out = mask.select(x, zeros);
        return out;
    }
    OpInOutType<Precision, Rank> backward(const Tensor<Precision, Rank> &dout)
    {
        OpInOutType<Precision, Rank> gradient;
        gradient["dx"] = mask.template cast<Precision>();
        return gradient;
    }
};
static_assert(OpValidation<RelULayer<float, 4>>::valid, OP_CONCEPT_ERR);

template <typename Precision, size_t Rank, Device dev=Device::CPU>
class SigmoidLayer{
private:
    Tensor<Precision, Rank> out;
public:
    SigmoidLayer()=default;
    ~SigmoidLayer()=default;
        Tensor<Precision, Rank> forward(const OpInOutType<Precision, Rank> &args)
    {
        auto negIn = -args.at("x");
        this->out = 1.f / (1.f + negIn.exp());
        return out;
    }
    OpInOutType<Precision, Rank> backward(const Tensor<Precision, Rank> &dout)
    {
        OpInOutType<Precision, Rank> gradient;
        gradient["dx"] = dout * (1.0 - this->out) * this->out;
        return gradient;
    }
};
static_assert(OpValidation<SigmoidLayer<float, 4>>::valid, OP_CONCEPT_ERR);



} // namespace RLDNN
