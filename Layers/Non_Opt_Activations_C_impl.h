#include <stdint.h>
static inline float relu_activate(float x){return x*(x>0);}
static inline float relu_gradient(float x){return (x>0);}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}

static inline float sigmoid_activate(float x){return  1 / (1 + exp(0 - x));}
static inline float sigmoid_gradient(float x){return (1 - sigmoid_activate(x))*sigmoid_activate(x);}

void batchReluForward(const float* input,float* output,const int32_t* size);
void batchReluBackward(const float* input,float* output,const int32_t* size);

void batchLeakyForward(const float* input,float* output,const int32_t* size);
void batchLeakyBackward(const float* input,float* output,const int32_t* size);

void batchSigmoidForward(const float* input,float* output,const int32_t* size);
void batchSigmoidBackward(const float* input,float* output,const int32_t* size);