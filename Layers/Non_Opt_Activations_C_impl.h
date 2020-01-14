#include <stdint.h>
static inline float relu_activate(float x){return x*(x>0);}
static inline float relu_gradient(float x){return (x>0);}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}

void batchReluForward(const float* input,float* output,const int32_t* size);
void batchReluBackward(const float* input,float* output,const int32_t* size);

void batchLeakyForward(const float* input,float* output,const int32_t* size);
void batchLeakyBackward(const float* input,float* output,const int32_t* size);