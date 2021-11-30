

#define q7_t char
#define q15_t short
#define q31_t int
#define q63_t long

#define int8_t char
#define int16_t short
#define int32_t int
#define int64_t long

#define NNOM_ROUND(out_shift) ((0x1 << out_shift) >> 1)
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

#define uint8_t unsigned char
#define uint16_t unsigned short
#define uint32_t unsigned int
#define uint64_t unsigned long

#define NNOM_ROUND(out_shift) ((0x1 << out_shift) >> 1)
#define __MY_SSAT(value) ((value < -(1<<7))?-(1<<7):((value > ((1<<7)-1))?((1<<7)-1):value))

static inline int __NNOM_SSAT(int value, int bit) {
  int min = -(1 << (bit - 1));
  int max = (1 << (bit - 1)) - 1;
  if (value < min)
    return min;
  else if (value > max)
    return max;
  else
    return value;
}

__kernel void fc_q7(__global q7_t *pV,    // pointer to vector
                    __global q7_t *pM,    // pointer to matrix
                    uint16_t dim_vec,     // length of the vector
                    uint16_t num_of_rows, // numCol of A
                    uint16_t bias_shift,  // amount of left-shift for bias
                    uint16_t out_shift,   // amount of right-shift for output
                    __global q7_t *bias,  // bias
                    __global q7_t *pOut,  // output operand
                    __global q15_t *vec_buffer // buffer
) {
  for (int i = 0; i < num_of_rows; i++) {
    int ip_out = 0;

    if (bias)
      ip_out = ((q31_t)(*bias++) << bias_shift) + NNOM_ROUND(out_shift);
    else
      ip_out = (q31_t)NNOM_ROUND(out_shift);

    for (int j = 0; j < dim_vec; j++) {
      ip_out += pV[j] * pM[i * dim_vec + j];
    }
    pOut[i] = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);
  }
}

__kernel void fc_dim1_q7(__global q7_t *pV,    // pointer to vector
                    __global q7_t *pM,    // pointer to matrix
                    uint16_t dim_vec,     // length of the vector
                    uint16_t num_of_rows, // numCol of A
                    uint16_t bias_shift,  // amount of left-shift for bias
                    uint16_t out_shift,   // amount of right-shift for output
                    __global q7_t *bias,  // bias
                    __global q7_t *pOut,  // output operand
                    __global q15_t *vec_buffer // buffer
) {
  int gx = get_global_id(0);

     int ip_out = (q31_t)NNOM_ROUND(out_shift);
  // int ip_out = ((int)(bias[gx]) << bias_shift) + (int)((0x1 << out_shift) >> 1);

  for (int j = 0; j < dim_vec; j++) {
    ip_out += pV[j] * pM[gx * dim_vec + j];
  }
// value= (value < -(1<<7))?-(1<<7):((value > ((1<<7)-1))?((1<<7)-1):value)
  // pOut[gx] = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);
    pOut[gx] = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);
  // pOut[gx] = (q7_t)__MY_SSAT(ip_out >> out_shift);

}

//   __kernel void fc_dim2_q7(__global q7_t *pV, __global q7_t *pM,
//                     __global q15_t *vec_buffer, __global q7_t *bias,
//                     __global q7_t *pOut, int dim_vec, int num_of_rows,
//                     int bias_shift, int out_shift) {

//   int gx = get_global_id(0);//num_of_rows
//   int sx = get_global_size(0);

//   int gy = get_global_id(1);//dim_vec
//   int sy = get_global_size(1);

//   int s = sx * sy;

//   int ip_out = ((int)(bias[gx]) << bias_shift) + (int)((0x1 << out_shift) >> 1);
  
//   for (int j = 0; j < dim_vec; j++) {
//     ip_out += pV[j] * pM[gx * dim_vec + j];
//   }

//   pOut[gx] = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);
// }
