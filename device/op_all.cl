
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

// #define nnom_qformat_param_t int
// typedef enum {
//   NNOM_QTYPE_PER_TENSOR = 0,
//   NNOM_QTYPE_PER_AXIS = 1
// } nnom_qtype_t;//int

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

__kernel void conv_hwc_q7(
    __global q7_t *Im_in,     // input image
    uint16_t dim_im_in_x,     // input image dimention x
    uint16_t dim_im_in_y,     // input image dimention y
    uint16_t ch_im_in,        // number of input image channels
    __global q7_t *wt,        // kernel weights
    uint16_t ch_im_out,       // number of filters, i.e., output image channels
    uint16_t dim_kernel_x,    // filter kernel size x
    uint16_t dim_kernel_y,    // filter kernel size y
    uint16_t padding_x,       // padding sizes x
    uint16_t padding_y,       // padding sizes y
    uint16_t stride_x,        // stride x
    uint16_t stride_y,        // stride y
    uint16_t dilation_x,      // dilation x
    uint16_t dilation_y,      // dilation y
    __global q7_t *bias,      // bias
    __global int *bias_shift, // bias shifts
    __global int *out_shift,  // output shift
    int q_type,               // per channel or per tensor
    __global q7_t *Im_out,    // output image
    uint16_t dim_im_out_x,    // output image dimension x
    uint16_t dim_im_out_y,    // output image dimension y
    __global q15_t *bufferA,  // buffer space for input
    __global q7_t *bufferB    // buffer space for output
) {
  int i, j, k, l, m, n;
  int conv_out;
  int in_row, in_col;
  int in_pix_loc, wt_loc;
  int shift_idx, shift_steps;
  if (q_type == 1)
    shift_steps = 1;
  else
    shift_steps = 0;

  for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps) {
    for (j = 0; j < dim_im_out_y; j++) {
      int32_t base_idx_y = stride_y * j - padding_y;
      for (k = 0; k < dim_im_out_x; k++) {
        int32_t base_idx_x = stride_x * k - padding_x;
        int32_t ker_y_start =
            MAX(0, -(base_idx_y - (dilation_y - 1)) / dilation_y);
        int32_t ker_x_start =
            MAX(0, -(base_idx_x - (dilation_x - 1)) / dilation_x);
        int32_t ker_y_end =
            MIN(dim_kernel_y,
                (dim_im_in_y - base_idx_y + (dilation_y - 1)) / dilation_y);
        int32_t ker_x_end =
            MIN(dim_kernel_x,
                (dim_im_in_x - base_idx_x + (dilation_x - 1)) / dilation_x);

        if (bias)
          conv_out = ((q31_t)(bias[i]) << bias_shift[shift_idx]) +
                     NNOM_ROUND(out_shift[shift_idx]);
        else
          conv_out = (q31_t)NNOM_ROUND(out_shift[shift_idx]);

        for (m = ker_y_start; m < ker_y_end; m++) {
          for (n = ker_x_start; n < ker_x_end; n++) {
            in_row = stride_y * j + m * dilation_y - padding_y;
            in_col = stride_x * k + n * dilation_x - padding_x;
            in_pix_loc = (in_row * dim_im_in_x + in_col) * ch_im_in;
            wt_loc = i * ch_im_in * dim_kernel_y * dim_kernel_x +
                     (m * dim_kernel_x + n) * ch_im_in;

            for (l = 0; l < ch_im_in; l++) {
              conv_out += Im_in[in_pix_loc + l] * wt[wt_loc + l];
            }
          }
        }
        Im_out[i + (j * dim_im_out_x + k) * ch_im_out] =
            (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
      }
    }
  }
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

// __kernel void fc_q7(__global q7_t *pV, __global q7_t *pM,
//                     __global q15_t *vec_buffer, __global q7_t *bias,
//                     __global q7_t *pOut, int dim_vec, int num_of_rows,
//                     int bias_shift, int out_shift) {

//   for (int i = 0; i < num_of_rows; i++) {
//     int ip_out = ((int)(*bias++) << bias_shift) + (int)NNOM_ROUND(out_shift);
//     for (int j = 0; j < dim_vec; j++) {
//       ip_out += pV[j] * pM[i * dim_vec + j];
//     }
//     pOut[i] = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);
//   }
// }
