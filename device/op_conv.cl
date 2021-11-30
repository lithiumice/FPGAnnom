
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
// } nnom_qtype_t;

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

__kernel void
conv_hwc_dim1_q7(__global q7_t *Im_in, uint16_t dim_im_in_x, uint16_t dim_im_in_y,
            uint16_t ch_im_in, __global q7_t *wt, uint16_t ch_im_out,
            uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t padding_x,
            uint16_t padding_y, uint16_t stride_x, uint16_t stride_y,
            uint16_t dilation_x, uint16_t dilation_y, __global q7_t *bias,
            __global int *bias_shift, __global int *out_shift, int q_type,
            __global q7_t *Im_out, uint16_t dim_im_out_x, uint16_t dim_im_out_y,
            __global q15_t *bufferA, __global q7_t *bufferB) {
  int i, j, k, l, m, n;
  int conv_out;
  int in_row, in_col;
  int in_pix_loc, wt_loc;
  int shift_idx, shift_steps;
  if (q_type == 1)
    shift_steps = 1;
  else
    shift_steps = 0;

  int gx = get_global_id(0);

    // for (gx = 0; gx < dim_im_out_y; gx++) {
  for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps) {
      int32_t base_idx_y = stride_y * gx - padding_y;
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
            in_row = stride_y * gx + m * dilation_y - padding_y;
            in_col = stride_x * k + n * dilation_x - padding_x;

            // pre-calculate the pixel location and weight location to improve
            // the performance.
            in_pix_loc = (in_row * dim_im_in_x + in_col) * ch_im_in;
            wt_loc = i * ch_im_in * dim_kernel_y * dim_kernel_x +
                     (m * dim_kernel_x + n) * ch_im_in;

            for (l = 0; l < ch_im_in; l++) {
              conv_out += Im_in[in_pix_loc + l] * wt[wt_loc + l];
            }
          }
        }
        Im_out[i + (gx * dim_im_out_x + k) * ch_im_out] =
            (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
      }
    }
  // }
}

// __kernel void
// conv_hwc_q7(__global q7_t *Im_in, uint16_t dim_im_in_x, uint16_t dim_im_in_y,
//             uint16_t ch_im_in, __global q7_t *wt, uint16_t ch_im_out,
//             uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t padding_x,
//             uint16_t padding_y, uint16_t stride_x, uint16_t stride_y,
//             uint16_t dilation_x, uint16_t dilation_y, __global q7_t *bias,
//             __global int *bias_shift, __global int *out_shift, int q_type,
//             __global q7_t *Im_out, uint16_t dim_im_out_x, uint16_t dim_im_out_y,
//             __global q15_t *bufferA, __global q7_t *bufferB) {
//   int i, j, k, l, m, n;
//   int conv_out;
//   int in_row, in_col;
//   int in_pix_loc, wt_loc;
//   int shift_idx, shift_steps;
//   if (q_type == 1)
//     shift_steps = 1;
//   else
//     shift_steps = 0;

//   for (i = 0, shift_idx = 0; i < ch_im_out; i++, shift_idx += shift_steps) {
//     for (j = 0; j < dim_im_out_y; j++) {
//       int32_t base_idx_y = stride_y * j - padding_y;
//       for (k = 0; k < dim_im_out_x; k++) {
//         int32_t base_idx_x = stride_x * k - padding_x;
//         int32_t ker_y_start =
//             MAX(0, -(base_idx_y - (dilation_y - 1)) / dilation_y);
//         int32_t ker_x_start =
//             MAX(0, -(base_idx_x - (dilation_x - 1)) / dilation_x);
//         int32_t ker_y_end =
//             MIN(dim_kernel_y,
//                 (dim_im_in_y - base_idx_y + (dilation_y - 1)) / dilation_y);
//         int32_t ker_x_end =
//             MIN(dim_kernel_x,
//                 (dim_im_in_x - base_idx_x + (dilation_x - 1)) / dilation_x);

//         if (bias)
//           conv_out = ((q31_t)(bias[i]) << bias_shift[shift_idx]) +
//                      NNOM_ROUND(out_shift[shift_idx]);
//         else
//           conv_out = (q31_t)NNOM_ROUND(out_shift[shift_idx]);

//         for (m = ker_y_start; m < ker_y_end; m++) {
//           for (n = ker_x_start; n < ker_x_end; n++) {
//             in_row = stride_y * j + m * dilation_y - padding_y;
//             in_col = stride_x * k + n * dilation_x - padding_x;

//             // pre-calculate the pixel location and weight location to improve
//             // the performance.
//             in_pix_loc = (in_row * dim_im_in_x + in_col) * ch_im_in;
//             wt_loc = i * ch_im_in * dim_kernel_y * dim_kernel_x +
//                      (m * dim_kernel_x + n) * ch_im_in;

//             for (l = 0; l < ch_im_in; l++) {
//               conv_out += Im_in[in_pix_loc + l] * wt[wt_loc + l];
//             }
//           }
//         }
//         Im_out[i + (j * dim_im_out_x + k) * ch_im_out] =
//             (q7_t)__NNOM_SSAT((conv_out >> out_shift[shift_idx]), 8);
//       }
//     }
//   }
// }
