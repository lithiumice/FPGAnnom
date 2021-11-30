
#define int8_t char
#define int16_t short
#define int32_t int
#define int64_t long

#define uint8_t unsigned char
#define uint16_t unsigned short
#define uint32_t unsigned int
#define uint64_t unsigned long

// #define q7_t cl_char
// #define q15_t cl_short
// #define q31_t cl_int
// #define q63_t cl_long

#define q7_t char
#define q15_t short
#define q31_t int
#define q63_t long

#define NNOM_ROUND(out_shift) ((0x1 << out_shift) >> 1)
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

// basic types
#define nnom_qformat_param_t int32_t
// this should match the backend, need a better way to do it.
#define nnom_shape_data_t uint16_t

typedef enum
{
    NNOM_QTYPE_PER_TENSOR = 0,
    NNOM_QTYPE_PER_AXIS = 1
} nnom_qtype_t;

static inline int __NNOM_SSAT(int value, int bit)
{
    int min = -(1 << (bit - 1));
    int max = (1 << (bit - 1)) - 1;
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}
