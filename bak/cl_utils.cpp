#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nnom.h"
#include "cl_utils.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

unsigned num_devices = 0;
cl_platform_id platform = NULL;
cl_context context = NULL;
cl_program program = NULL;

scoped_array<cl_device_id> device;
scoped_array<cl_command_queue> queue;
scoped_array<cl_kernel> my_kernel;

scoped_array<cl_mem> input_pV_buf;
scoped_array<cl_mem> input_pM_buf;
scoped_array<cl_mem> input_bias_buf;
scoped_array<cl_mem> input_vec_buffer;
scoped_array<cl_mem> output_pOut_buf;

// scoped_array<scoped_aligned_ptr<float>> input_a, input_b;
// scoped_array<scoped_aligned_ptr<float>> output;
// scoped_array<scoped_array<float>> ref_output;
// scoped_array<unsigned> n_per_device;
#define q7_t int8_t
#define q15_t int16_t
#define q31_t int32_t
#define q63_t int64_t

extern "C"
{
#if 0
void local_fully_connected_q7(const q7_t *pV,               // pointer to vector
                              const q7_t *pM,               // pointer to matrix
                              const uint16_t dim_vec,       // length of the vector
                              const uint16_t num_of_rows,   // numCol of A
                              const uint16_t bias_shift,    // amount of left-shift for bias
                              const uint16_t out_shift,     // amount of right-shift for output
                              const q7_t *bias, q7_t *pOut, // output operand
                              q15_t *vec_buffer)
{
    printf("dim_vec:%d\n",dim_vec);
    printf("num_of_rows:%d\n",num_of_rows);
    printf("bias_shift:%d\n",bias_shift);
    printf("out_shift:%d\n",out_shift);
    
    // printf("pV:\n");
    //     for (int j = 0; j < dim_vec; j++)
    //     {
    //         printf("%d ", pV[j]);
    //     }
    //     printf("\n");
    

    // printf("pM:\n");
    // for (int i = 0; i < num_of_rows; i++)
    // {
    //     for (int j = 0; j < dim_vec; j++)
    //     {
    //         printf("%d ", pM[i * dim_vec + j]);
    //     }
    //     printf("\n");
    // }
        const double start_time = getCurrentTimestamp();
    
 if (bias)
    {
        for (int i = 0; i < num_of_rows; i++)
        {
            int ip_out = ((q31_t)(*bias++) << bias_shift) + NNOM_ROUND(out_shift);
            for (int j = 0; j < dim_vec; j++)
            {
                ip_out += pV[j] * pM[i * dim_vec + j];
            }
            pOut[i] = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);
        }
    }
    else
    {
        for (int i = 0; i < num_of_rows; i++)
        {
            int ip_out = (q31_t)NNOM_ROUND(out_shift);
            for (int j = 0; j < dim_vec; j++)
            {
                ip_out += pV[j] * pM[i * dim_vec + j];
            }
            pOut[i] = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);
        }
    }
    
        const double end_time = getCurrentTimestamp();
        printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

    printf("pOut:\n");
        for (int j = 0; j < dim_vec; j++)
        {
            printf("%d ", pOut[j]);
        }
        printf("\n");
}
#else
    void local_fully_connected_q7(const q7_t *pV,               // pointer to vector
                                  const q7_t *pM,               // pointer to matrix
                                  const uint16_t dim_vec,       // length of the vector
                                  const uint16_t num_of_rows,   // numCol of A
                                  const uint16_t bias_shift,    // amount of left-shift for bias
                                  const uint16_t out_shift,     // amount of right-shift for output
                                  const q7_t *bias, q7_t *pOut, // output operand
                                  q15_t *vec_buffer)
    {
        //   init_opencl();

        cl_int status;

        // for (unsigned i = 0; i < num_devices; ++i)
        // {

        unsigned i = 0;
        printf("clCreateBuffer input_pV_buf\n");
        input_pV_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         dim_vec * sizeof(q7_t), NULL, &status);
        checkError(status, "Failed to create buffer for input_pV_buf");

        printf("clCreateBuffer input_pM_buf\n");
        input_pM_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                         dim_vec * num_of_rows * sizeof(q7_t), NULL, &status);
        checkError(status, "Failed to create buffer for input_pM_buf");

        printf("clCreateBuffer input_bias_buf\n");
        input_bias_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                           num_of_rows * sizeof(q7_t), NULL, &status);
        checkError(status, "Failed to create buffer for input_bias_buf");

        printf("clCreateBuffer output_pOut_buf\n");
        output_pOut_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            num_of_rows * sizeof(q7_t), NULL, &status);
        checkError(status, "Failed to create buffer for output_pOut_buf");

        printf("clCreateBuffer input_vec_buffer\n");
        input_vec_buffer[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             sizeof(q7_t), NULL, &status);
        checkError(status, "Failed to create buffer for output");

        const double start_time = getCurrentTimestamp();
        scoped_array<cl_event> kernel_event(num_devices);
        scoped_array<cl_event> finish_event(num_devices);

        // for (unsigned i = 0; i < num_devices; ++i)
        // {
        i = 0;
        cl_event write_event[3];
        printf("clEnqueueWriteBuffer input_pV_buf\n");
        status = clEnqueueWriteBuffer(queue[i], input_pV_buf[i], CL_FALSE,
                                      0, dim_vec * sizeof(q7_t), pV, 0, NULL, &write_event[0]);
        checkError(status, "Failed to transfer pV");

        printf("clEnqueueWriteBuffer input_pM_buf\n");
        status = clEnqueueWriteBuffer(queue[i], input_pM_buf[i], CL_FALSE,
                                      0, dim_vec * num_of_rows * sizeof(q7_t), pM, 0, NULL, &write_event[1]);
        checkError(status, "Failed to transfer pM");

        printf("clEnqueueWriteBuffer input_bias_buf\n");
        status = clEnqueueWriteBuffer(queue[i], input_bias_buf[i], CL_FALSE,
                                      0, num_of_rows * sizeof(q7_t), bias, 0, NULL, &write_event[2]);
        checkError(status, "Failed to transfer bias");

        unsigned argi = 0;


        // printf("clSetKernelArg input_pV_buf\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_pV_buf[i]);
        // checkError(status, "Failed to set argument %d", argi - 1);

        // printf("clSetKernelArg input_pM_buf\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_pM_buf[i]);
        // checkError(status, "Failed to set argument %d", argi - 1);

        // printf("clSetKernelArg input_vec_buffer\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_vec_buffer[i]);
        // checkError(status, "Failed to set argument %d", argi - 1);

        // printf("clSetKernelArg input_bias_buf\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_bias_buf[i]);
        // checkError(status, "Failed to set argument %d", argi - 1);

        // printf("clSetKernelArg output_pOut_buf\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &output_pOut_buf[i]);
        // checkError(status, "Failed to set argument %d", argi - 1);

        // printf("clSetKernelArg dim_vec\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(int), &dim_vec);
        // checkError(status, "Failed to set argument %d", argi - 1);

        // printf("clSetKernelArg num_of_rows\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(int), &num_of_rows);
        // checkError(status, "Failed to set argument %d", argi - 1);

        // printf("clSetKernelArg bias_shift\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(int), &bias_shift);
        // checkError(status, "Failed to set argument %d", argi - 1);

        // printf("clSetKernelArg out_shift\n");
        // status = clSetKernelArg(my_kernel[i], argi++, sizeof(int), &out_shift);
        // checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg input_pV_buf\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_pV_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg input_pM_buf\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_pM_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg dim_vec\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(uint16_t), &dim_vec);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg num_of_rows\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(uint16_t), &num_of_rows);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg bias_shift\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(uint16_t), &bias_shift);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg out_shift\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(uint16_t), &out_shift);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg input_bias_buf\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_bias_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg output_pOut_buf\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &output_pOut_buf[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        printf("clSetKernelArg input_vec_buffer\n");
        status = clSetKernelArg(my_kernel[i], argi++, sizeof(cl_mem), &input_vec_buffer[i]);
        checkError(status, "Failed to set argument %d", argi - 1);

        // const size_t global_work_size = 1;
        const size_t global_work_size = num_of_rows;
        printf("Launching for device %d (%zd elements)\n", i, global_work_size);

        printf("clEnqueueNDRangeKernel\n");
        status = clEnqueueNDRangeKernel(queue[i], my_kernel[i], 1, NULL,
                                        &global_work_size, NULL, 3, write_event, &kernel_event[i]);
        checkError(status, "Failed to launch my_kernel");

        printf("clEnqueueReadBuffer\n");
        status = clEnqueueReadBuffer(queue[i], output_pOut_buf[i], CL_FALSE,
                                     0, num_of_rows * sizeof(q7_t), pOut, 1, &kernel_event[i], &finish_event[i]);

        clReleaseEvent(write_event[0]);
        clReleaseEvent(write_event[1]);
        clReleaseEvent(write_event[2]);
        // }

        clWaitForEvents(num_devices, finish_event);

        const double end_time = getCurrentTimestamp();
        printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);
        for (unsigned i = 0; i < num_devices; ++i)
        {
            cl_ulong time_ns = getStartEndTime(kernel_event[i]);
            printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
        }

        for (unsigned i = 0; i < num_devices; ++i)
        {
            clReleaseEvent(kernel_event[i]);
            clReleaseEvent(finish_event[i]);
        }

        // cleanup();

        printf("pOut:\n");
        for (int j = 0; j < dim_vec; j++)
        {
            printf("%d ", pOut[j]);
        }
        printf("\n");
    }
#endif
}

// float rand_float()
// {
//     return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
// }

bool init_opencl()
{
    cl_int status;

    printf("Initializing OpenCL\n");

    if (!setCwdToExeDir())
    {
        printf("ERROR: setCwdToExeDir Failed.\n");
        return false;
    }

    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    if (platform == NULL)
    {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Using %d device(s)\n", num_devices);
    for (unsigned i = 0; i < num_devices; ++i)
    {
        printf("  %s\n", getDeviceName(device[i]).c_str());
    }

    context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    std::string binary_file = getBoardBinaryFile("op_fc", device[0]);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

    printf("clBuildProgram\n");
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    queue.reset(1);
    my_kernel.reset(1);
    input_pV_buf.reset(1);
    input_pM_buf.reset(1);
    input_bias_buf.reset(1);
    input_vec_buffer.reset(1);
    output_pOut_buf.reset(1);

    unsigned i = 0;
    printf("clCreateCommandQueue\n");
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    printf("clCreateKernel\n");
    const char *kernel_name = "fc_dim1_q7";
    my_kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create my_kernel");

    return true;
}

void cleanup()
{
    for (unsigned i = 0; i < num_devices; ++i)
    {
        if (my_kernel && my_kernel[i])
        {
            clReleaseKernel(my_kernel[i]);
        }
        if (queue && queue[i])
        {
            clReleaseCommandQueue(queue[i]);
        }
        if (input_pV_buf && input_pV_buf[i])
        {
            clReleaseMemObject(input_pV_buf[i]);
        }
        if (input_pM_buf && input_pM_buf[i])
        {
            clReleaseMemObject(input_pM_buf[i]);
        }
        if (input_bias_buf && input_bias_buf[i])
        {
            clReleaseMemObject(input_bias_buf[i]);
        }
    }

    if (program)
    {
        clReleaseProgram(program);
    }
    if (context)
    {
        clReleaseContext(context);
    }
}
