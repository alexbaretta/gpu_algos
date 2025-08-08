s,<mma.h>,<rocwmma/rocwmma.hpp>,
s,nvcuda::wmma::precision::tf32,rocwmma::xfloat32_t,
s,cudaDataType_t,rocblas_datatype,
s,CUDA_R_32I,rocblas_datatype_i32_r,
s,CUDA_R_8I,rocblas_datatype_i8_r,
s,CUDA_R_8U,rocblas_datatype_u8_r,
s,CUDA_R_16F,rocblas_datatype_f16_r,
s,CUDA_R_32F,rocblas_datatype_f32_r,
s,CUDA_R_64F,rocblas_datatype_f64_r,
s,<cublasLt.h>,<rocblas/rocblas.h>,
s,cudaDeviceProp,hipDeviceProp_t,g
s,cuda_fp16.h,hip/hip_fp16.h,
s,cuda_runtime.h,hip/hip_runtime.h,
s,.cuh,.hiph,
s,CUDA_,HIP_,g
s,CUDART_,HIPRT_,
s,cuda,hip,g
s,cublasLt,hipblasLt,g
/cublasComputeType_t/d
/nvcc/d
# HIP does not permit querying the size of dynamic shared memory
s@assert(dynamic_shared_mem_size@//assert(dynamic_shared_mem_size@
s@asm volatile ("mov.u32 %0, %@//asm volatile ("mov.u32 %0, %@
