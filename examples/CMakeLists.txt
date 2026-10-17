if(SAF_ENABLE_VULKAN_COMPUTE)
    add_subdirectory(template_slang)
endif()

if(SAF_ENABLE_CUDA_INTEROP)
    add_subdirectory(template_cuda)
endif()
