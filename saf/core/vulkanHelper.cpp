/**
 * @file      vulkanHelper.cpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2025
 * @copyright Apache License 2.0
 */

#include "vulkanHelper.hpp"
#include <imgui.h>

#ifndef IM_MAX
#define IM_MAX(A, B) (((A) >= (B)) ? (A) : (B))
#endif

using namespace saf;

// This is pretty much the imgui impl helper for vulkan slightly adapted.

// dear imgui: Renderer Backend for Vulkan
// This needs to be used along with a Platform Backend (e.g. GLFW, SDL, Win32, custom..)

// Implemented features:
//  [!] Renderer: User texture binding. Use 'VkDescriptorSet' as ImTextureID. Call ImGui_ImplVulkan_AddTexture() to register one. Read the FAQ about ImTextureID! See https://github.com/ocornut/imgui/pull/914 for discussions.
//  [X] Renderer: Large meshes support (64k+ vertices) even with 16-bit indices (ImGuiBackendFlags_RendererHasVtxOffset).
//  [X] Renderer: Expose selected render state for draw callbacks to use. Access in '(ImGui_ImplXXXX_RenderState*)GetPlatformIO().Renderer_RenderState'.
//  [x] Renderer: Multi-viewport / platform windows. With issues (flickering when creating a new viewport).

// Important: on 32-bit systems, user texture binding is only supported if your imconfig file has '#define ImTextureID ImU64'.
// This is because we need ImTextureID to carry a 64-bit value and by default ImTextureID is defined as void*.
// To build this on 32-bit systems and support texture changes:
// - [Solution 1] IDE/msbuild: in "Properties/C++/Preprocessor Definitions" add 'ImTextureID=ImU64' (this is what we do in our .vcxproj files)
// - [Solution 2] IDE/msbuild: in "Properties/C++/Preprocessor Definitions" add 'USER_CONFIG="my_imgui_config.height"' and inside 'my_imgui_config.height' add '#define ImTextureID ImU64' and as many other options as you like.
// - [Solution 3] IDE/msbuild: edit imconfig.height and add '#define ImTextureID ImU64' (prefer solution 2 to create your own config file!)
// - [Solution 4] command-line: add '/D ImTextureID=ImU64' to your cl.exe command-line (this is what we do in our batch files)

// Visual Studio warnings
#ifdef _MSC_VER
#pragma warning(disable : 4127) // condition expression is constant
#endif

// Reusable buffers used for rendering 1 current in-flight frame, for vkRenderDrawData()
struct VulkanFrameRenderBuffers
{
    VkDeviceMemory vertexBufferMemory;
    VkDeviceMemory indexBufferMemory;
    VkDeviceSize vertexBufferSize;
    VkDeviceSize indexBufferSize;
    VkBuffer vertexBuffer;
    VkBuffer indexBuffer;

    VulkanFrameRenderBuffers()
    {
        memset(static_cast<void*>(this), 0, sizeof(*this));
    }
};

//  Each viewport will hold one of these
struct VulkanContextRenderBuffers
{
    U32 index;
    U32 count;
    ImVector<VulkanFrameRenderBuffers> frameRenderBuffers;

    VulkanContextRenderBuffers()
    {
        memset(static_cast<void*>(this), 0, sizeof(*this));
    }
};

struct VulkanTexture
{
    VkDeviceMemory memory;
    VkImage image;
    VkImageView imageView;
    VkDescriptorSet descriptorSet;

    VulkanTexture()
    {
        memset((void*)this, 0, sizeof(*this));
    }
};

// For multi-viewport support:
// Helper structure we store in the void* RendererUserData field of each Viewport to easily retrieve our backend data.
struct VulkanViewportData
{
    bool contextOwned;
    VulkanContext context;                    // Used by secondary viewports only
    VulkanContextRenderBuffers renderBuffers; // Used by all viewports
    bool swapChainNeedRebuild;                // Flag when viewport swapchain resized in the middle of processing a frame
    bool swapChainSuboptimal;                 // Flag when VK_SUBOPTIMAL_KHR was returned.

    VulkanViewportData()
    {
        contextOwned = swapChainNeedRebuild = swapChainSuboptimal = false;
        memset(&renderBuffers, 0, sizeof(renderBuffers));
    }
    ~VulkanViewportData() {}
};

// Vulkan data
struct VulkanData
{
    VulkanInitInfo vulkanInitInfo;
    VkDeviceSize bufferMemoryAlignment;
    VkPipelineCreateFlags pipelineCreateFlags;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkPipeline pipelineForViewports;
    VkShaderModule shaderModuleVert;
    VkShaderModule shaderModuleFrag;
    VkDescriptorPool descriptorPool;

    // Texture Management
    VulkanTexture fontTexture;
    VkSampler texSampler;
    VkCommandPool texCommandPool;
    VkCommandBuffer texCommandBuffer;

    // Render buffers for main window
    VulkanContextRenderBuffers mainContextRenderBuffers;

    VulkanData()
    {
        memset(static_cast<void*>(this), 0, sizeof(*this));
        bufferMemoryAlignment = 256;
    }
};

// Forward Declarations
bool vkCreateDeviceObjects();
void vkDestroyDeviceObjects();
void vkDestroyFrameRenderBuffers(VkDevice logicalDevice, VulkanFrameRenderBuffers* buffers, const VkAllocationCallbacks* allocator);
void vkDestroyContextRenderBuffers(VkDevice logicalDevice, VulkanContextRenderBuffers* buffers, const VkAllocationCallbacks* allocator);
void vkDestroyFrame(VkDevice logicalDevice, VulkanFrameData* fd, const VkAllocationCallbacks* allocator);
void vkDestroyFrameSemaphores(VkDevice logicalDevice, VulkanFrameSemaphores* fsd, const VkAllocationCallbacks* allocator);
void vkDestroyAllViewportsRenderBuffers(VkDevice logicalDevice, const VkAllocationCallbacks* allocator);
void vkCreateContextSwapChain(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VulkanContext* context, const VkAllocationCallbacks* allocator, I32 width, I32 height, U32 minImageCount);
void vkCreateContextCommandBuffers(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VulkanContext* context, U32 queueFamily, const VkAllocationCallbacks* allocator);

// Vulkan prototypes for use with custom loaders
// (see description of VK_NO_PROTOTYPES in vulkanHelper.hpp
#ifdef VK_NO_PROTOTYPES
static bool gFunctionsLoaded = false;
#else
static bool gFunctionsLoaded = true;
#endif
#ifdef VK_NO_PROTOTYPES
#define VULKAN_FUNC_MAP(VULKAN_FUNC_MAP_MACRO)                       \
    VULKAN_FUNC_MAP_MACRO(vkAllocateCommandBuffers)                  \
    VULKAN_FUNC_MAP_MACRO(vkAllocateDescriptorSets)                  \
    VULKAN_FUNC_MAP_MACRO(vkAllocateMemory)                          \
    VULKAN_FUNC_MAP_MACRO(vkBindBufferMemory)                        \
    VULKAN_FUNC_MAP_MACRO(vkBindImageMemory)                         \
    VULKAN_FUNC_MAP_MACRO(vkCmdBindDescriptorSets)                   \
    VULKAN_FUNC_MAP_MACRO(vkCmdBindIndexBuffer)                      \
    VULKAN_FUNC_MAP_MACRO(vkCmdBindPipeline)                         \
    VULKAN_FUNC_MAP_MACRO(vkCmdBindVertexBuffers)                    \
    VULKAN_FUNC_MAP_MACRO(vkCmdCopyBufferToImage)                    \
    VULKAN_FUNC_MAP_MACRO(vkCmdDrawIndexed)                          \
    VULKAN_FUNC_MAP_MACRO(vkCmdPipelineBarrier)                      \
    VULKAN_FUNC_MAP_MACRO(vkCmdPushConstants)                        \
    VULKAN_FUNC_MAP_MACRO(vkCmdSetScissor)                           \
    VULKAN_FUNC_MAP_MACRO(vkCmdSetViewport)                          \
    VULKAN_FUNC_MAP_MACRO(vkCreateBuffer)                            \
    VULKAN_FUNC_MAP_MACRO(vkCreateCommandPool)                       \
    VULKAN_FUNC_MAP_MACRO(vkCreateDescriptorSetLayout)               \
    VULKAN_FUNC_MAP_MACRO(vkCreateFence)                             \
    VULKAN_FUNC_MAP_MACRO(vkCreateFramebuffer)                       \
    VULKAN_FUNC_MAP_MACRO(vkCreateGraphicsPipelines)                 \
    VULKAN_FUNC_MAP_MACRO(vkCreateImage)                             \
    VULKAN_FUNC_MAP_MACRO(vkCreateImageView)                         \
    VULKAN_FUNC_MAP_MACRO(vkCreatePipelineLayout)                    \
    VULKAN_FUNC_MAP_MACRO(vkCreateRenderPass)                        \
    VULKAN_FUNC_MAP_MACRO(vkCreateSampler)                           \
    VULKAN_FUNC_MAP_MACRO(vkCreateSemaphore)                         \
    VULKAN_FUNC_MAP_MACRO(vkCreateShaderModule)                      \
    VULKAN_FUNC_MAP_MACRO(vkCreateSwapchainKHR)                      \
    VULKAN_FUNC_MAP_MACRO(vkDestroyBuffer)                           \
    VULKAN_FUNC_MAP_MACRO(vkDestroyCommandPool)                      \
    VULKAN_FUNC_MAP_MACRO(vkDestroyDescriptorSetLayout)              \
    VULKAN_FUNC_MAP_MACRO(vkDestroyFence)                            \
    VULKAN_FUNC_MAP_MACRO(vkDestroyFramebuffer)                      \
    VULKAN_FUNC_MAP_MACRO(vkDestroyImage)                            \
    VULKAN_FUNC_MAP_MACRO(vkDestroyImageView)                        \
    VULKAN_FUNC_MAP_MACRO(vkDestroyPipeline)                         \
    VULKAN_FUNC_MAP_MACRO(vkDestroyPipelineLayout)                   \
    VULKAN_FUNC_MAP_MACRO(vkDestroyRenderPass)                       \
    VULKAN_FUNC_MAP_MACRO(vkDestroySampler)                          \
    VULKAN_FUNC_MAP_MACRO(vkDestroySemaphore)                        \
    VULKAN_FUNC_MAP_MACRO(vkDestroyShaderModule)                     \
    VULKAN_FUNC_MAP_MACRO(vkDestroySurfaceKHR)                       \
    VULKAN_FUNC_MAP_MACRO(vkDestroySwapchainKHR)                     \
    VULKAN_FUNC_MAP_MACRO(vkDeviceWaitIdle)                          \
    VULKAN_FUNC_MAP_MACRO(vkFlushMappedMemoryRanges)                 \
    VULKAN_FUNC_MAP_MACRO(vkFreeCommandBuffers)                      \
    VULKAN_FUNC_MAP_MACRO(vkFreeDescriptorSets)                      \
    VULKAN_FUNC_MAP_MACRO(vkFreeMemory)                              \
    VULKAN_FUNC_MAP_MACRO(vkGetBufferMemoryRequirements)             \
    VULKAN_FUNC_MAP_MACRO(vkGetImageMemoryRequirements)              \
    VULKAN_FUNC_MAP_MACRO(vkGetPhysicalDeviceMemoryProperties)       \
    VULKAN_FUNC_MAP_MACRO(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) \
    VULKAN_FUNC_MAP_MACRO(vkGetPhysicalDeviceSurfaceFormatsKHR)      \
    VULKAN_FUNC_MAP_MACRO(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    VULKAN_FUNC_MAP_MACRO(vkGetSwapchainImagesKHR)                   \
    VULKAN_FUNC_MAP_MACRO(vkMapMemory)                               \
    VULKAN_FUNC_MAP_MACRO(vkUnmapMemory)                             \
    VULKAN_FUNC_MAP_MACRO(vkUpdateDescriptorSets)                    \
    VULKAN_FUNC_MAP_MACRO(vkGetPhysicalDeviceSurfaceSupportKHR)      \
    VULKAN_FUNC_MAP_MACRO(vkWaitForFences)                           \
    VULKAN_FUNC_MAP_MACRO(vkCmdBeginRenderPass)                      \
    VULKAN_FUNC_MAP_MACRO(vkCmdEndRenderPass)                        \
    VULKAN_FUNC_MAP_MACRO(vkQueuePresentKHR)                         \
    VULKAN_FUNC_MAP_MACRO(vkBeginCommandBuffer)                      \
    VULKAN_FUNC_MAP_MACRO(vkEndCommandBuffer)                        \
    VULKAN_FUNC_MAP_MACRO(vkResetFences)                             \
    VULKAN_FUNC_MAP_MACRO(vkQueueSubmit)                             \
    VULKAN_FUNC_MAP_MACRO(vkResetCommandPool)                        \
    VULKAN_FUNC_MAP_MACRO(vkAcquireNextImageKHR)

// Define function pointers
#define VULKAN_FUNC_DEF(func) static PFN_##func func;
VULKAN_FUNC_MAP(VULKAN_FUNC_DEF)
#undef VULKAN_FUNC_DEF
#endif // VK_NO_PROTOTYPES

#ifdef IMPL_VULKAN_HAS_DYNAMIC_RENDERING
static PFN_vkCmdBeginRenderingKHR VulkanFuncs_vkCmdBeginRenderingKHR;
static PFN_vkCmdEndRenderingKHR VulkanFuncs_vkCmdEndRenderingKHR;
#endif

//-----------------------------------------------------------------------------
// SHADERS
//-----------------------------------------------------------------------------

// Forward Declarations
static void vkInitMultiViewportSupport();
static void vkShutdownMultiViewportSupport();

// glsl_shader.vert, compiled with:
// # glslangValidator -V -x -o glsl_shader.vert.u32 glsl_shader.vert
/*
#version 450 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;

out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out struct { vec4 Color; vec2 UV; } Out;

void main()
{
    Out.Color = aColor;
    Out.UV = aUV;
    gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
}
*/
static U32 __glsl_shader_vert_spv[] = {
    0x07230203, 0x00010000, 0x00080001, 0x0000002e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000a000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x0000000f, 0x00000015,
    0x0000001b, 0x0000001c, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00030005, 0x00000009, 0x00000000, 0x00050006, 0x00000009, 0x00000000, 0x6f6c6f43,
    0x00000072, 0x00040006, 0x00000009, 0x00000001, 0x00005655, 0x00030005, 0x0000000b, 0x0074754f,
    0x00040005, 0x0000000f, 0x6c6f4361, 0x0000726f, 0x00030005, 0x00000015, 0x00565561, 0x00060005,
    0x00000019, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x00000019, 0x00000000,
    0x505f6c67, 0x7469736f, 0x006e6f69, 0x00030005, 0x0000001b, 0x00000000, 0x00040005, 0x0000001c,
    0x736f5061, 0x00000000, 0x00060005, 0x0000001e, 0x73755075, 0x6e6f4368, 0x6e617473, 0x00000074,
    0x00050006, 0x0000001e, 0x00000000, 0x61635375, 0x0000656c, 0x00060006, 0x0000001e, 0x00000001,
    0x61725475, 0x616c736e, 0x00006574, 0x00030005, 0x00000020, 0x00006370, 0x00040047, 0x0000000b,
    0x0000001e, 0x00000000, 0x00040047, 0x0000000f, 0x0000001e, 0x00000002, 0x00040047, 0x00000015,
    0x0000001e, 0x00000001, 0x00050048, 0x00000019, 0x00000000, 0x0000000b, 0x00000000, 0x00030047,
    0x00000019, 0x00000002, 0x00040047, 0x0000001c, 0x0000001e, 0x00000000, 0x00050048, 0x0000001e,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000001e, 0x00000001, 0x00000023, 0x00000008,
    0x00030047, 0x0000001e, 0x00000002, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040017,
    0x00000008, 0x00000006, 0x00000002, 0x0004001e, 0x00000009, 0x00000007, 0x00000008, 0x00040020,
    0x0000000a, 0x00000003, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000003, 0x00040015,
    0x0000000c, 0x00000020, 0x00000001, 0x0004002b, 0x0000000c, 0x0000000d, 0x00000000, 0x00040020,
    0x0000000e, 0x00000001, 0x00000007, 0x0004003b, 0x0000000e, 0x0000000f, 0x00000001, 0x00040020,
    0x00000011, 0x00000003, 0x00000007, 0x0004002b, 0x0000000c, 0x00000013, 0x00000001, 0x00040020,
    0x00000014, 0x00000001, 0x00000008, 0x0004003b, 0x00000014, 0x00000015, 0x00000001, 0x00040020,
    0x00000017, 0x00000003, 0x00000008, 0x0003001e, 0x00000019, 0x00000007, 0x00040020, 0x0000001a,
    0x00000003, 0x00000019, 0x0004003b, 0x0000001a, 0x0000001b, 0x00000003, 0x0004003b, 0x00000014,
    0x0000001c, 0x00000001, 0x0004001e, 0x0000001e, 0x00000008, 0x00000008, 0x00040020, 0x0000001f,
    0x00000009, 0x0000001e, 0x0004003b, 0x0000001f, 0x00000020, 0x00000009, 0x00040020, 0x00000021,
    0x00000009, 0x00000008, 0x0004002b, 0x00000006, 0x00000028, 0x00000000, 0x0004002b, 0x00000006,
    0x00000029, 0x3f800000, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003d, 0x00000007, 0x00000010, 0x0000000f, 0x00050041, 0x00000011, 0x00000012,
    0x0000000b, 0x0000000d, 0x0003003e, 0x00000012, 0x00000010, 0x0004003d, 0x00000008, 0x00000016,
    0x00000015, 0x00050041, 0x00000017, 0x00000018, 0x0000000b, 0x00000013, 0x0003003e, 0x00000018,
    0x00000016, 0x0004003d, 0x00000008, 0x0000001d, 0x0000001c, 0x00050041, 0x00000021, 0x00000022,
    0x00000020, 0x0000000d, 0x0004003d, 0x00000008, 0x00000023, 0x00000022, 0x00050085, 0x00000008,
    0x00000024, 0x0000001d, 0x00000023, 0x00050041, 0x00000021, 0x00000025, 0x00000020, 0x00000013,
    0x0004003d, 0x00000008, 0x00000026, 0x00000025, 0x00050081, 0x00000008, 0x00000027, 0x00000024,
    0x00000026, 0x00050051, 0x00000006, 0x0000002a, 0x00000027, 0x00000000, 0x00050051, 0x00000006,
    0x0000002b, 0x00000027, 0x00000001, 0x00070050, 0x00000007, 0x0000002c, 0x0000002a, 0x0000002b,
    0x00000028, 0x00000029, 0x00050041, 0x00000011, 0x0000002d, 0x0000001b, 0x0000000d, 0x0003003e,
    0x0000002d, 0x0000002c, 0x000100fd, 0x00010038
};

// glsl_shader.frag, compiled with:
// # glslangValidator -V -x -o glsl_shader.frag.u32 glsl_shader.frag
/*
#version 450 core
layout(location = 0) out vec4 fColor;
layout(set=0, binding=0) uniform sampler2D sTexture;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;
void main()
{
    fColor = In.Color * texture(sTexture, In.UV.st);
}
*/
static U32 __glsl_shader_frag_spv[] = {
    0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000d, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6c6f4366, 0x0000726f, 0x00030005, 0x0000000b, 0x00000000,
    0x00050006, 0x0000000b, 0x00000000, 0x6f6c6f43, 0x00000072, 0x00040006, 0x0000000b, 0x00000001,
    0x00005655, 0x00030005, 0x0000000d, 0x00006e49, 0x00050005, 0x00000016, 0x78655473, 0x65727574,
    0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000d, 0x0000001e,
    0x00000000, 0x00040047, 0x00000016, 0x00000022, 0x00000000, 0x00040047, 0x00000016, 0x00000021,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003,
    0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040017, 0x0000000a, 0x00000006,
    0x00000002, 0x0004001e, 0x0000000b, 0x00000007, 0x0000000a, 0x00040020, 0x0000000c, 0x00000001,
    0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000001, 0x00040015, 0x0000000e, 0x00000020,
    0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040020, 0x00000010, 0x00000001,
    0x00000007, 0x00090019, 0x00000013, 0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000001, 0x00000000, 0x0003001b, 0x00000014, 0x00000013, 0x00040020, 0x00000015, 0x00000000,
    0x00000014, 0x0004003b, 0x00000015, 0x00000016, 0x00000000, 0x0004002b, 0x0000000e, 0x00000018,
    0x00000001, 0x00040020, 0x00000019, 0x00000001, 0x0000000a, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x00000010, 0x00000011, 0x0000000d,
    0x0000000f, 0x0004003d, 0x00000007, 0x00000012, 0x00000011, 0x0004003d, 0x00000014, 0x00000017,
    0x00000016, 0x00050041, 0x00000019, 0x0000001a, 0x0000000d, 0x00000018, 0x0004003d, 0x0000000a,
    0x0000001b, 0x0000001a, 0x00050057, 0x00000007, 0x0000001c, 0x00000017, 0x0000001b, 0x00050085,
    0x00000007, 0x0000001d, 0x00000012, 0x0000001c, 0x0003003e, 0x00000009, 0x0000001d, 0x000100fd,
    0x00010038
};

//-----------------------------------------------------------------------------
// FUNCTIONS
//-----------------------------------------------------------------------------

// Backend data stored in io.BackendRendererUserData to allow support for multiple Dear ImGui contexts
// It is STRONGLY preferred that you use docking branch with multi-viewports (== single Dear ImGui context + multiple windows) instead of multiple Dear ImGui contexts.
// FIXME: multi-context support is not tested and probably dysfunctional in this backend.
static VulkanData* vkGetBackendData()
{
    return ImGui::GetCurrentContext() ? static_cast<VulkanData*>(ImGui::GetIO().BackendRendererUserData) : nullptr;
}

static U32 vkMemoryType(VkMemoryPropertyFlags properties, U32 typeBits)
{
    VulkanData* bd    = vkGetBackendData();
    VulkanInitInfo* v = &bd->vulkanInitInfo;
    VkPhysicalDeviceMemoryProperties prop;
    vkGetPhysicalDeviceMemoryProperties(v->physicalDevice, &prop);
    for (U32 i = 0; i < prop.memoryTypeCount; i++)
    {
        if ((prop.memoryTypes[i].propertyFlags & properties) == properties && typeBits & (1 << i))
        {
            return i;
        }
    }
    return 0xFFFFFFFF;
}

static void checkVkResultBD(VkResult err)
{
    VulkanData* bd = vkGetBackendData();
    if (!bd)
    {
        return;
    }
    VulkanInitInfo* v = &bd->vulkanInitInfo;
    if (v->checkVkResultFn)
    {
        v->checkVkResultFn(err);
    }
}

// Same as IM_MEMALIGN(). 'alignment' must be a power of two.
static inline VkDeviceSize alignBufferSize(VkDeviceSize size, VkDeviceSize alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

static void createOrResizeBuffer(VkBuffer& buffer, VkDeviceMemory& bufferMemory, VkDeviceSize& outBufferSize, PtrSize newSize, VkBufferUsageFlagBits usage)
{
    VulkanData* bd    = vkGetBackendData();
    VulkanInitInfo* v = &bd->vulkanInitInfo;
    VkResult err;
    if (buffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(v->device, buffer, v->allocator);
    }
    if (bufferMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(v->device, bufferMemory, v->allocator);
    }

    VkDeviceSize vertexBufferAlignedSize = alignBufferSize(IM_MAX(v->minAllocationSize, newSize), bd->bufferMemoryAlignment);
    VkBufferCreateInfo bufferInfo        = {};
    bufferInfo.sType                     = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size                      = vertexBufferAlignedSize;
    bufferInfo.usage                     = usage;
    bufferInfo.sharingMode               = VK_SHARING_MODE_EXCLUSIVE;
    err                                  = vkCreateBuffer(v->device, &bufferInfo, v->allocator, &buffer);
    checkVkResultBD(err);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(v->device, buffer, &req);
    bd->bufferMemoryAlignment      = (bd->bufferMemoryAlignment > req.alignment) ? bd->bufferMemoryAlignment : req.alignment;
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize       = req.size;
    allocInfo.memoryTypeIndex      = vkMemoryType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, req.memoryTypeBits);
    err                            = vkAllocateMemory(v->device, &allocInfo, v->allocator, &bufferMemory);
    checkVkResultBD(err);

    err = vkBindBufferMemory(v->device, buffer, bufferMemory, 0);
    checkVkResultBD(err);
    outBufferSize = req.size;
}

static void vkSetupImGuiRenderState(ImDrawData* drawData, VkPipeline pipeline, VkCommandBuffer commandBuffer, VulkanFrameRenderBuffers* rb, I32 width, I32 height)
{
    VulkanData* bd = vkGetBackendData();

    // Bind pipeline:
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    }

    // Bind Vertex And Index Buffer:
    if (drawData->TotalVtxCount > 0)
    {
        VkBuffer vertexBuffers[1]    = { rb->vertexBuffer };
        VkDeviceSize vertexOffset[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffset);
        vkCmdBindIndexBuffer(commandBuffer, rb->indexBuffer, 0, sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    }

    // Setup viewport:
    {
        VkViewport viewport;
        viewport.x        = 0;
        viewport.y        = 0;
        viewport.width    = static_cast<F32>(width);
        viewport.height   = static_cast<F32>(height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    }

    // Setup scale and translation:
    // Our visible imgui space lies from drawData->displayPps (top left) to drawData->displayPos+data_data->displaySize (bottom right). DisplayPos is (0,0) for single viewport apps.
    {
        F32 scale[2];
        scale[0] = 2.0f / drawData->DisplaySize.x;
        scale[1] = 2.0f / drawData->DisplaySize.y;
        F32 translate[2];
        translate[0] = -1.0f - drawData->DisplayPos.x * scale[0];
        translate[1] = -1.0f - drawData->DisplayPos.y * scale[1];
        vkCmdPushConstants(commandBuffer, bd->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, sizeof(F32) * 0, sizeof(F32) * 2, scale);
        vkCmdPushConstants(commandBuffer, bd->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, sizeof(F32) * 2, sizeof(F32) * 2, translate);
    }
}

// Render function
void saf::vkRenderImGuiDrawData(ImDrawData* drawData, VkCommandBuffer commandBuffer, VkPipeline pipeline)
{
    // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
    I32 width  = static_cast<I32>(drawData->DisplaySize.x * drawData->FramebufferScale.x);
    I32 height = static_cast<I32>(drawData->DisplaySize.y * drawData->FramebufferScale.y);
    if (width <= 0 || height <= 0)
    {
        return;
    }

    VulkanData* bd    = vkGetBackendData();
    VulkanInitInfo* v = &bd->vulkanInitInfo;
    if (pipeline == VK_NULL_HANDLE)
    {
        pipeline = bd->pipeline;
    }

    // Allocate array to store enough vertex/index buffers. Each unique viewport gets its own storage.
    VulkanViewportData* viewportRendererData = static_cast<VulkanViewportData*>(drawData->OwnerViewport->RendererUserData);
    SAF_ASSERT(viewportRendererData != nullptr);
    VulkanContextRenderBuffers* wrb = &viewportRendererData->renderBuffers;
    if (wrb->frameRenderBuffers.Size == 0)
    {
        wrb->index = 0;
        wrb->count = v->imageCount;
        wrb->frameRenderBuffers.resize(wrb->count);
        memset(wrb->frameRenderBuffers.Data, 0, wrb->frameRenderBuffers.size_in_bytes());
    }
    SAF_ASSERT(wrb->count == v->imageCount);
    wrb->index                   = (wrb->index + 1) % wrb->count;
    VulkanFrameRenderBuffers* rb = &wrb->frameRenderBuffers[wrb->index];

    if (drawData->TotalVtxCount > 0)
    {
        // Create or resize the vertex/index buffers
        VkDeviceSize vertexSize = alignBufferSize(drawData->TotalVtxCount * sizeof(ImDrawVert), bd->bufferMemoryAlignment);
        VkDeviceSize indexSize  = alignBufferSize(drawData->TotalIdxCount * sizeof(ImDrawIdx), bd->bufferMemoryAlignment);
        if (rb->vertexBuffer == VK_NULL_HANDLE || rb->vertexBufferSize < vertexSize)
        {
            createOrResizeBuffer(rb->vertexBuffer, rb->vertexBufferMemory, rb->vertexBufferSize, vertexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        }
        if (rb->indexBuffer == VK_NULL_HANDLE || rb->indexBufferSize < indexSize)
        {
            createOrResizeBuffer(rb->indexBuffer, rb->indexBufferMemory, rb->indexBufferSize, indexSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        }

        // Upload vertex/index data into a single contiguous GPU buffer
        ImDrawVert* vtxDst = nullptr;
        ImDrawIdx* idxDst  = nullptr;
        VkResult err       = vkMapMemory(v->device, rb->vertexBufferMemory, 0, rb->vertexBufferSize, 0, reinterpret_cast<void**>(&vtxDst));
        checkVkResultBD(err);
        err = vkMapMemory(v->device, rb->indexBufferMemory, 0, rb->indexBufferSize, 0, reinterpret_cast<void**>(&idxDst));
        checkVkResultBD(err);
        for (I32 n = 0; n < drawData->CmdListsCount; n++)
        {
            const ImDrawList* cmdList = drawData->CmdLists[n];
            memcpy(vtxDst, cmdList->VtxBuffer.Data, cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
            memcpy(idxDst, cmdList->IdxBuffer.Data, cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
            vtxDst += cmdList->VtxBuffer.Size;
            idxDst += cmdList->IdxBuffer.Size;
        }
        VkMappedMemoryRange range[2] = {};
        range[0].sType               = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range[0].memory              = rb->vertexBufferMemory;
        range[0].size                = VK_WHOLE_SIZE;
        range[1].sType               = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range[1].memory              = rb->indexBufferMemory;
        range[1].size                = VK_WHOLE_SIZE;
        err                          = vkFlushMappedMemoryRanges(v->device, 2, range);
        checkVkResultBD(err);
        vkUnmapMemory(v->device, rb->vertexBufferMemory);
        vkUnmapMemory(v->device, rb->indexBufferMemory);
    }

    // Setup desired Vulkan state
    vkSetupImGuiRenderState(drawData, pipeline, commandBuffer, rb, width, height);

    // Setup render state structure (for callbacks and custom texture bindings)
    ImGuiPlatformIO& platformIo = ImGui::GetPlatformIO();
    VulkanRenderState renderState;
    renderState.commandBuffer       = commandBuffer;
    renderState.pipeline            = pipeline;
    renderState.pipelineLayout      = bd->pipelineLayout;
    platformIo.Renderer_RenderState = &renderState;

    // Will project scissor/clipping rectangles into framebuffer space
    ImVec2 clipOff   = drawData->DisplayPos;       // (0,0) unless using multi-viewports
    ImVec2 clipScale = drawData->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

    // Render command lists
    // (Because we merged all buffers into a single one, we maintain our own offset into them)
    I32 globalVertexOffset = 0;
    I32 globalIndexOffset  = 0;
    for (I32 n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList* cmdList = drawData->CmdLists[n];
        for (I32 i = 0; i < cmdList->CmdBuffer.Size; i++)
        {
            const ImDrawCmd* pcmd = &cmdList->CmdBuffer[i];
            if (pcmd->UserCallback != nullptr)
            {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                {
                    vkSetupImGuiRenderState(drawData, pipeline, commandBuffer, rb, width, height);
                }
                else
                {
                    pcmd->UserCallback(cmdList, pcmd);
                }
            }
            else
            {
                // Project scissor/clipping rectangles into framebuffer space
                ImVec2 clipMin((pcmd->ClipRect.x - clipOff.x) * clipScale.x, (pcmd->ClipRect.y - clipOff.y) * clipScale.y);
                ImVec2 clipMax((pcmd->ClipRect.z - clipOff.x) * clipScale.x, (pcmd->ClipRect.w - clipOff.y) * clipScale.y);

                // Clamp to viewport as vkCmdSetScissor() won't accept values that are off bounds
                if (clipMin.x < 0.0f)
                {
                    clipMin.x = 0.0f;
                }
                if (clipMin.y < 0.0f)
                {
                    clipMin.y = 0.0f;
                }
                if (clipMax.x > width)
                {
                    clipMax.x = static_cast<F32>(width);
                }
                if (clipMax.y > height)
                {
                    clipMax.y = static_cast<F32>(height);
                }
                if (clipMax.x <= clipMin.x || clipMax.y <= clipMin.y)
                {
                    continue;
                }

                // Apply scissor/clipping rectangle
                VkRect2D scissor;
                scissor.offset.x      = static_cast<I32>(clipMin.x);
                scissor.offset.y      = static_cast<I32>(clipMin.y);
                scissor.extent.width  = static_cast<U32>(clipMax.x - clipMin.x);
                scissor.extent.height = static_cast<U32>(clipMax.y - clipMin.y);
                vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

                // Bind DescriptorSet with font or user texture
                VkDescriptorSet descSet = reinterpret_cast<VkDescriptorSet>(pcmd->GetTexID());
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, bd->pipelineLayout, 0, 1, &descSet, 0, nullptr);

                // Draw
                vkCmdDrawIndexed(commandBuffer, pcmd->ElemCount, 1, pcmd->IdxOffset + globalIndexOffset, pcmd->VtxOffset + globalVertexOffset, 0);
            }
        }
        globalIndexOffset += cmdList->IdxBuffer.Size;
        globalVertexOffset += cmdList->VtxBuffer.Size;
    }
    platformIo.Renderer_RenderState = nullptr;

    // Note: at this point both vkCmdSetViewport() and vkCmdSetScissor() have been called.
    // Our last values will leak into user/application rendering IF:
    // - Your app uses a pipeline with VK_DYNAMIC_STATE_VIEWPORT or VK_DYNAMIC_STATE_SCISSOR dynamic state
    // - And you forgot to call vkCmdSetViewport() and vkCmdSetScissor() yourself to explicitly set that state.
    // If you use VK_DYNAMIC_STATE_VIEWPORT or VK_DYNAMIC_STATE_SCISSOR you are responsible for setting the values before rendering.
    // In theory we should aim to backup/restore those values but I am not sure this is possible.
    // We perform a call to vkCmdSetScissor() to set back a full viewport which is likely to fix things for 99% users but technically this is not perfect. (See github #4644)
    VkRect2D scissor = { { 0, 0 }, { static_cast<U32>(width), static_cast<U32>(height) } };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

bool saf::vkCreateImGuiFontsTexture()
{
    ImGuiIO& io       = ImGui::GetIO();
    VulkanData* bd    = vkGetBackendData();
    VulkanInitInfo* v = &bd->vulkanInitInfo;
    VkResult err;

    // Destroy existing texture (if any)
    if (bd->fontTexture.descriptorSet)
    {
        vkQueueWaitIdle(v->queue);
        vkDestroyImGuiFontsTexture();
    }

    // Create command pool/buffer
    if (bd->texCommandPool == VK_NULL_HANDLE)
    {
        VkCommandPoolCreateInfo info = {};
        info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags                   = 0;
        info.queueFamilyIndex        = v->queueFamily;
        vkCreateCommandPool(v->device, &info, v->allocator, &bd->texCommandPool);
    }
    if (bd->texCommandBuffer == VK_NULL_HANDLE)
    {
        VkCommandBufferAllocateInfo info = {};
        info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool                 = bd->texCommandPool;
        info.commandBufferCount          = 1;
        err                              = vkAllocateCommandBuffers(v->device, &info, &bd->texCommandBuffer);
        checkVkResultBD(err);
    }

    // Start command buffer
    {
        err = vkResetCommandPool(v->device, bd->texCommandPool, 0);
        checkVkResultBD(err);
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(bd->texCommandBuffer, &beginInfo);
        checkVkResultBD(err);
    }

    unsigned char* pixels;
    I32 width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    PtrSize uploadSize = width * height * 4 * sizeof(char);

    // Create the Image:
    VulkanTexture* backendTex = &bd->fontTexture;
    {
        VkImageCreateInfo info = {};
        info.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType         = VK_IMAGE_TYPE_2D;
        info.format            = VK_FORMAT_R8G8B8A8_UNORM;
        info.extent.width      = width;
        info.extent.height     = height;
        info.extent.depth      = 1;
        info.mipLevels         = 1;
        info.arrayLayers       = 1;
        info.samples           = VK_SAMPLE_COUNT_1_BIT;
        info.tiling            = VK_IMAGE_TILING_OPTIMAL;
        info.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        err                    = vkCreateImage(v->device, &info, v->allocator, &backendTex->image);
        checkVkResultBD(err);
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(v->device, backendTex->image, &req);
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize       = IM_MAX(v->minAllocationSize, req.size);
        allocInfo.memoryTypeIndex      = vkMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, req.memoryTypeBits);
        err                            = vkAllocateMemory(v->device, &allocInfo, v->allocator, &backendTex->memory);
        checkVkResultBD(err);
        err = vkBindImageMemory(v->device, backendTex->image, backendTex->memory, 0);
        checkVkResultBD(err);
    }

    // Create the Image View:
    {
        VkImageViewCreateInfo info       = {};
        info.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image                       = backendTex->image;
        info.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        info.format                      = VK_FORMAT_R8G8B8A8_UNORM;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        err                              = vkCreateImageView(v->device, &info, v->allocator, &backendTex->imageView);
        checkVkResultBD(err);
    }

    // Create the Descriptor Set:
    backendTex->descriptorSet = vkAddTexture(bd->texSampler, backendTex->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Create the Upload Buffer:
    VkDeviceMemory uploadBufferMemory;
    VkBuffer uploadBuffer;
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size               = uploadSize;
        bufferInfo.usage              = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
        err                           = vkCreateBuffer(v->device, &bufferInfo, v->allocator, &uploadBuffer);
        checkVkResultBD(err);
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(v->device, uploadBuffer, &req);
        bd->bufferMemoryAlignment      = (bd->bufferMemoryAlignment > req.alignment) ? bd->bufferMemoryAlignment : req.alignment;
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize       = IM_MAX(v->minAllocationSize, req.size);
        allocInfo.memoryTypeIndex      = vkMemoryType(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, req.memoryTypeBits);
        err                            = vkAllocateMemory(v->device, &allocInfo, v->allocator, &uploadBufferMemory);
        checkVkResultBD(err);
        err = vkBindBufferMemory(v->device, uploadBuffer, uploadBufferMemory, 0);
        checkVkResultBD(err);
    }

    // Upload to Buffer:
    {
        char* map = nullptr;
        err       = vkMapMemory(v->device, uploadBufferMemory, 0, uploadSize, 0, (void**)(&map));
        checkVkResultBD(err);
        memcpy(map, pixels, uploadSize);
        VkMappedMemoryRange range[1] = {};
        range[0].sType               = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range[0].memory              = uploadBufferMemory;
        range[0].size                = uploadSize;
        err                          = vkFlushMappedMemoryRanges(v->device, 1, range);
        checkVkResultBD(err);
        vkUnmapMemory(v->device, uploadBufferMemory);
    }

    // Copy to Image:
    {
        VkImageMemoryBarrier copyBarrier[1]        = {};
        copyBarrier[0].sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        copyBarrier[0].dstAccessMask               = VK_ACCESS_TRANSFER_WRITE_BIT;
        copyBarrier[0].oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
        copyBarrier[0].newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copyBarrier[0].srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        copyBarrier[0].dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        copyBarrier[0].image                       = backendTex->image;
        copyBarrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyBarrier[0].subresourceRange.levelCount = 1;
        copyBarrier[0].subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(bd->texCommandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, copyBarrier);

        VkBufferImageCopy region           = {};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width           = width;
        region.imageExtent.height          = height;
        region.imageExtent.depth           = 1;
        vkCmdCopyBufferToImage(bd->texCommandBuffer, uploadBuffer, backendTex->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        VkImageMemoryBarrier useBarrier[1]        = {};
        useBarrier[0].sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        useBarrier[0].srcAccessMask               = VK_ACCESS_TRANSFER_WRITE_BIT;
        useBarrier[0].dstAccessMask               = VK_ACCESS_SHADER_READ_BIT;
        useBarrier[0].oldLayout                   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        useBarrier[0].newLayout                   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        useBarrier[0].srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        useBarrier[0].dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        useBarrier[0].image                       = backendTex->image;
        useBarrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        useBarrier[0].subresourceRange.levelCount = 1;
        useBarrier[0].subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(bd->texCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, useBarrier);
    }

    // Store our identifier
    io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(backendTex->descriptorSet));

    // End command buffer
    VkSubmitInfo endInfo       = {};
    endInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    endInfo.commandBufferCount = 1;
    endInfo.pCommandBuffers    = &bd->texCommandBuffer;
    err                        = vkEndCommandBuffer(bd->texCommandBuffer);
    checkVkResultBD(err);
    err = vkQueueSubmit(v->queue, 1, &endInfo, VK_NULL_HANDLE);
    checkVkResultBD(err);

    err = vkQueueWaitIdle(v->queue);
    checkVkResultBD(err);

    vkDestroyBuffer(v->device, uploadBuffer, v->allocator);
    vkFreeMemory(v->device, uploadBufferMemory, v->allocator);

    return true;
}

void saf::vkDestroyImGuiFontsTexture()
{
    ImGuiIO& io       = ImGui::GetIO();
    VulkanData* bd    = vkGetBackendData();
    VulkanInitInfo* v = &bd->vulkanInitInfo;

    VulkanTexture* backendTex = &bd->fontTexture;

    if (backendTex->descriptorSet)
    {
        vkRemoveTexture(backendTex->descriptorSet);
        backendTex->descriptorSet = VK_NULL_HANDLE;
        io.Fonts->SetTexID(0);
    }
    if (backendTex->imageView)
    {
        vkDestroyImageView(v->device, backendTex->imageView, v->allocator);
        backendTex->imageView = VK_NULL_HANDLE;
    }
    if (backendTex->image)
    {
        vkDestroyImage(v->device, backendTex->image, v->allocator);
        backendTex->image = VK_NULL_HANDLE;
    }
    if (backendTex->memory)
    {
        vkFreeMemory(v->device, backendTex->memory, v->allocator);
        backendTex->memory = VK_NULL_HANDLE;
    }
}

static void vkCreateShaderModules(VkDevice logicalDevice, const VkAllocationCallbacks* allocator)
{
    // Create the shader modules
    VulkanData* bd = vkGetBackendData();
    if (bd->shaderModuleVert == VK_NULL_HANDLE)
    {
        VkShaderModuleCreateInfo vertInfo = {};
        vertInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vertInfo.codeSize                 = sizeof(__glsl_shader_vert_spv);
        vertInfo.pCode                    = static_cast<U32*>(__glsl_shader_vert_spv);
        VkResult err                      = vkCreateShaderModule(logicalDevice, &vertInfo, allocator, &bd->shaderModuleVert);
        checkVkResultBD(err);
    }
    if (bd->shaderModuleFrag == VK_NULL_HANDLE)
    {
        VkShaderModuleCreateInfo fragInfo = {};
        fragInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fragInfo.codeSize                 = sizeof(__glsl_shader_frag_spv);
        fragInfo.pCode                    = static_cast<U32*>(__glsl_shader_frag_spv);
        VkResult err                      = vkCreateShaderModule(logicalDevice, &fragInfo, allocator, &bd->shaderModuleFrag);
        checkVkResultBD(err);
    }
}

static void vkCreatePipeline(VkDevice logicalDevice, const VkAllocationCallbacks* allocator, VkPipelineCache pipelineCache, VkRenderPass renderPass, VkSampleCountFlagBits MSAASamples, VkPipeline* pipeline, U32 subpass)
{
    VulkanData* bd = vkGetBackendData();
    vkCreateShaderModules(logicalDevice, allocator);

    VkPipelineShaderStageCreateInfo stage[2] = {};
    stage[0].sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage[0].stage                           = VK_SHADER_STAGE_VERTEX_BIT;
    stage[0].module                          = bd->shaderModuleVert;
    stage[0].pName                           = "main";
    stage[1].sType                           = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage[1].stage                           = VK_SHADER_STAGE_FRAGMENT_BIT;
    stage[1].module                          = bd->shaderModuleFrag;
    stage[1].pName                           = "main";

    VkVertexInputBindingDescription bindingDesc[1] = {};
    bindingDesc[0].stride                          = sizeof(ImDrawVert);
    bindingDesc[0].inputRate                       = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDesc[3] = {};
    attributeDesc[0].location                          = 0;
    attributeDesc[0].binding                           = bindingDesc[0].binding;
    attributeDesc[0].format                            = VK_FORMAT_R32G32_SFLOAT;
    attributeDesc[0].offset                            = IM_OFFSETOF(ImDrawVert, pos);
    attributeDesc[1].location                          = 1;
    attributeDesc[1].binding                           = bindingDesc[0].binding;
    attributeDesc[1].format                            = VK_FORMAT_R32G32_SFLOAT;
    attributeDesc[1].offset                            = IM_OFFSETOF(ImDrawVert, uv);
    attributeDesc[2].location                          = 2;
    attributeDesc[2].binding                           = bindingDesc[0].binding;
    attributeDesc[2].format                            = VK_FORMAT_R8G8B8A8_UNORM;
    attributeDesc[2].offset                            = IM_OFFSETOF(ImDrawVert, col);

    VkPipelineVertexInputStateCreateInfo vertexInfo = {};
    vertexInfo.sType                                = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInfo.vertexBindingDescriptionCount        = 1;
    vertexInfo.pVertexBindingDescriptions           = bindingDesc;
    vertexInfo.vertexAttributeDescriptionCount      = 3;
    vertexInfo.pVertexAttributeDescriptions         = attributeDesc;

    VkPipelineInputAssemblyStateCreateInfo iaInfo = {};
    iaInfo.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iaInfo.topology                               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportInfo = {};
    viewportInfo.sType                             = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportInfo.viewportCount                     = 1;
    viewportInfo.scissorCount                      = 1;

    VkPipelineRasterizationStateCreateInfo rasterInfo = {};
    rasterInfo.sType                                  = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterInfo.polygonMode                            = VK_POLYGON_MODE_FILL;
    rasterInfo.cullMode                               = VK_CULL_MODE_NONE;
    rasterInfo.frontFace                              = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterInfo.lineWidth                              = 1.0f;

    VkPipelineMultisampleStateCreateInfo msInfo = {};
    msInfo.sType                                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msInfo.rasterizationSamples                 = (MSAASamples != 0) ? MSAASamples : VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorAttachment[1] = {};
    colorAttachment[0].blendEnable                         = VK_TRUE;
    colorAttachment[0].srcColorBlendFactor                 = VK_BLEND_FACTOR_SRC_ALPHA;
    colorAttachment[0].dstColorBlendFactor                 = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorAttachment[0].colorBlendOp                        = VK_BLEND_OP_ADD;
    colorAttachment[0].srcAlphaBlendFactor                 = VK_BLEND_FACTOR_ONE;
    colorAttachment[0].dstAlphaBlendFactor                 = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorAttachment[0].alphaBlendOp                        = VK_BLEND_OP_ADD;
    colorAttachment[0].colorWriteMask                      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineDepthStencilStateCreateInfo depthInfo = {};
    depthInfo.sType                                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineColorBlendStateCreateInfo blendInfo = {};
    blendInfo.sType                               = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendInfo.attachmentCount                     = 1;
    blendInfo.pAttachments                        = colorAttachment;

    VkDynamicState dynamicStates[2]               = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType                            = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount                = static_cast<I32>(ARRAYSIZE(dynamicStates));
    dynamicState.pDynamicStates                   = dynamicStates;

    VkGraphicsPipelineCreateInfo info = {};
    info.sType                        = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.flags                        = bd->pipelineCreateFlags;
    info.stageCount                   = 2;
    info.pStages                      = stage;
    info.pVertexInputState            = &vertexInfo;
    info.pInputAssemblyState          = &iaInfo;
    info.pViewportState               = &viewportInfo;
    info.pRasterizationState          = &rasterInfo;
    info.pMultisampleState            = &msInfo;
    info.pDepthStencilState           = &depthInfo;
    info.pColorBlendState             = &blendInfo;
    info.pDynamicState                = &dynamicState;
    info.layout                       = bd->pipelineLayout;
    info.renderPass                   = renderPass;
    info.subpass                      = subpass;

#ifdef IMPL_VULKAN_HAS_DYNAMIC_RENDERING
    if (bd->vulkanInitInfo.useDynamicRendering)
    {
        SAF_ASSERT(bd->vulkanInitInfo.pipelineRenderingCreateInfo.sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR && "PipelineRenderingCreateInfo sType must be VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR");
        SAF_ASSERT(bd->vulkanInitInfo.pipelineRenderingCreateInfo.pNext == nullptr && "PipelineRenderingCreateInfo pNext must be nullptr");
        info.pNext      = &bd->vulkanInitInfo.pipelineRenderingCreateInfo;
        info.renderPass = VK_NULL_HANDLE; // Just make sure it's actually nullptr.
    }
#endif

    VkResult err = vkCreateGraphicsPipelines(logicalDevice, pipelineCache, 1, &info, allocator, pipeline);
    checkVkResultBD(err);
}

bool vkCreateDeviceObjects()
{
    VulkanData* bd    = vkGetBackendData();
    VulkanInitInfo* v = &bd->vulkanInitInfo;
    VkResult err;

    if (!bd->texSampler)
    {
        // Bilinear sampling is required by default. Set 'io.Fonts->flags |= ImFontAtlasFlags_NoBakedLines' or 'style.AntiAliasedLinesUseTex = false' to allow point/nearest sampling.
        VkSamplerCreateInfo info = {};
        info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter           = VK_FILTER_LINEAR;
        info.minFilter           = VK_FILTER_LINEAR;
        info.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.minLod              = -1000;
        info.maxLod              = 1000;
        info.maxAnisotropy       = 1.0f;
        err                      = vkCreateSampler(v->device, &info, v->allocator, &bd->texSampler);
        checkVkResultBD(err);
    }

    if (!bd->descriptorSetLayout)
    {
        VkDescriptorSetLayoutBinding binding[1] = {};
        binding[0].descriptorType               = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding[0].descriptorCount              = 1;
        binding[0].stageFlags                   = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo info    = {};
        info.sType                              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount                       = 1;
        info.pBindings                          = binding;
        err                                     = vkCreateDescriptorSetLayout(v->device, &info, v->allocator, &bd->descriptorSetLayout);
        checkVkResultBD(err);
    }

    if (v->descriptorPoolSize != 0)
    {
        SAF_ASSERT(v->descriptorPoolSize > 1);
        VkDescriptorPoolSize poolSize       = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, v->descriptorPoolSize };
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets                    = v->descriptorPoolSize;
        poolInfo.poolSizeCount              = 1;
        poolInfo.pPoolSizes                 = &poolSize;

        err = vkCreateDescriptorPool(v->device, &poolInfo, v->allocator, &bd->descriptorPool);
        checkVkResultBD(err);
    }

    if (!bd->pipelineLayout)
    {
        // Constants: we are using 'vec2 offset' and 'vec2 scale' instead of a full 3d projection matrix
        VkPushConstantRange pushConstants[1]  = {};
        pushConstants[0].stageFlags           = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstants[0].offset               = sizeof(F32) * 0;
        pushConstants[0].size                 = sizeof(F32) * 4;
        VkDescriptorSetLayout setLayout[1]    = { bd->descriptorSetLayout };
        VkPipelineLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount             = 1;
        layoutInfo.pSetLayouts                = setLayout;
        layoutInfo.pushConstantRangeCount     = 1;
        layoutInfo.pPushConstantRanges        = pushConstants;
        err                                   = vkCreatePipelineLayout(v->device, &layoutInfo, v->allocator, &bd->pipelineLayout);
        checkVkResultBD(err);
    }

    vkCreatePipeline(v->device, v->allocator, v->pipelineCache, v->renderPass, v->msaaSamples, &bd->pipeline, v->subpass);

    return true;
}

void vkDestroyDeviceObjects()
{
    VulkanData* bd    = vkGetBackendData();
    VulkanInitInfo* v = &bd->vulkanInitInfo;
    vkDestroyAllViewportsRenderBuffers(v->device, v->allocator);
    vkDestroyImGuiFontsTexture();

    if (bd->texCommandBuffer)
    {
        vkFreeCommandBuffers(v->device, bd->texCommandPool, 1, &bd->texCommandBuffer);
        bd->texCommandBuffer = VK_NULL_HANDLE;
    }
    if (bd->texCommandPool)
    {
        vkDestroyCommandPool(v->device, bd->texCommandPool, v->allocator);
        bd->texCommandPool = VK_NULL_HANDLE;
    }
    if (bd->texSampler)
    {
        vkDestroySampler(v->device, bd->texSampler, v->allocator);
        bd->texSampler = VK_NULL_HANDLE;
    }
    if (bd->shaderModuleVert)
    {
        vkDestroyShaderModule(v->device, bd->shaderModuleVert, v->allocator);
        bd->shaderModuleVert = VK_NULL_HANDLE;
    }
    if (bd->shaderModuleFrag)
    {
        vkDestroyShaderModule(v->device, bd->shaderModuleFrag, v->allocator);
        bd->shaderModuleFrag = VK_NULL_HANDLE;
    }
    if (bd->descriptorSetLayout)
    {
        vkDestroyDescriptorSetLayout(v->device, bd->descriptorSetLayout, v->allocator);
        bd->descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (bd->pipelineLayout)
    {
        vkDestroyPipelineLayout(v->device, bd->pipelineLayout, v->allocator);
        bd->pipelineLayout = VK_NULL_HANDLE;
    }
    if (bd->pipeline)
    {
        vkDestroyPipeline(v->device, bd->pipeline, v->allocator);
        bd->pipeline = VK_NULL_HANDLE;
    }
    if (bd->pipelineForViewports)
    {
        vkDestroyPipeline(v->device, bd->pipelineForViewports, v->allocator);
        bd->pipelineForViewports = VK_NULL_HANDLE;
    }
    if (bd->descriptorPool)
    {
        vkDestroyDescriptorPool(v->device, bd->descriptorPool, v->allocator);
        bd->descriptorPool = VK_NULL_HANDLE;
    }
}

bool saf::vkLoadFunctions(PFN_vkVoidFunction (*loaderFunc)(const char* functionName, void* userData), void* userData)
{
    // Load function pointers
    // You can use the default Vulkan loader using:
    //      VulkanLoadFunctions([](const char* functionName, void*) { return vkGetInstanceProcAddr(vulkanInstance, functionName); });
    // But this would be equivalent to not setting VK_NO_PROTOTYPES.
#ifdef VK_NO_PROTOTYPES
#define VULKAN_FUNC_LOAD(func)                                            \
    func = reinterpret_cast<decltype(func)>(loaderFunc(#func, userData)); \
    if (func == nullptr)                                                  \
        return false;
    VULKAN_FUNC_MAP(VULKAN_FUNC_LOAD)
#undef VULKAN_FUNC_LOAD

#ifdef IMPL_VULKAN_HAS_DYNAMIC_RENDERING
    // Manually load those two (see #5446)
    VulkanFuncs_vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(loaderFunc("vkCmdBeginRenderingKHR", userData));
    VulkanFuncs_vkCmdEndRenderingKHR   = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(loaderFunc("vkCmdEndRenderingKHR", userData));
#endif
#else
    IM_UNUSED(loaderFunc);
    IM_UNUSED(userData);
#endif

    gFunctionsLoaded = true;
    return true;
}

bool saf::vkInit(VulkanInitInfo* info)
{
    SAF_ASSERT(gFunctionsLoaded && "Need to call VulkanLoadFunctions() if IMPL_VULKAN_NO_PROTOTYPES or VK_NO_PROTOTYPES are set!");

    if (info->useDynamicRendering)
    {
#ifdef IMPL_VULKAN_HAS_DYNAMIC_RENDERING
#ifndef VK_NO_PROTOTYPES
        VulkanFuncs_vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(vkGetInstanceProcAddr(info->instance, "vkCmdBeginRenderingKHR"));
        VulkanFuncs_vkCmdEndRenderingKHR   = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(vkGetInstanceProcAddr(info->instance, "vkCmdEndRenderingKHR"));
#endif
        SAF_ASSERT(VulkanFuncs_vkCmdBeginRenderingKHR != nullptr);
        SAF_ASSERT(VulkanFuncs_vkCmdEndRenderingKHR != nullptr);
#else
        SAF_ASSERT(0 && "Can't use dynamic rendering when neither VK_VERSION_1_3 or VK_KHR_dynamic_rendering is defined.");
#endif
    }

    ImGuiIO& io = ImGui::GetIO();
    SAF_ASSERT(io.BackendRendererUserData == nullptr && "Already initialized a renderer backend!");

    // Setup backend capabilities flags
    VulkanData* bd             = IM_NEW(VulkanData)();
    io.BackendRendererUserData = static_cast<void*>(bd);
    io.BackendRendererName     = "imgui_impl_vulkan";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset; // We can honor the ImDrawCmd::VtxOffset field, allowing for large meshes.
    io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports; // We can create multi-viewports on the Renderer side (optional)

    SAF_ASSERT(info->instance != VK_NULL_HANDLE);
    SAF_ASSERT(info->physicalDevice != VK_NULL_HANDLE);
    SAF_ASSERT(info->device != VK_NULL_HANDLE);
    SAF_ASSERT(info->queue != VK_NULL_HANDLE);
    if (info->descriptorPool != VK_NULL_HANDLE) // Either DescriptorPool or DescriptorPoolSize must be set, not both!
        SAF_ASSERT(info->descriptorPoolSize == 0);
    else
        SAF_ASSERT(info->descriptorPoolSize > 0);
    SAF_ASSERT(info->minImageCount >= 2);
    SAF_ASSERT(info->imageCount >= info->minImageCount);
    if (info->useDynamicRendering == false)
        SAF_ASSERT(info->renderPass != VK_NULL_HANDLE);

    bd->vulkanInitInfo = *info;

    vkCreateDeviceObjects();

    // Our render function expect RendererUserData to be storing the window render buffer we need (for the main viewport we won't use ->context)
    ImGuiViewport* mainViewport    = ImGui::GetMainViewport();
    mainViewport->RendererUserData = IM_NEW(VulkanViewportData)();

    vkInitMultiViewportSupport();

    return true;
}

void saf::vkShutdown()
{
    VulkanData* bd = vkGetBackendData();
    SAF_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();

    // First destroy objects in all viewports
    vkDestroyDeviceObjects();

    // Manually delete main viewport render data in-case we haven't initialized for viewports
    ImGuiViewport* mainViewport = ImGui::GetMainViewport();
    if (VulkanViewportData* vd = static_cast<VulkanViewportData*>(mainViewport->RendererUserData))
    {
        IM_DELETE(vd);
    }
    mainViewport->RendererUserData = nullptr;

    // Clean up windows
    vkShutdownMultiViewportSupport();

    io.BackendRendererName     = nullptr;
    io.BackendRendererUserData = nullptr;
    io.BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasViewports);
    IM_DELETE(bd);
}

void saf::vkNewFrame()
{
    VulkanData* bd = vkGetBackendData();
    SAF_ASSERT(bd != nullptr && "Did you call VulkanInit()?");

    if (!bd->fontTexture.descriptorSet)
        vkCreateImGuiFontsTexture();
}

void saf::vkSetMinImageCount(U32 minImageCount)
{
    VulkanData* bd = vkGetBackendData();
    SAF_ASSERT(minImageCount >= 2);
    if (bd->vulkanInitInfo.minImageCount == minImageCount)
    {
        return;
    }

    SAF_ASSERT(0); // FIXME-VIEWPORT: Unsupported. Need to recreate all swap chains!
    VulkanInitInfo* v = &bd->vulkanInitInfo;
    VkResult err      = vkDeviceWaitIdle(v->device);
    checkVkResultBD(err);
    vkDestroyAllViewportsRenderBuffers(v->device, v->allocator);

    bd->vulkanInitInfo.minImageCount = minImageCount;
}

// Register a texture
// FIXME: This is experimental in the sense that we are unsure how to best design/tackle this problem, please post to https://github.com/ocornut/imgui/pull/914 if you have suggestions.
VkDescriptorSet saf::vkAddTexture(VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout)
{
    VulkanData* bd        = vkGetBackendData();
    VulkanInitInfo* v     = &bd->vulkanInitInfo;
    VkDescriptorPool pool = bd->descriptorPool ? bd->descriptorPool : v->descriptorPool;

    // Create Descriptor Set:
    VkDescriptorSet descriptorSet;
    {
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool              = pool;
        allocInfo.descriptorSetCount          = 1;
        allocInfo.pSetLayouts                 = &bd->descriptorSetLayout;
        VkResult err                          = vkAllocateDescriptorSets(v->device, &allocInfo, &descriptorSet);
        checkVkResultBD(err);
    }

    // Update the Descriptor Set:
    {
        VkDescriptorImageInfo imageDesc[1] = {};
        imageDesc[0].sampler               = sampler;
        imageDesc[0].imageView             = imageView;
        imageDesc[0].imageLayout           = imageLayout;
        VkWriteDescriptorSet writeDesc[1]  = {};
        writeDesc[0].sType                 = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDesc[0].dstSet                = descriptorSet;
        writeDesc[0].descriptorCount       = 1;
        writeDesc[0].descriptorType        = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDesc[0].pImageInfo            = imageDesc;
        vkUpdateDescriptorSets(v->device, 1, writeDesc, 0, nullptr);
    }
    return descriptorSet;
}

void saf::vkRemoveTexture(VkDescriptorSet descriptorSet)
{
    VulkanData* bd        = vkGetBackendData();
    VulkanInitInfo* v     = &bd->vulkanInitInfo;
    VkDescriptorPool pool = bd->descriptorPool ? bd->descriptorPool : v->descriptorPool;

    vkFreeDescriptorSets(v->device, pool, 1, &descriptorSet);
}

//-------------------------------------------------------------------------
// Internal / Miscellaneous Vulkan Helpers
// (Used by example's main.cpp. Used by multi-viewport features. PROBABLY NOT used by your own app.)
//-------------------------------------------------------------------------
// You probably do NOT need to use or care about those functions.
// Those functions only exist because:
//   1) they facilitate the readability and maintenance of the multiple main.cpp examples files.
//   2) the upcoming multi-viewport feature will need them internally.
// Generally we avoid exposing any kind of superfluous high-level helpers in the backends,
// but it is too much code to duplicate everywhere so we exceptionally expose them.
//
// Your engine/app will likely _already_ have code to setup all that stuff (swap chain, render pass, frame buffers, etc.).
// You may read this code to learn about Vulkan, but it is recommended you use you own custom tailored code to do equivalent work.
// (The VulkanXXX functions do not interact with any of the state used by the regular VulkanXXX functions)
//-------------------------------------------------------------------------

VkSurfaceFormatKHR saf::vkSelectSurfaceFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, const VkFormat* requestedFormats, I32 requestedFormatsCount, VkColorSpaceKHR requestedColorSpace)
{
    SAF_ASSERT(gFunctionsLoaded && "Need to call VulkanLoadFunctions() if IMPL_VULKAN_NO_PROTOTYPES or VK_NO_PROTOTYPES are set!");
    SAF_ASSERT(requestedFormats != nullptr);
    SAF_ASSERT(requestedFormatsCount > 0);

    // Per Spec Format and View Format are expected to be the same unless VK_IMAGE_CREATE_MUTABLE_BIT was set at image creation
    // Assuming that the default behavior is without setting this bit, there is no need for separate Swapchain image and image view format
    // Additionally several new color spaces were introduced with Vulkan Spec v1.0.40,
    // hence we must make sure that a format with the mostly available color space, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, is found and used.
    U32 availableCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &availableCount, nullptr);
    ImVector<VkSurfaceFormatKHR> availableFormats;
    availableFormats.resize(static_cast<I32>(availableCount));
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &availableCount, availableFormats.Data);

    // First check if only one format, VK_FORMAT_UNDEFINED, is available, which would imply that any format is available
    if (availableCount == 1)
    {
        if (availableFormats[0].format == VK_FORMAT_UNDEFINED)
        {
            VkSurfaceFormatKHR ret;
            ret.format     = requestedFormats[0];
            ret.colorSpace = requestedColorSpace;
            return ret;
        }
        else
        {
            // No point in searching another format
            return availableFormats[0];
        }
    }
    else
    {
        // Request several formats, the first found will be used
        for (I32 requestI = 0; requestI < requestedFormatsCount; requestI++)
            for (U32 availI = 0; availI < availableCount; availI++)
                if (availableFormats[availI].format == requestedFormats[requestI] && availableFormats[availI].colorSpace == requestedColorSpace)
                    return availableFormats[availI];

        // If none of the requested image formats could be found, use the first available
        return availableFormats[0];
    }
}

VkPresentModeKHR saf::vkSelectPresentMode(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, const VkPresentModeKHR* requestedModes, I32 requestedModesCount)
{
    SAF_ASSERT(gFunctionsLoaded && "Need to call VulkanLoadFunctions() if IMPL_VULKAN_NO_PROTOTYPES or VK_NO_PROTOTYPES are set!");
    SAF_ASSERT(requestedModes != nullptr);
    SAF_ASSERT(requestedModesCount > 0);

    // Request a certain mode and confirm that it is available. If not use VK_PRESENT_MODE_FIFO_KHR which is mandatory
    U32 availableCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &availableCount, nullptr);
    ImVector<VkPresentModeKHR> availableModes;
    availableModes.resize(static_cast<I32>(availableCount));
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &availableCount, availableModes.Data);
    // for (U32 availableI = 0; availableI < availableCount; availableI++)
    //     printf("[vulkan] availableModes[%d] = %d\n", availableI, availableModes[availableI]);

    for (I32 requestedI = 0; requestedI < requestedModesCount; requestedI++)
        for (U32 availableI = 0; availableI < availableCount; availableI++)
            if (requestedModes[requestedI] == availableModes[availableI])
                return requestedModes[requestedI];

    return VK_PRESENT_MODE_FIFO_KHR; // Always available
}

VkPhysicalDevice saf::vkSelectPhysicalDevice(VkInstance instance)
{
    U32 gpuCount;
    VkResult err = vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    checkVkResultBD(err);
    SAF_ASSERT(gpuCount > 0);

    ImVector<VkPhysicalDevice> gpus;
    gpus.resize(gpuCount);
    err = vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.Data);
    checkVkResultBD(err);

    // If a number >1 of GPUs got reported, find discrete GPU if present, or use first one available. This covers
    // most common cases (multi-gpu/integrated+dedicated graphics). Handling more complicated setups (multiple
    // dedicated GPUs) is out of scope of this sample.
    for (VkPhysicalDevice& device : gpus)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            return device;
    }

    // Use first GPU (Integrated) is a Discrete one is not available.
    if (gpuCount > 0)
        return gpus[0];
    return VK_NULL_HANDLE;
}

U32 saf::vkSelectQueueFamilyIndex(VkPhysicalDevice physicalDevice)
{
    U32 count;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    ImVector<VkQueueFamilyProperties> queuesProperties;
    queuesProperties.resize((I32)count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, queuesProperties.Data);
    for (U32 i = 0; i < count; i++)
        if (queuesProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            return i;
    return (U32)-1;
}

void vkCreateContextCommandBuffers(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VulkanContext* context, U32 queueFamily, const VkAllocationCallbacks* allocator)
{
    SAF_ASSERT(physicalDevice != VK_NULL_HANDLE && logicalDevice != VK_NULL_HANDLE);
    IM_UNUSED(physicalDevice);

    // Create Command Buffers
    VkResult err;
    for (U32 i = 0; i < context->framesInFlight; i++)
    {
        VulkanFrameData* fd = &context->frames[i];
        {
            VkCommandPoolCreateInfo info = {};
            info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            info.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            info.queueFamilyIndex        = queueFamily;
            err                          = vkCreateCommandPool(logicalDevice, &info, allocator, &fd->commandPool);
            checkVkResultBD(err);
        }
        {
            VkCommandBufferAllocateInfo info = {};
            info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            info.commandPool                 = fd->commandPool;
            info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            info.commandBufferCount          = 1;
            err                              = vkAllocateCommandBuffers(logicalDevice, &info, &fd->commandBuffer);
            checkVkResultBD(err);
        }
        {
            VkFenceCreateInfo info = {};
            info.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            info.flags             = VK_FENCE_CREATE_SIGNALED_BIT;
            err                    = vkCreateFence(logicalDevice, &info, allocator, &fd->fence);
            checkVkResultBD(err);
        }
    }

    for (U32 i = 0; i < context->semaphoreCount; i++)
    {
        VulkanFrameSemaphores* fsd = &context->frameSemaphores[i];
        {
            VkSemaphoreCreateInfo info = {};
            info.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            err                        = vkCreateSemaphore(logicalDevice, &info, allocator, &fsd->imageAcquiredSemaphore);
            checkVkResultBD(err);
            err = vkCreateSemaphore(logicalDevice, &info, allocator, &fsd->renderCompleteSemaphore);
            checkVkResultBD(err);
        }
    }

    // resource
    {
        VkCommandPoolCreateInfo info = {};
        info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex        = queueFamily;
        err                          = vkCreateCommandPool(logicalDevice, &info, allocator, &context->ressourceCommandPool);
        checkVkResultBD(err);
    }
    {
        VkCommandBufferAllocateInfo info = {};
        info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool                 = context->ressourceCommandPool;
        info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount          = 1;
        err                              = vkAllocateCommandBuffers(logicalDevice, &info, &context->ressourceCommandBuffer);
        checkVkResultBD(err);
    }
}

I32 saf::vkGetMinImageCountFromPresentMode(VkPresentModeKHR presentMode)
{
    if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        return 3;
    if (presentMode == VK_PRESENT_MODE_FIFO_KHR || presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
        return 2;
    if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        return 1;
    SAF_ASSERT(0);
    return 1;
}

// Also destroy old swap chain and in-flight frames data, if any.
void vkCreateContextSwapChain(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VulkanContext* context, const VkAllocationCallbacks* allocator, I32 width, I32 height, U32 minImageCount)
{
    VkResult err;
    VkSwapchainKHR oldSwapchain = context->swapchain;
    context->swapchain          = VK_NULL_HANDLE;
    err                         = vkDeviceWaitIdle(logicalDevice);
    checkVkResultBD(err);

    // We don't use vkDestroyContext() because we want to preserve the old swapchain to create the new one.
    // Destroy old Framebuffer
    for (U32 i = 0; i < context->framesInFlight; i++)
    {
        vkDestroyFrame(logicalDevice, &context->frames[i], allocator);
    }

    for (U32 i = 0; i < context->semaphoreCount; i++)
    {
        vkDestroyFrameSemaphores(logicalDevice, &context->frameSemaphores[i], allocator);
    }

    context->frames.clear();
    context->frameSemaphores.clear();
    context->framesInFlight = 0;

    if (context->renderPass)
    {
        vkDestroyRenderPass(logicalDevice, context->renderPass, allocator);
    }

    if (minImageCount == 0)
    {
        minImageCount = vkGetMinImageCountFromPresentMode(context->presentMode);
    }

    // Destroy Ressource Command*
    if (context->ressourceCommandBuffer)
    {
        vkFreeCommandBuffers(logicalDevice, context->ressourceCommandPool, 1, &context->ressourceCommandBuffer);
        vkDestroyCommandPool(logicalDevice, context->ressourceCommandPool, allocator);
        context->ressourceCommandBuffer = VK_NULL_HANDLE;
        context->ressourceCommandPool   = VK_NULL_HANDLE;
    }

    // Create Swapchain
    {
        VkSurfaceCapabilitiesKHR cap;
        err = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, context->surface, &cap);
        checkVkResultBD(err);

        VkSwapchainCreateInfoKHR info = {};
        info.sType                    = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.surface                  = context->surface;
        info.minImageCount            = minImageCount;
        info.imageFormat              = context->surfaceFormat.format;
        info.imageColorSpace          = context->surfaceFormat.colorSpace;
        info.imageArrayLayers         = 1;
        info.imageUsage               = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        info.imageSharingMode         = VK_SHARING_MODE_EXCLUSIVE; // Assume that graphics family == present family
        info.preTransform             = (cap.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : cap.currentTransform;
        info.compositeAlpha           = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        info.presentMode              = context->presentMode;
        info.clipped                  = VK_TRUE;
        info.oldSwapchain             = oldSwapchain;

        if (info.minImageCount < cap.minImageCount)
        {
            info.minImageCount = cap.minImageCount;
        }
        else if (cap.maxImageCount != 0 && info.minImageCount > cap.maxImageCount)
        {
            info.minImageCount = cap.maxImageCount;
        }

        if (cap.currentExtent.width == 0xffffffff)
        {
            info.imageExtent.width = context->width = width;
            info.imageExtent.height = context->height = height;
        }
        else
        {
            info.imageExtent.width = context->width = cap.currentExtent.width;
            info.imageExtent.height = context->height = cap.currentExtent.height;
        }
        err = vkCreateSwapchainKHR(logicalDevice, &info, allocator, &context->swapchain);
        checkVkResultBD(err);
        err = vkGetSwapchainImagesKHR(logicalDevice, context->swapchain, &context->framesInFlight, nullptr);
        checkVkResultBD(err);
        VkImage backbuffers[16] = {};
        SAF_ASSERT(context->framesInFlight >= minImageCount);
        SAF_ASSERT(context->framesInFlight < ARRAYSIZE(backbuffers));
        err = vkGetSwapchainImagesKHR(logicalDevice, context->swapchain, &context->framesInFlight, backbuffers);
        checkVkResultBD(err);

        context->semaphoreCount = context->framesInFlight + 1;
        context->frames.resize(context->framesInFlight);
        context->frameSemaphores.resize(context->semaphoreCount);
        memset(context->frames.Data, 0, context->frames.size_in_bytes());
        memset(context->frameSemaphores.Data, 0, context->frameSemaphores.size_in_bytes());
        for (U32 i = 0; i < context->framesInFlight; i++)
        {
            context->frames[i].backbuffer = backbuffers[i];
        }
    }
    if (oldSwapchain)
    {
        vkDestroySwapchainKHR(logicalDevice, oldSwapchain, allocator);
    }

    // Create the Render Pass
    if (context->useDynamicRendering == false)
    {
        VkAttachmentDescription attachment    = {};
        attachment.format                     = context->surfaceFormat.format;
        attachment.samples                    = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp                     = context->clearEnable ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.storeOp                    = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp              = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp             = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout              = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout                = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference colorAttachment = {};
        colorAttachment.attachment            = 0;
        colorAttachment.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpass          = {};
        subpass.pipelineBindPoint             = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount          = 1;
        subpass.pColorAttachments             = &colorAttachment;
        VkSubpassDependency dependency        = {};
        dependency.srcSubpass                 = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass                 = 0;
        dependency.srcStageMask               = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask               = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask              = 0;
        dependency.dstAccessMask              = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo info           = {};
        info.sType                            = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount                  = 1;
        info.pAttachments                     = &attachment;
        info.subpassCount                     = 1;
        info.pSubpasses                       = &subpass;
        info.dependencyCount                  = 1;
        info.pDependencies                    = &dependency;
        err                                   = vkCreateRenderPass(logicalDevice, &info, allocator, &context->renderPass);
        checkVkResultBD(err);

        // We do not create a pipeline by default as this is also used by examples' main.cpp,
        // but secondary viewport in multi-viewport mode may want to create one with:
        // vkCreatePipeline(logicalDevice, allocator, VK_NULL_HANDLE, context->renderPass, VK_SAMPLE_COUNT_1_BIT, &context->pipeline, v->subpass);
    }

    // Create The Image Views
    {
        VkImageViewCreateInfo info         = {};
        info.sType                         = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.viewType                      = VK_IMAGE_VIEW_TYPE_2D;
        info.format                        = context->surfaceFormat.format;
        info.components.r                  = VK_COMPONENT_SWIZZLE_R;
        info.components.g                  = VK_COMPONENT_SWIZZLE_G;
        info.components.b                  = VK_COMPONENT_SWIZZLE_B;
        info.components.a                  = VK_COMPONENT_SWIZZLE_A;
        VkImageSubresourceRange imageRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        info.subresourceRange              = imageRange;
        for (U32 i = 0; i < context->framesInFlight; i++)
        {
            VulkanFrameData* fd = &context->frames[i];
            info.image          = fd->backbuffer;
            err                 = vkCreateImageView(logicalDevice, &info, allocator, &fd->backbufferView);
            checkVkResultBD(err);
        }
    }

    // Create Framebuffer
    if (context->useDynamicRendering == false)
    {
        VkImageView attachment[1];
        VkFramebufferCreateInfo info = {};
        info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass              = context->renderPass;
        info.attachmentCount         = 1;
        info.pAttachments            = attachment;
        info.width                   = context->width;
        info.height                  = context->height;
        info.layers                  = 1;
        for (U32 i = 0; i < context->framesInFlight; i++)
        {
            VulkanFrameData* fd = &context->frames[i];
            attachment[0]       = fd->backbufferView;
            err                 = vkCreateFramebuffer(logicalDevice, &info, allocator, &fd->framebuffer);
            checkVkResultBD(err);
        }
    }
}

// Create or resize the context
void saf::vkCreateOrResizeContext(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VulkanContext* context, U32 queueFamily, const VkAllocationCallbacks* allocator, I32 width, I32 height, U32 minImageCount)
{
    SAF_ASSERT(gFunctionsLoaded && "Need to call VulkanLoadFunctions() if IMPL_VULKAN_NO_PROTOTYPES or VK_NO_PROTOTYPES are set!");
    (void)instance;
    vkCreateContextSwapChain(physicalDevice, logicalDevice, context, allocator, width, height, minImageCount);
    // vkCreatePipeline(logicalDevice, allocator, VK_NULL_HANDLE, context->renderPass, VK_SAMPLE_COUNT_1_BIT, &context->pipeline, g_VulkanInitInfo.Subpass);
    vkCreateContextCommandBuffers(physicalDevice, logicalDevice, context, queueFamily, allocator);
}

void saf::vkDestroyContext(VkInstance instance, VkDevice logicalDevice, VulkanContext* context, const VkAllocationCallbacks* allocator)
{
    vkDeviceWaitIdle(logicalDevice); // FIXME: We could wait on the Queue if we had the queue in context-> (otherwise functions can't use globals)
    // vkQueueWaitIdle(bd->Queue);

    for (U32 i = 0; i < context->framesInFlight; i++)
    {
        vkDestroyFrame(logicalDevice, &context->frames[i], allocator);
    }

    for (U32 i = 0; i < context->semaphoreCount; i++)
    {
        vkDestroyFrameSemaphores(logicalDevice, &context->frameSemaphores[i], allocator);
    }

    context->frames.clear();
    context->frameSemaphores.clear();

    // Destroy Ressource Command*
    if (context->ressourceCommandBuffer)
    {
        vkFreeCommandBuffers(logicalDevice, context->ressourceCommandPool, 1, &context->ressourceCommandBuffer);
        vkDestroyCommandPool(logicalDevice, context->ressourceCommandPool, allocator);
        context->ressourceCommandBuffer = VK_NULL_HANDLE;
        context->ressourceCommandPool   = VK_NULL_HANDLE;
    }

    vkDestroyRenderPass(logicalDevice, context->renderPass, allocator);
    vkDestroySwapchainKHR(logicalDevice, context->swapchain, allocator);
    vkDestroySurfaceKHR(instance, context->surface, allocator);

    *context = VulkanContext();
}

void vkDestroyFrame(VkDevice logicalDevice, VulkanFrameData* fd, const VkAllocationCallbacks* allocator)
{
    vkDestroyFence(logicalDevice, fd->fence, allocator);
    vkFreeCommandBuffers(logicalDevice, fd->commandPool, 1, &fd->commandBuffer);
    vkDestroyCommandPool(logicalDevice, fd->commandPool, allocator);
    fd->fence         = VK_NULL_HANDLE;
    fd->commandBuffer = VK_NULL_HANDLE;
    fd->commandPool   = VK_NULL_HANDLE;

    vkDestroyImageView(logicalDevice, fd->backbufferView, allocator);
    vkDestroyFramebuffer(logicalDevice, fd->framebuffer, allocator);
}

void vkDestroyFrameSemaphores(VkDevice logicalDevice, VulkanFrameSemaphores* fsd, const VkAllocationCallbacks* allocator)
{
    vkDestroySemaphore(logicalDevice, fsd->imageAcquiredSemaphore, allocator);
    vkDestroySemaphore(logicalDevice, fsd->renderCompleteSemaphore, allocator);
    fsd->imageAcquiredSemaphore = fsd->renderCompleteSemaphore = VK_NULL_HANDLE;
}

void vkDestroyFrameRenderBuffers(VkDevice logicalDevice, VulkanFrameRenderBuffers* buffers, const VkAllocationCallbacks* allocator)
{
    if (buffers->vertexBuffer)
    {
        vkDestroyBuffer(logicalDevice, buffers->vertexBuffer, allocator);
        buffers->vertexBuffer = VK_NULL_HANDLE;
    }
    if (buffers->vertexBufferMemory)
    {
        vkFreeMemory(logicalDevice, buffers->vertexBufferMemory, allocator);
        buffers->vertexBufferMemory = VK_NULL_HANDLE;
    }
    if (buffers->indexBuffer)
    {
        vkDestroyBuffer(logicalDevice, buffers->indexBuffer, allocator);
        buffers->indexBuffer = VK_NULL_HANDLE;
    }
    if (buffers->indexBufferMemory)
    {
        vkFreeMemory(logicalDevice, buffers->indexBufferMemory, allocator);
        buffers->indexBufferMemory = VK_NULL_HANDLE;
    }
    buffers->vertexBufferSize = 0;
    buffers->indexBufferSize  = 0;
}

void vkDestroyContextRenderBuffers(VkDevice logicalDevice, VulkanContextRenderBuffers* buffers, const VkAllocationCallbacks* allocator)
{
    for (U32 n = 0; n < buffers->count; n++)
    {
        vkDestroyFrameRenderBuffers(logicalDevice, &buffers->frameRenderBuffers[n], allocator);
    }
    buffers->frameRenderBuffers.clear();
    buffers->index = 0;
    buffers->count = 0;
}

void vkDestroyAllViewportsRenderBuffers(VkDevice logicalDevice, const VkAllocationCallbacks* allocator)
{
    ImGuiPlatformIO& platformIO = ImGui::GetPlatformIO();
    for (I32 n = 0; n < platformIO.Viewports.Size; n++)
    {
        if (VulkanViewportData* vd = static_cast<VulkanViewportData*>(platformIO.Viewports[n]->RendererUserData))
        {
            vkDestroyContextRenderBuffers(logicalDevice, &vd->renderBuffers, allocator);
        }
    }
}

//--------------------------------------------------------------------------------------------------------
// MULTI-VIEWPORT / PLATFORM INTERFACE SUPPORT
// This is an _advanced_ and _optional_ feature, allowing the backend to create and handle multiple viewports simultaneously.
// If you are new to dear imgui or creating a new binding for dear imgui, it is recommended that you completely ignore this section first..
//--------------------------------------------------------------------------------------------------------

static void vkCreateImGuiViewportContext(ImGuiViewport* viewport)
{
    VulkanData* bd             = vkGetBackendData();
    VulkanViewportData* vd     = IM_NEW(VulkanViewportData)();
    viewport->RendererUserData = vd;
    VulkanContext* context     = &vd->context;
    VulkanInitInfo* v          = &bd->vulkanInitInfo;

    // Create surface
    ImGuiPlatformIO& platformIO = ImGui::GetPlatformIO();
    VkResult err                = (VkResult)platformIO.Platform_CreateVkSurface(viewport, reinterpret_cast<ImU64>(v->instance), static_cast<const void*>(v->allocator), reinterpret_cast<ImU64*>(&context->surface));
    checkVkResultBD(err);

    // Check for WSI support
    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(v->physicalDevice, v->queueFamily, context->surface, &res);
    if (res != VK_TRUE)
    {
        SAF_ASSERT(0); // Error: no WSI support on physical logicalDevice
        return;
    }

    // Select Surface Format
    ImVector<VkFormat> requestSurfaceImageFormats;
#ifdef IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
    for (U32 n = 0; n < v->pipelineRenderingCreateInfo.colorAttachmentCount; n++)
        requestSurfaceImageFormats.push_back(v->pipelineRenderingCreateInfo.pColorAttachmentFormats[n]);
#endif
    const VkFormat defaultFormats[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
    for (VkFormat format : defaultFormats)
        requestSurfaceImageFormats.push_back(format);

    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    context->surfaceFormat                         = vkSelectSurfaceFormat(v->physicalDevice, context->surface, requestSurfaceImageFormats.Data, (size_t)requestSurfaceImageFormats.Size, requestSurfaceColorSpace);

    // Select Present Mode
    // FIXME-VULKAN: Even thought mailbox seems to get us maximum framerate with a single window, it halves framerate with a second window etc. (w/ Nvidia and SDK 1.82.1)
    VkPresentModeKHR presentModes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
    context->presentMode            = vkSelectPresentMode(v->physicalDevice, context->surface, &presentModes[0], ARRAYSIZE(presentModes));

    // Create SwapChain, RenderPass, Framebuffer, etc.presentModes
    context->clearEnable         = (viewport->Flags & ImGuiViewportFlags_NoRendererClear) ? false : true;
    context->useDynamicRendering = v->useDynamicRendering;
    vkCreateOrResizeContext(v->instance, v->physicalDevice, v->device, context, v->queueFamily, v->allocator, static_cast<I32>(viewport->Size.x), static_cast<I32>(viewport->Size.y), v->minImageCount);
    vd->contextOwned = true;

    // Create pipeline (shared by all secondary viewports)
    if (bd->pipelineForViewports == VK_NULL_HANDLE)
        vkCreatePipeline(v->device, v->allocator, VK_NULL_HANDLE, context->renderPass, VK_SAMPLE_COUNT_1_BIT, &bd->pipelineForViewports, 0);
}

static void vkDestroyImGuiViewportContext(ImGuiViewport* viewport)
{
    // The main viewport (owned by the application) will always have RendererUserData == 0 since we didn't create the data for it.
    VulkanData* bd = vkGetBackendData();
    if (VulkanViewportData* vd = static_cast<VulkanViewportData*>(viewport->RendererUserData))
    {
        VulkanInitInfo* v = &bd->vulkanInitInfo;
        if (vd->contextOwned)
            vkDestroyContext(v->instance, v->device, &vd->context, v->allocator);
        vkDestroyContextRenderBuffers(v->device, &vd->renderBuffers, v->allocator);
        IM_DELETE(vd);
    }
    viewport->RendererUserData = nullptr;
}

static void vkSetImGuiViewportContextSize(ImGuiViewport* viewport, ImVec2 size)
{
    VulkanData* bd         = vkGetBackendData();
    VulkanViewportData* vd = static_cast<VulkanViewportData*>(viewport->RendererUserData);
    if (vd == nullptr) // This is nullptr for the main viewport (which is left to the user/app to handle)
    {
        return;
    }
    VulkanInitInfo* v       = &bd->vulkanInitInfo;
    vd->context.clearEnable = (viewport->Flags & ImGuiViewportFlags_NoRendererClear) ? false : true;
    vkCreateOrResizeContext(v->instance, v->physicalDevice, v->device, &vd->context, v->queueFamily, v->allocator, static_cast<I32>(size.x), static_cast<I32>(size.y), v->minImageCount);
}

static void vkRenderImGuiViewportContext(ImGuiViewport* viewport, void*)
{
    VulkanData* bd         = vkGetBackendData();
    VulkanViewportData* vd = static_cast<VulkanViewportData*>(viewport->RendererUserData);
    VulkanContext* context = &vd->context;
    VulkanInitInfo* v      = &bd->vulkanInitInfo;
    VkResult err;

    if (vd->swapChainNeedRebuild || vd->swapChainSuboptimal)
    {
        vkCreateOrResizeContext(v->instance, v->physicalDevice, v->device, context, v->queueFamily, v->allocator, static_cast<I32>(viewport->Size.x), static_cast<I32>(viewport->Size.y), v->minImageCount);
        vd->swapChainNeedRebuild = vd->swapChainSuboptimal = false;
    }

    VulkanFrameData* fd        = &context->frames[context->frameIndex];
    VulkanFrameSemaphores* fsd = &context->frameSemaphores[context->semaphoreIndex];
    {
        {
            err = vkAcquireNextImageKHR(v->device, context->swapchain, UINT64_MAX, fsd->imageAcquiredSemaphore, VK_NULL_HANDLE, &context->frameIndex);
            if (err == VK_ERROR_OUT_OF_DATE_KHR)
            {
                vd->swapChainNeedRebuild = true; // Since we are not going to swap this frame anyway, it's ok that recreation happens on next frame.
                return;
            }
            if (err == VK_SUBOPTIMAL_KHR)
            {
                vd->swapChainSuboptimal = true;
            }
            else
            {
                checkVkResultBD(err);
            }
            fd = &context->frames[context->frameIndex];
        }
        for (;;)
        {
            err = vkWaitForFences(v->device, 1, &fd->fence, VK_TRUE, 100);
            if (err == VK_SUCCESS)
                break;
            if (err == VK_TIMEOUT)
                continue;
            checkVkResultBD(err);
        }
        {
            err = vkResetCommandPool(v->device, fd->commandPool, 0);
            checkVkResultBD(err);
            VkCommandBufferBeginInfo info = {};
            info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            err = vkBeginCommandBuffer(fd->commandBuffer, &info);
            checkVkResultBD(err);
        }
        {
            ImVec4 clearColor = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
            memcpy(&context->clearValue.color.float32[0], &clearColor, 4 * sizeof(F32));
        }
#ifdef IMPL_VULKAN_HAS_DYNAMIC_RENDERING
        if (v->useDynamicRendering)
        {
            // Transition swapchain image to a layout suitable for drawing.
            VkImageMemoryBarrier barrier        = {};
            barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.dstAccessMask               = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.oldLayout                   = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout                   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.image                       = fd->backbuffer;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(fd->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkRenderingAttachmentInfo attachmentInfo = {};
            attachmentInfo.sType                     = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
            attachmentInfo.imageView                 = fd->backbufferView;
            attachmentInfo.imageLayout               = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachmentInfo.resolveMode               = VK_RESOLVE_MODE_NONE;
            attachmentInfo.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachmentInfo.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
            attachmentInfo.clearValue                = context->clearValue;

            VkRenderingInfo renderingInfo          = {};
            renderingInfo.sType                    = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
            renderingInfo.renderArea.extent.width  = context->width;
            renderingInfo.renderArea.extent.height = context->height;
            renderingInfo.layerCount               = 1;
            renderingInfo.viewMask                 = 0;
            renderingInfo.colorAttachmentCount     = 1;
            renderingInfo.pColorAttachments        = &attachmentInfo;

            VulkanFuncs_vkCmdBeginRenderingKHR(fd->commandBuffer, &renderingInfo);
        }
        else
#endif
        {
            VkRenderPassBeginInfo info    = {};
            info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            info.renderPass               = context->renderPass;
            info.framebuffer              = fd->framebuffer;
            info.renderArea.extent.width  = context->width;
            info.renderArea.extent.height = context->height;
            info.clearValueCount          = (viewport->Flags & ImGuiViewportFlags_NoRendererClear) ? 0 : 1;
            info.pClearValues             = (viewport->Flags & ImGuiViewportFlags_NoRendererClear) ? nullptr : &context->clearValue;
            vkCmdBeginRenderPass(fd->commandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
        }
    }

    vkRenderImGuiDrawData(viewport->DrawData, fd->commandBuffer, bd->pipelineForViewports);

    {
#ifdef IMPL_VULKAN_HAS_DYNAMIC_RENDERING
        if (v->useDynamicRendering)
        {
            VulkanFuncs_vkCmdEndRenderingKHR(fd->commandBuffer);

            // Transition image to a layout suitable for presentation
            VkImageMemoryBarrier barrier        = {};
            barrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask               = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.oldLayout                   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.newLayout                   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.image                       = fd->backbuffer;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(fd->commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        }
        else
#endif
        {
            vkCmdEndRenderPass(fd->commandBuffer);
        }
        {
            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo info               = {};
            info.sType                      = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            info.waitSemaphoreCount         = 1;
            info.pWaitSemaphores            = &fsd->imageAcquiredSemaphore;
            info.pWaitDstStageMask          = &wait_stage;
            info.commandBufferCount         = 1;
            info.pCommandBuffers            = &fd->commandBuffer;
            info.signalSemaphoreCount       = 1;
            info.pSignalSemaphores          = &fsd->renderCompleteSemaphore;

            err = vkEndCommandBuffer(fd->commandBuffer);
            checkVkResultBD(err);
            err = vkResetFences(v->device, 1, &fd->fence);
            checkVkResultBD(err);
            err = vkQueueSubmit(v->queue, 1, &info, fd->fence);
            checkVkResultBD(err);
        }
    }
}

static void vkSwapImGuiViewportBuffers(ImGuiViewport* viewport, void*)
{
    VulkanData* bd         = vkGetBackendData();
    VulkanViewportData* vd = static_cast<VulkanViewportData*>(viewport->RendererUserData);
    VulkanContext* context = &vd->context;
    VulkanInitInfo* v      = &bd->vulkanInitInfo;

    if (vd->swapChainNeedRebuild) // Frame data became invalid in the middle of rendering
        return;

    VkResult err;
    U32 presentIndex = context->frameIndex;

    VulkanFrameSemaphores* fsd = &context->frameSemaphores[context->semaphoreIndex];
    VkPresentInfoKHR info      = {};
    info.sType                 = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount    = 1;
    info.pWaitSemaphores       = &fsd->renderCompleteSemaphore;
    info.swapchainCount        = 1;
    info.pSwapchains           = &context->swapchain;
    info.pImageIndices         = &presentIndex;
    err                        = vkQueuePresentKHR(v->queue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR)
    {
        vd->swapChainNeedRebuild = true;
        return;
    }
    if (err == VK_SUBOPTIMAL_KHR)
    {
        vd->swapChainSuboptimal = true;
    }
    else
    {
        checkVkResultBD(err);
    }

    context->semaphoreIndex = (context->semaphoreIndex + 1) % context->semaphoreCount; // Now we can use the next set of semaphores
}

void vkInitMultiViewportSupport()
{
    ImGuiPlatformIO& platformIO = ImGui::GetPlatformIO();
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        SAF_ASSERT(platformIO.Platform_CreateVkSurface != nullptr && "Platform needs to setup the CreateVkSurface handler.");
    platformIO.Renderer_CreateWindow  = vkCreateImGuiViewportContext;
    platformIO.Renderer_DestroyWindow = vkDestroyImGuiViewportContext;
    platformIO.Renderer_SetWindowSize = vkSetImGuiViewportContextSize;
    platformIO.Renderer_RenderWindow  = vkRenderImGuiViewportContext;
    platformIO.Renderer_SwapBuffers   = vkSwapImGuiViewportBuffers;
}

void vkShutdownMultiViewportSupport()
{
    ImGui::DestroyPlatformWindows();
}
