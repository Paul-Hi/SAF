/**
 * @file      application.cpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

// GLFW declares its Vulkan functions only if the Vulkan headers come first; keep this block separate so sorting cannot move it.
#include <volk.h>

#include <GLFW/glfw3.h>
#include <app/application.hpp>
#include <app/settings.hpp>
#include <chrono>
#include <core/helpers.hpp>
#include <core/vulkanSwapchain.hpp>
#include <implot.h>
#include <implot3d.h>
#include <limits>
#include <ui/guiStyle.hpp>
#include <ui/imguiBackend.hpp>

#ifdef SAF_VULKAN_COMPUTE
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <spirv_reflect.h>
#endif

#ifdef SAF_CUDA_INTEROP
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <VersionHelpers.h>
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#define _USE_MATH_DEFINES
#endif

using namespace saf;

static void glfwErrorCallback(I32 error, const char* description)
{
    std::cerr << "[glfw] Error " << error << ": " << description << '\n';
}

#ifdef SAF_VULKAN_COMPUTE
static Str describeBinding(const ComputeBinding& binding)
{
    return "'" + binding.name + "' (set " + std::to_string(binding.set) + ", binding " + std::to_string(binding.binding) + ")";
}

/**
 * @brief Reads the descriptor bindings and the push constant block size of a compute entry point.
 * @param[in] module The reflected shader module.
 * @param[in] entryPoint The name of the compute entry point.
 * @param[out] outBindings The reflected bindings, appended to.
 * @param[out] outPushConstantSize The push constant block size, 0 if the entry point has none.
 * @return VkResult VK_ERROR_FEATURE_NOT_PRESENT for an unsupported descriptor type or an array of descriptors, VK_ERROR_INITIALIZATION_FAILED if reflection fails or the entry point is missing or not a compute shader, else VK_SUCCESS.
 */
static VkResult reflectComputeShader(const SpvReflectShaderModule& module, const char* entryPoint, std::vector<ComputeBinding>& outBindings, U32& outPushConstantSize)
{
    // Uses the SPirVReflect lib (I think this is the smallest dependency to add? TODO maybe check that)
    const SpvReflectEntryPoint* entry = spvReflectGetEntryPoint(&module, entryPoint);
    if (entry == nullptr || entry->shader_stage != SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    U32 count = 0;
    if (spvReflectEnumerateEntryPointDescriptorBindings(&module, entryPoint, &count, nullptr) != SPV_REFLECT_RESULT_SUCCESS)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::vector<SpvReflectDescriptorBinding*> bindings(count);
    if (spvReflectEnumerateEntryPointDescriptorBindings(&module, entryPoint, &count, bindings.data()) != SPV_REFLECT_RESULT_SUCCESS)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    for (const SpvReflectDescriptorBinding* binding : bindings)
    {
        // SpvReflectDescriptorType mirrors VkDescriptorType's values
        const auto type = static_cast<VkDescriptorType>(binding->descriptor_type);
        if ((type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) || binding->count != 1)
        {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        // An anonymous GLSL block is named after its block type
        const char* name = binding->name;
        if ((name == nullptr || name[0] == '\0') && binding->type_description != nullptr)
        {
            name = binding->type_description->type_name;
        }

        outBindings.push_back({ .name = name != nullptr ? name : "", .set = binding->set, .binding = binding->binding, .type = type, .image = {}, .buffer = {} });
    }

    if (spvReflectEnumerateEntryPointPushConstantBlocks(&module, entryPoint, &count, nullptr) != SPV_REFLECT_RESULT_SUCCESS)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::vector<SpvReflectBlockVariable*> blocks(count);
    if (spvReflectEnumerateEntryPointPushConstantBlocks(&module, entryPoint, &count, blocks.data()) != SPV_REFLECT_RESULT_SUCCESS)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    // A shader stage has at most one push constant block per entry poiint (at least thats what the spec says) - so we can assume 0 or 1
    outPushConstantSize = blocks.empty() ? 0 : blocks[0]->size;

    return VK_SUCCESS;
}

VkResult saf::loadSpirv(const Str& path, std::vector<U32>& outWords) // we load in words instead of bytes, since the spec does it and it guarantees some alignment
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const std::streamsize size = file.tellg();
    if (size < 0 || size % sizeof(U32) != 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    outWords.resize(static_cast<size_t>(size) / sizeof(U32));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(outWords.data()), size))
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return VK_SUCCESS;
}

Str saf::shaderPath(const Str& fileName) // TODO: This is not the most elegant version - but it works most of the time for our usecases...
{
#ifdef WIN32
    // GetModuleFileNameW fills the whole buffer when it truncates
    std::wstring executable(MAX_PATH, L'\0');
    DWORD length;
    while ((length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()))) == executable.size())
    {
        executable.resize(2 * executable.size());
    }
    executable.resize(length);
    const std::filesystem::path executablePath(executable);
#else
    // argv[0] would not do: started through PATH it has no directory
    const std::filesystem::path executablePath = std::filesystem::read_symlink("/proc/self/exe");
#endif
    return (executablePath.parent_path() / "shaders" / fileName).string();
}
#endif

Application::Application(const ApplicationSettings& settings, bool loadFromPersistedSettings)
    : mName(settings.name)
    , mWindowWidth(settings.windowWidth)
    , mWindowHeight(settings.windowHeight)
    , mFontScale(settings.fontScale)
    , mTheme(settings.theme)
    , mClearColor(settings.clearColor)
    , mVSyncEnabled(settings.vSyncEnabled)
    , mMaxImages(settings.maxImages)
    , mRunning(true)
#ifdef SAF_VULKAN_COMPUTE
    , mMaxComputeDescriptorSets(settings.maxComputeDescriptorSets)
    , mAsyncComputeRequested(settings.asyncCompute)
#endif
#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
    , mMaxTimingPairs(settings.maxTimingPairs)
#endif
{
#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
    if (mMaxTimingPairs == 0)
    {
        reportMisuse("ApplicationSettings::maxTimingPairs must be at least 1.");
    }
#endif

    PersistentSettings persistedSettings;
    if (loadFromPersistedSettings)
    {
        if (loadPersistentSettings(persistedSettings, "saf.ini"))
        {
            mName         = persistedSettings.name;
            mWindowWidth  = persistedSettings.windowWidth;
            mWindowHeight = persistedSettings.windowHeight;
            mTheme        = persistedSettings.theme;
            mFontScale    = persistedSettings.fontScale;
            mClearColor   = persistedSettings.clearColor;
            mVSyncEnabled = persistedSettings.vSyncEnabled;
        }
    }

    storePersistentSettings(
        { .name         = mName,
          .windowWidth  = mWindowWidth,
          .windowHeight = mWindowHeight,
          .fontScale    = mFontScale,
          .theme        = mTheme,
          .clearColor   = mClearColor,
          .vSyncEnabled = mVSyncEnabled },
        "saf.ini");

    mVulkanContext   = std::make_unique<VulkanContext>();
    mVulkanSwapchain = std::make_unique<VulkanSwapchain>();
    mInitialized     = init();
}

Application::~Application()
{
    if (mLogicalDevice != VK_NULL_HANDLE)
    {
        VK_CHECK(vkDeviceWaitIdle(mLogicalDevice));
    }

    while (!mLayerStack.empty())
    {
        popLayer();
    }

    destroy();
}

bool Application::init()
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
    {
        std::cerr << "[glfw] Failed to initialize GLFW." << '\n';
        return false;
    }

    if (!glfwVulkanSupported())
    {
        std::cerr << "[glfw] Vulkan Not Supported." << '\n';
        return false;
    }

    U32 extensionsCount         = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionsCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + extensionsCount);

    constexpr auto queueGCT = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    constexpr auto queueT   = VK_QUEUE_TRANSFER_BIT;
    constexpr auto queueC   = VK_QUEUE_COMPUTE_BIT;

    ContextCreateInfo contextCreateInfo{
        .applicationName    = mName.c_str(),
        .apiVersion         = API_VERSION,
        .instanceExtensions = std::move(extensions),
        .deviceExtensions   = {
            { VK_KHR_SWAPCHAIN_EXTENSION_NAME, {} },
#ifdef SAF_CUDA_INTEROP
#ifdef WIN32
            { VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, {} },
            { VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME, {} },
#else
            { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, {} },
            { VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME, {} },
#endif
#endif
        },
        .queues = { queueGCT, queueT, queueC }
    };

    if (mVulkanContext->create(contextCreateInfo) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create Vulkan context." << '\n';
        return false;
    }

    mInstance       = mVulkanContext->getInstance();
    mPhysicalDevice = mVulkanContext->getPhysicalDevice();
    mLogicalDevice  = mVulkanContext->getLogicalDevice();

    mQueueGCT = mVulkanContext->getQueue(0);
    mQueueT   = mVulkanContext->getQueue(1);
    mQueueC   = mVulkanContext->getQueue(2);

#if defined(SAF_VULKAN_COMPUTE) and not defined(NDEBUG)
    std::cout << "[vulkan] Async compute: " << (hasAsyncCompute() ? "Yes" : "No") << '\n';
#endif

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE); // DPI aware
    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, mName.c_str(), nullptr, nullptr);

    if (!mWindow)
    {
        std::cerr << "[glfw] Failed to create GLFW window." << '\n';
        return false;
    }

    glfwSetWindowSize(mWindow, mWindowWidth, mWindowHeight);

    // Create Window Surface
    if (glfwCreateWindowSurface(mInstance, mWindow, nullptr, &mSurface) != VK_SUCCESS)
    {
        std::cerr << "[glfw] Failed to create window surface." << '\n';
        return false;
    }

    const SwapchainCreateInfo swapchainCreateInfo{
        .logicalDevice    = mLogicalDevice,
        .physicalDevice   = mPhysicalDevice,
        .queue            = mQueueGCT.queue,
        .queueFamilyIndex = mQueueGCT.familyIndex,
        .surface          = mSurface,
    };

    if (mVulkanSwapchain->create(swapchainCreateInfo) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create Vulkan swapchain." << '\n';
        return false;
    }

    VkExtent2D swapchainExtent = { static_cast<U32>(mWindowWidth), static_cast<U32>(mWindowHeight) };
    if (mVulkanSwapchain->update(swapchainExtent, mVSyncEnabled) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to update swapchain." << '\n';
        return false;
    }

    mWindowWidth  = swapchainExtent.width;
    mWindowHeight = swapchainExtent.height;

    // Create Command Pools ("transient")
    const VkCommandPoolCreateInfo commandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = mQueueGCT.familyIndex
    };
    if (vkCreateCommandPool(mLogicalDevice, &commandPoolCreateInfo, nullptr, &mCommandPool) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create command pool." << '\n';
        return false;
    }

    const VkCommandPoolCreateInfo transientCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = mQueueGCT.familyIndex
    };

    if (vkCreateCommandPool(mLogicalDevice, &transientCommandPoolCreateInfo, nullptr, &mTransientCommandPool) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create transient command pool." << '\n';
        return false;
    }

    // Create Command Buffers
    const VkCommandBufferAllocateInfo commandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = mCommandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = mVulkanSwapchain->getFramesInFlight()
    };

    mCommandBuffers.resize(mVulkanSwapchain->getFramesInFlight());
    if (vkAllocateCommandBuffers(mLogicalDevice, &commandBufferAllocateInfo, mCommandBuffers.data()) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to allocate command buffer." << '\n';
        return false;
    }

#ifdef SAF_VULKAN_COMPUTE
    // Compute command pool
    const VkCommandPoolCreateInfo computeCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = hasAsyncCompute() ? mQueueC.familyIndex : mQueueGCT.familyIndex
    };
    if (vkCreateCommandPool(mLogicalDevice, &computeCommandPoolCreateInfo, nullptr, &mComputeCommandPool) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create compute command pool." << '\n';
        return false;
    }

    // Compute command buffers
    const VkCommandBufferAllocateInfo computeCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = mComputeCommandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    // Compute descriptor pools (some random guess for that number - but I think 1024 (default) should be enough for our use cases)
    const U32 computeDescriptorsPerType           = 4 * mMaxComputeDescriptorSets;
    const VkDescriptorPoolSize computePoolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, computeDescriptorsPerType },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, computeDescriptorsPerType },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, computeDescriptorsPerType }
    };
    const VkDescriptorPoolCreateInfo computeDescriptorPoolCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = mMaxComputeDescriptorSets,
        .poolSizeCount = static_cast<U32>(std::size(computePoolSizes)),
        .pPoolSizes    = computePoolSizes
    };

    // Timestamp query pools
    const VkQueryPoolCreateInfo timestampQueryPoolCreateInfo{
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2 * mMaxTimingPairs
    };

    // Compute frame slots (holds the pools and cbs)
    mComputeSlots.resize(mVulkanSwapchain->getFramesInFlight());
    for (auto& slot : mComputeSlots)
    {
        if (vkAllocateCommandBuffers(mLogicalDevice, &computeCommandBufferAllocateInfo, &slot.commandBuffer) != VK_SUCCESS ||
            vkCreateDescriptorPool(mLogicalDevice, &computeDescriptorPoolCreateInfo, nullptr, &slot.descriptorPool) != VK_SUCCESS ||
            vkCreateQueryPool(mLogicalDevice, &timestampQueryPoolCreateInfo, nullptr, &slot.queryPool) != VK_SUCCESS)
        {
            std::cerr << "[vulkan] Failed to create compute frame slots." << '\n';
            return false;
        }
    }

    // Checks for the time queries
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &physicalDeviceProperties);
    std::ranges::copy(physicalDeviceProperties.limits.maxComputeWorkGroupCount, mMaxComputeWorkGroupCount.begin());
    mTimestampPeriod = physicalDeviceProperties.limits.timestampPeriod;

    U32 queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, queueFamilies.data());
    const U32 timestampValidBits = queueFamilies[hasAsyncCompute() ? mQueueC.familyIndex : mQueueGCT.familyIndex].timestampValidBits;
    mTimestampMask               = timestampValidBits == 64 ? std::numeric_limits<U64>::max() : (U64(1) << timestampValidBits) - 1;
    if (timestampValidBits == 0)
    {
        std::cout << "[vulkan] Timing scopes: unavailable, the compute queue writes no timestamps." << '\n';
    }

#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
    // timeline semaphores - standard for >= Vulkan 1.2 and >= CUDA 11.2
    VkSemaphoreTypeCreateInfo timelineCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue  = 0
    };
    const VkSemaphoreCreateInfo timelineSemaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &timelineCreateInfo
    };
#ifdef SAF_VULKAN_COMPUTE
    if (vkCreateSemaphore(mLogicalDevice, &timelineSemaphoreCreateInfo, nullptr, &mComputeTimeline) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create timeline semaphores." << '\n';
        return false;
    }
#endif

#ifdef SAF_CUDA_INTEROP
    // CUDA waits on the graphics timeline and signals its own, so both are exported
#ifdef WIN32
    const VkExternalSemaphoreHandleTypeFlagBits timelineHandleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    const cudaExternalSemaphoreHandleType cudaTimelineHandleType   = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
#else
    const VkExternalSemaphoreHandleTypeFlagBits timelineHandleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    const cudaExternalSemaphoreHandleType cudaTimelineHandleType   = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
#endif

    // Queried before the export info joins the chain, which the query does not accept
    const VkPhysicalDeviceExternalSemaphoreInfo externalTimelineInfo{
        .sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
        .pNext      = &timelineCreateInfo,
        .handleType = timelineHandleType
    };
    VkExternalSemaphoreProperties externalTimelineProperties{ .sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES };
    vkGetPhysicalDeviceExternalSemaphoreProperties(mPhysicalDevice, &externalTimelineInfo, &externalTimelineProperties);

    I32 cudaDevice            = 0;
    I32 cudaTimelineSupported = 0;
    CUDA_CHECK(cudaGetDevice(&cudaDevice));
    CUDA_CHECK(cudaDeviceGetAttribute(&cudaTimelineSupported, cudaDevAttrTimelineSemaphoreInteropSupported, cudaDevice));

    if (!(externalTimelineProperties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) || !cudaTimelineSupported)
    {
        std::cerr << "[vulkan] Error: CUDA interop needs timeline semaphores shared with CUDA, which this device does not support." << '\n';
        return false;
    }

#ifdef WIN32
    // Default security, not inherited by child processes; CUDA imports the handle in this process
    VkExportSemaphoreWin32HandleInfoKHR exportSemaphoreWin32HandleInfoKHR{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
        .pAttributes = nullptr,
        .dwAccess    = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        .name        = (LPCWSTR)NULL
    };
#endif
    VkExportSemaphoreCreateInfo exportTimelineCreateInfo{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
        .handleTypes = static_cast<VkExternalSemaphoreHandleTypeFlags>(timelineHandleType)
    };
#ifdef WIN32
    exportTimelineCreateInfo.pNext = &exportSemaphoreWin32HandleInfoKHR;
#endif
    timelineCreateInfo.pNext = &exportTimelineCreateInfo;

    if (vkCreateSemaphore(mLogicalDevice, &timelineSemaphoreCreateInfo, nullptr, &mCudaTimeline) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create timeline semaphores." << '\n';
        return false;
    }
#endif

    if (vkCreateSemaphore(mLogicalDevice, &timelineSemaphoreCreateInfo, nullptr, &mGraphicsTimeline) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create timeline semaphores." << '\n';
        return false;
    }

#ifdef SAF_CUDA_INTEROP
    const auto importTimeline = [&](VkSemaphore semaphore, cudaExternalSemaphore_t& cudaSemaphore)
    {
        cudaExternalSemaphoreHandleDesc handleDesc{};
        handleDesc.type = cudaTimelineHandleType;
#ifdef WIN32
        handleDesc.handle.win32.handle = getSemaphoreHandle(semaphore);
#else
        handleDesc.handle.fd = static_cast<I32>(reinterpret_cast<uintptr_t>(getSemaphoreHandle(semaphore)));
#endif
        CUDA_CHECK(cudaImportExternalSemaphore(&cudaSemaphore, &handleDesc));
#ifdef WIN32
        // Unlike a file descriptor, CUDA does not take ownership of the NT handle
        CloseHandle(handleDesc.handle.win32.handle);
#endif
    };
    importTimeline(mGraphicsTimeline, mGraphicsTimelineCuda);
    importTimeline(mCudaTimeline, mCudaTimelineCuda);
#endif
#endif

#ifdef SAF_CUDA_INTEROP
    // CUDA timing frame slots
    mCudaTimingSlots.resize(mVulkanSwapchain->getFramesInFlight());
    for (auto& slot : mCudaTimingSlots)
    {
        slot.events.resize(2 * mMaxTimingPairs);
        for (cudaEvent_t& event : slot.events)
        {
            CUDA_CHECK(cudaEventCreate(&event));
        }
    }
#endif

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImPlot3D::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Enable Docking
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;   // Enable Multi-Viewport / Platform Windows (disabled for now, since Wayland and GLFW do not like each other yet)
    // io.ConfigViewportsNoAutoMerge = true;
    // io.ConfigViewportsNoTaskBarIcon = true;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(mWindow, true);

    const VkFormat swapChainFormat = mVulkanSwapchain->getSurfaceFormat();

    const VkPipelineRenderingCreateInfo pipelineRenderingInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &swapChainFormat,
    };

    ImGui_ImplVulkan_InitInfo initInfo{
        .ApiVersion         = API_VERSION,
        .Instance           = mInstance,
        .PhysicalDevice     = mPhysicalDevice,
        .Device             = mLogicalDevice,
        .QueueFamily        = mQueueGCT.familyIndex,
        .Queue              = mQueueGCT.queue,
        .DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE + mMaxImages,
        .MinImageCount      = mVulkanSwapchain->getImageCount(),
        .ImageCount         = mVulkanSwapchain->getImageCount(),
        .PipelineInfoMain{
            .PipelineRenderingCreateInfo = pipelineRenderingInfo },
        .PipelineInfoForViewports{
            .PipelineRenderingCreateInfo = pipelineRenderingInfo },
        .UseDynamicRendering = true
    };

    ImGui_ImplVulkan_Init(&initInfo);

    float xScale, yScale;
    glfwGetWindowContentScale(mWindow, &xScale, &yScale);

    // Setup Dear ImGui style
    if (mTheme == 0)
    {
        setupImGuiStyleDark(mFontScale, xScale);
    }
    else
    {
        setupImGuiStyleLight(mFontScale, xScale);
    }

    ImGuiStyle& style = ImGui::GetStyle();

    style.FontScaleDpi = xScale;

    return true;
}

void Application::destroy()
{
    if (mLogicalDevice != VK_NULL_HANDLE)
    {
        VK_CHECK(vkDeviceWaitIdle(mLogicalDevice));
    }

    for (const auto& image : mImages)
    {
        destroyImageResources(image.second);
    }

    mImages.clear();

#ifdef SAF_VULKAN_COMPUTE
    for (const auto& buffer : mBuffers)
    {
        destroyBufferResources(buffer.second);
    }

    mBuffers.clear();

    for (const auto& pipeline : mComputePipelines)
    {
        destroyComputePipelineResources(pipeline.second);
    }

    mComputePipelines.clear();
#endif

    if (mInitialized)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot3D::DestroyContext();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }

    if (mCommandPool != VK_NULL_HANDLE)
    {
        vkFreeCommandBuffers(mLogicalDevice, mCommandPool, static_cast<U32>(mCommandBuffers.size()), mCommandBuffers.data());
        vkDestroyCommandPool(mLogicalDevice, mCommandPool, nullptr);
    }

    if (mTransientCommandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(mLogicalDevice, mTransientCommandPool, nullptr);
    }

#ifdef SAF_VULKAN_COMPUTE
    for (const auto& slot : mComputeSlots)
    {
        if (slot.descriptorPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(mLogicalDevice, slot.descriptorPool, nullptr);
        }

        if (slot.queryPool != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(mLogicalDevice, slot.queryPool, nullptr);
        }
    }

    if (mComputeCommandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(mLogicalDevice, mComputeCommandPool, nullptr);
    }

    if (mComputeTimeline != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(mLogicalDevice, mComputeTimeline, nullptr);
    }
#endif

#ifdef SAF_CUDA_INTEROP
    for (const auto& slot : mCudaTimingSlots)
    {
        for (cudaEvent_t event : slot.events)
        {
            CUDA_CHECK(cudaEventDestroy(event));
        }
    }

    // release cuda  imports before destroying - might be weird otherwise?
    if (mGraphicsTimelineCuda)
    {
        CUDA_CHECK(cudaDestroyExternalSemaphore(mGraphicsTimelineCuda));
    }
    if (mCudaTimelineCuda)
    {
        CUDA_CHECK(cudaDestroyExternalSemaphore(mCudaTimelineCuda));
    }
    if (mCudaTimeline != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(mLogicalDevice, mCudaTimeline, nullptr);
    }
#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
    if (mGraphicsTimeline != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(mLogicalDevice, mGraphicsTimeline, nullptr);
    }
#endif

    mVulkanSwapchain->destroy();

    if (mSurface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
    }

    mVulkanContext->destroy();

    if (mWindow)
    {
        glfwDestroyWindow(mWindow);
    }
    glfwTerminate();
}

bool Application::run()
{
    if (!mInitialized)
    {
        return false;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    const bool res = [&]()
    {
        const VkQueue& queue = mQueueGCT.queue;

        while (!glfwWindowShouldClose(mWindow) && mRunning)
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            F32 dt           = std::chrono::duration<F32, std::chrono::seconds::period>(currentTime - startTime).count();
            startTime        = currentTime;

            std::vector<std::unique_ptr<Layer>> pendingLayerOps;
            pendingLayerOps.swap(mPendingLayerOps);
            for (auto& op : pendingLayerOps)
            {
                if (op)
                {
                    pushLayer(std::move(op));
                }
                else
                {
                    popLayer();
                }
            }

            glfwPollEvents();

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

#ifdef SAF_CUDA_INTEROP
            // CUDA work in onUpdate may write images the last frame submit still samples
            if (hasCudaSharedImage())
            {
                waitForVulkanCompletion(mGraphicsTimelineCuda, mGraphicsTimelineValue);
            }
#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
            collectTimings();
#endif

            mLayerPhase = LayerPhase::Update;
            for (auto& layer : mLayerStack)
            {
                layer->onUpdate(this, dt);
            }
            mLayerPhase = LayerPhase::None;

#ifdef SAF_VULKAN_COMPUTE
            if (mComputeRecording)
            {
                checkTimingPairsClosed(mComputeSlots[mFrameSlotIndex].timingPairs);

                const VkCommandBuffer computeCommandBuffer = mComputeSlots[mFrameSlotIndex].commandBuffer;
                VK_CHECK_RETURN_BOOL(vkEndCommandBuffer(computeCommandBuffer));

                const VkSemaphoreSubmitInfo computeWait{
                    .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .semaphore = mGraphicsTimeline,
                    .value     = mGraphicsTimelineValue,
                    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                };
                const VkSemaphoreSubmitInfo computeSignal{
                    .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .semaphore = mComputeTimeline,
                    .value     = ++mComputeTimelineValue,
                    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                };
                const VkCommandBufferSubmitInfo computeCommandBufferInfo{
                    .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                    .commandBuffer = computeCommandBuffer
                };
                const VkSubmitInfo2 computeSubmitInfo{
                    .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                    .waitSemaphoreInfoCount   = 1,
                    .pWaitSemaphoreInfos      = &computeWait,
                    .commandBufferInfoCount   = 1,
                    .pCommandBufferInfos      = &computeCommandBufferInfo,
                    .signalSemaphoreInfoCount = 1,
                    .pSignalSemaphoreInfos    = &computeSignal
                };

                const VkQueue computeQueue = hasAsyncCompute() ? mQueueC.queue : mQueueGCT.queue;
                VK_CHECK_RETURN_BOOL(vkQueueSubmit2(computeQueue, 1, &computeSubmitInfo, VK_NULL_HANDLE));

                mComputeSlots[mFrameSlotIndex].timelineValue = mComputeTimelineValue;
                mComputeRecording                            = false;
            }
#endif

#ifdef SAF_CUDA_INTEROP
            checkTimingPairsClosed(mCudaTimingSlots[mFrameSlotIndex].timingPairs);
#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
            mFrameSlotIndex = (mFrameSlotIndex + 1) % mVulkanSwapchain->getFramesInFlight();
#endif

#ifdef SAF_CUDA_INTEROP
            // Checked again: an image created in onUpdate may already hold CUDA writes the frame submit must wait for
            if (hasCudaSharedImage())
            {
                signalCudaCompletion(mCudaTimelineCuda, ++mCudaTimelineValue);
            }
#endif

            I32 width, height;
            glfwGetFramebufferSize(mWindow, &width, &height);

            if (width == 0 || height == 0)
            {
                ImGui::EndFrame();
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }

            if (mWindowWidth != width || mWindowHeight != height)
            {
                mWindowWidth  = width;
                mWindowHeight = height;
                mVulkanSwapchain->requestRebuild();
            }

            if (mVulkanSwapchain->needsRebuild())
            {
                VkExtent2D swapchainExtent = { static_cast<U32>(mWindowWidth), static_cast<U32>(mWindowHeight) };
                if (mVulkanSwapchain->update(swapchainExtent, mVSyncEnabled) != VK_SUCCESS)
                {
                    std::cerr << "[vulkan] Failed to update swapchain." << '\n';
                    return false;
                }

                mWindowWidth  = swapchainExtent.width;
                mWindowHeight = swapchainExtent.height;

                float xScale, yScale;
                glfwGetWindowContentScale(mWindow, &xScale, &yScale);

                if (mTheme == 0)
                {
                    setupImGuiStyleDark(mFontScale, xScale);
                }
                else
                {
                    setupImGuiStyleLight(mFontScale, xScale);
                }

                storePersistentSettings(
                    { .name         = mName,
                      .windowWidth  = mWindowWidth,
                      .windowHeight = mWindowHeight,
                      .fontScale    = mFontScale,
                      .theme        = mTheme,
                      .clearColor   = mClearColor,
                      .vSyncEnabled = mVSyncEnabled },
                    "saf.ini");
            }

            if (glfwGetWindowAttrib(mWindow, GLFW_ICONIFIED) != 0)
            {
                ImGui::EndFrame();
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }

            const VkResult res = mVulkanSwapchain->acquire();

            if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)
            {

                const VkFence inFlightFence        = mVulkanSwapchain->getCurrentInFlightFence();
                const VkSemaphore acquireSemaphore = mVulkanSwapchain->getCurrentAcquireSemaphore();
                const VkSemaphore presentSemaphore = mVulkanSwapchain->getCurrentPresentCompleteSemaphore();

                const VkCommandBuffer commandBuffer = mCommandBuffers[mVulkanSwapchain->getCurrentFrameIndex()];

                VK_CHECK_RETURN_BOOL(vkResetCommandBuffer(commandBuffer, 0));

                const VkCommandBufferBeginInfo beginInfo{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
                };

                VK_CHECK_RETURN_BOOL(vkBeginCommandBuffer(commandBuffer, &beginInfo));

                // UI
                {
                    // Dockspace
                    ImGuiViewport* viewport = ImGui::GetMainViewport();
                    ImGui::SetNextWindowPos(viewport->WorkPos);
                    ImGui::SetNextWindowSize(viewport->WorkSize);
                    ImGui::SetNextWindowViewport(viewport->ID);
                    static ImGuiDockNodeFlags dockNodeFlags = ImGuiDockNodeFlags_PassthruCentralNode;

                    static ImGuiWindowFlags dockingWindowFlags;
                    dockingWindowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
                    dockingWindowFlags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

                    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
                    ImGui::Begin("DockSpace", nullptr, dockingWindowFlags);
                    ImGui::PopStyleVar(3);

                    // Real Dockspace
                    ImGuiID dockspaceId = ImGui::GetID("SAFDockSpace");
                    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), dockNodeFlags);

                    // Menu bar
                    if (ImGui::BeginMainMenuBar())
                    {
                        if (ImGui::BeginMenu("File"))
                        {
                            if (ImGui::MenuItem("Exit"))
                            {
                                close();
                            }
                            ImGui::EndMenu();
                        }

                        if (ImGui::BeginMenu("View"))
                        {
                            float xScale, yScale;
                            glfwGetWindowContentScale(mWindow, &xScale, &yScale);

                            if (ImGui::MenuItem("Dark Theme", nullptr, mTheme == 0))
                            {
                                mTheme = 0;
                                setupImGuiStyleDark(mFontScale, xScale);

                                storePersistentSettings(
                                    { .name         = mName,
                                      .windowWidth  = mWindowWidth,
                                      .windowHeight = mWindowHeight,
                                      .fontScale    = mFontScale,
                                      .theme        = mTheme,
                                      .clearColor   = mClearColor,
                                      .vSyncEnabled = mVSyncEnabled },
                                    "saf.ini");
                            }

                            if (ImGui::MenuItem("Light Theme", nullptr, mTheme == 1))
                            {
                                mTheme = 1;
                                setupImGuiStyleLight(mFontScale, xScale);

                                storePersistentSettings(
                                    { .name         = mName,
                                      .windowWidth  = mWindowWidth,
                                      .windowHeight = mWindowHeight,
                                      .fontScale    = mFontScale,
                                      .theme        = mTheme,
                                      .clearColor   = mClearColor,
                                      .vSyncEnabled = mVSyncEnabled },
                                    "saf.ini");
                            }

                            if (ImGui::MenuItem("Toggle VSync", nullptr, mVSyncEnabled))
                            {
                                mVSyncEnabled = !mVSyncEnabled;
                                mVulkanSwapchain->requestRebuild();

                                storePersistentSettings(
                                    { .name         = mName,
                                      .windowWidth  = mWindowWidth,
                                      .windowHeight = mWindowHeight,
                                      .fontScale    = mFontScale,
                                      .theme        = mTheme,
                                      .clearColor   = mClearColor,
                                      .vSyncEnabled = mVSyncEnabled },
                                    "saf.ini");
                            }
                            ImGui::EndMenu();
                        }

                        if (mMenubarCallback)
                        {
                            mMenubarCallback();
                        }

                        ImGui::EndMainMenuBar();
                    }

                    mLayerPhase = LayerPhase::UIRender;
                    for (auto& layer : mLayerStack)
                    {
                        layer->onUIRender(this);
                    }
                    mLayerPhase = LayerPhase::None;

                    ImGui::End();
                }

                ImGui::EndFrame();
                ImGui::Render();

                // Main Draw
                {

                    const VkRenderingAttachmentInfo colorAttachment{
                        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                        .imageView   = mVulkanSwapchain->getCurrentImage().imageView,
                        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
                        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
                        .clearValue  = { .color = { mClearColor.x(), mClearColor.y(), mClearColor.z(), mClearColor.w() } }
                    };

                    const VkRenderingInfo renderingInfo{
                        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
                        .renderArea           = { .offset = { 0, 0 }, .extent = { static_cast<U32>(mWindowWidth), static_cast<U32>(mWindowHeight) } },
                        .layerCount           = 1,
                        .viewMask             = 0,
                        .colorAttachmentCount = 1,
                        .pColorAttachments    = &colorAttachment,
                    };

                    transitionImageLayout(mVulkanSwapchain->getCurrentImage().image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_NONE, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, commandBuffer);

                    vkCmdBeginRendering(commandBuffer, &renderingInfo);

                    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

                    vkCmdEndRendering(commandBuffer);

                    transitionImageLayout(mVulkanSwapchain->getCurrentImage().image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, commandBuffer);
                }

                VK_CHECK_RETURN_BOOL(vkEndCommandBuffer(commandBuffer));

                // FIXME: Inefficient?
                std::vector<VkSemaphoreSubmitInfo> waitSemaphores, signalSemaphores;
                waitSemaphores.push_back({ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = acquireSemaphore, .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT });
                signalSemaphores.push_back({ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = presentSemaphore, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });

#ifdef SAF_CUDA_INTEROP
                // Already satisfied when CUDA signalled nothing since the previous frame submit
                waitSemaphores.push_back({ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = mCudaTimeline, .value = mCudaTimelineValue, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
#endif

#ifdef SAF_VULKAN_COMPUTE
                waitSemaphores.push_back({ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = mComputeTimeline, .value = mComputeTimelineValue, .stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT });
#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
                signalSemaphores.push_back({ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = mGraphicsTimeline, .value = ++mGraphicsTimelineValue, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
#endif

                const VkCommandBufferSubmitInfo commandBufferInfo{
                    .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                    .commandBuffer = commandBuffer
                };

                const VkSubmitInfo2 submitInfo{
                    .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                    .waitSemaphoreInfoCount   = static_cast<U32>(waitSemaphores.size()),
                    .pWaitSemaphoreInfos      = waitSemaphores.data(),
                    .commandBufferInfoCount   = 1,
                    .pCommandBufferInfos      = &commandBufferInfo,
                    .signalSemaphoreInfoCount = static_cast<U32>(signalSemaphores.size()),
                    .pSignalSemaphoreInfos    = signalSemaphores.data()
                };

                VK_CHECK_RETURN_BOOL(vkQueueSubmit2(queue, 1, &submitInfo, inFlightFence));

                ImGuiIO& io = ImGui::GetIO();
                (void)io;
                if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
                {
                    ImGui::UpdatePlatformWindows();
                    ImGui::RenderPlatformWindowsDefault();
                }

                const VkResult presentResult = mVulkanSwapchain->present();
                if (presentResult != VK_ERROR_OUT_OF_DATE_KHR)
                {
                    VK_CHECK_RETURN_BOOL(presentResult);
                }
            }
            else
            {
                ImGui::EndFrame();
            }
        }

        return true;
    }();

    vkDeviceWaitIdle(mLogicalDevice);

    return res;
}

void Application::close()
{
    mRunning = false;
}

void Application::popLayer()
{
    if (mLayerPhase != LayerPhase::None)
    {
        mPendingLayerOps.push_back(nullptr);
        return;
    }

    if (mLayerStack.empty())
        return;

    mLayerStack.back()->onDetach(this);
    mLayerStack.pop_back();
}

VkResult Application::executeSingleTimeCommandBuffer(const std::function<VkResult(VkCommandBuffer)>& immediateFunction)
{
    VkCommandBuffer commandBuffer;

    const VkCommandBufferAllocateInfo commandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = mTransientCommandPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VK_CHECK_RETURN(vkAllocateCommandBuffers(mLogicalDevice, &commandBufferAllocateInfo, &commandBuffer));

    VK_CHECK_RETURN(vkResetCommandPool(mLogicalDevice, mTransientCommandPool, 0));

    const VkCommandBufferBeginInfo commandBufferBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    VK_CHECK_RETURN(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo));

    VK_CHECK_RETURN(immediateFunction(commandBuffer));

    VK_CHECK_RETURN(vkEndCommandBuffer(commandBuffer));

    const VkSubmitInfo submitInfo{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &commandBuffer
    };

    VK_CHECK_RETURN(vkQueueSubmit(mQueueGCT.queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK_RETURN(vkQueueWaitIdle(mQueueGCT.queue));

    vkFreeCommandBuffers(mLogicalDevice, mTransientCommandPool, 1, &commandBuffer);

    return VK_SUCCESS;
}

VkResult Application::createImage(const VulkanImageCreateInfo& createInfo, ImageHandle& outImage)
{
    VulkanImage image;
    image.width   = createInfo.width;
    image.height  = createInfo.height;
    image.format  = createInfo.format;
    image.sharing = createInfo.sharing;

    const VkResult result = createImageResources(image, createInfo);
    if (result < 0)
    {
        destroyImageResources(image);
        return result;
    }

    outImage = mImages.add(image);

    return VK_SUCCESS;
}

VkResult Application::createImageResources(VulkanImage& image, const VulkanImageCreateInfo& createInfo)
{
    VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = createInfo.format,
        .extent        = { createInfo.width, createInfo.height, 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = createInfo.usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, // We need the trasnfer src bit for downloadImage
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

#ifdef SAF_CUDA_INTEROP
    VkExternalMemoryImageCreateInfo externalMemoryImageCreateInfo{};
    if (createInfo.sharing == ImageSharing::Cuda)
    {
        imageCreateInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT; // For CUDA read/write access
        imageCreateInfo.flags |= VK_IMAGE_CREATE_ALIAS_BIT;  // Allow aliasing for CUDA interop

#ifdef WIN32
        externalMemoryImageCreateInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        externalMemoryImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
        externalMemoryImageCreateInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        externalMemoryImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
        imageCreateInfo.pNext = &externalMemoryImageCreateInfo;
    }
#endif

#ifdef SAF_VULKAN_COMPUTE
    const U32 computeQueueFamilies[] = { mQueueGCT.familyIndex, mQueueC.familyIndex };
    if (createInfo.sharing == ImageSharing::Compute)
    {
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(mPhysicalDevice, createInfo.format, &formatProperties);
        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
        {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }

        imageCreateInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;

        // CONCURRENT needs two distinct families; without async compute the image stays EXCLUSIVE to graphics (which is equal to compute in that case (GCT))
        if (hasAsyncCompute())
        {
            imageCreateInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
            imageCreateInfo.queueFamilyIndexCount = 2;
            imageCreateInfo.pQueueFamilyIndices   = computeQueueFamilies;
        }
    }
#endif

    VK_CHECK_RETURN(vkCreateImage(mLogicalDevice, &imageCreateInfo, nullptr, &image.image));

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(mLogicalDevice, image.image, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = memoryRequirements.size,
        .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    };

#ifdef SAF_CUDA_INTEROP

    VkExportMemoryAllocateInfo exportMemoryAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO
    };

#ifdef WIN32
    // Default security... CUDA imports the handle in this process
    const VkExportMemoryWin32HandleInfoKHR exportMemoryWin32HandleInfo{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
        .pAttributes = nullptr,
        .dwAccess    = GENERIC_ALL
    };
#endif
    if (createInfo.sharing == ImageSharing::Cuda)
    {
#ifdef WIN32
        exportMemoryAllocateInfo.pNext       = &exportMemoryWin32HandleInfo;
        exportMemoryAllocateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
#else
        exportMemoryAllocateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
        memoryAllocateInfo.pNext = &exportMemoryAllocateInfo;
    }
#endif

    VK_CHECK_RETURN(vkAllocateMemory(mLogicalDevice, &memoryAllocateInfo, nullptr, &image.deviceMemory));
    VK_CHECK_RETURN(vkBindImageMemory(mLogicalDevice, image.image, image.deviceMemory, 0));

#ifdef SAF_CUDA_INTEROP
    if (createInfo.sharing == ImageSharing::Cuda)
    {
        cudaExternalMemoryHandleDesc handleDesc{};
        handleDesc.size = memoryRequirements.size;

#ifdef WIN32
        handleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
        handleDesc.handle.win32.handle = getMemoryHandle(image.deviceMemory);
#else
        handleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
        handleDesc.handle.fd = static_cast<I32>(reinterpret_cast<uintptr_t>(getMemoryHandle(image.deviceMemory)));
#endif

        CUDA_CHECK(cudaImportExternalMemory(&image.cudaExternalImageMemory, &handleDesc));
#ifdef WIN32
        // CUDA does not take ownership of the handle
        CloseHandle(handleDesc.handle.win32.handle);
#endif

        const cudaExternalMemoryMipmappedArrayDesc arrayDesc{
            .formatDesc = getFormatDescriptor(createInfo.format),
            .extent     = { createInfo.width, createInfo.height, 0 },
            .flags      = cudaArraySurfaceLoadStore,
            .numLevels  = 1
        };

        CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&image.cudaMipmappedImageArray, image.cudaExternalImageMemory, &arrayDesc));

        cudaArray_t cudaMipmappedLevelArray;
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&cudaMipmappedLevelArray, image.cudaMipmappedImageArray, 0));

        const cudaResourceDesc surfaceResourceDesc{
            .resType = cudaResourceTypeArray,
            .res     = { .array{ .array = cudaMipmappedLevelArray } },
        };

        CUDA_CHECK(cudaCreateSurfaceObject(&image.cudaSurfaceObject, &surfaceResourceDesc));

        const cudaResourceDesc textureResourceDesc{
            .resType = cudaResourceTypeMipmappedArray,
            .res     = { .mipmap{ .mipmap = image.cudaMipmappedImageArray } },
        };

        const cudaTextureDesc textureDesc{
            .addressMode      = { cudaAddressModeWrap, cudaAddressModeWrap, cudaAddressModeWrap },
            .filterMode       = cudaFilterModePoint,
            .readMode         = cudaReadModeElementType,
            .normalizedCoords = 1,
            .mipmapFilterMode = cudaFilterModePoint
        };

        CUDA_CHECK(cudaCreateTextureObject(&image.cudaTextureObject, &textureResourceDesc, &textureDesc, nullptr));
    }
#endif

    // Image View
    const VkImageViewCreateInfo imageViewCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image            = image.image,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = createInfo.format,
        .components       = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1 }
    };

    VK_CHECK_RETURN(vkCreateImageView(mLogicalDevice, &imageViewCreateInfo, nullptr, &image.imageView));

    // Sampler
    const VkSamplerCreateInfo samplerCreateInfo{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_LINEAR,
        .minFilter    = VK_FILTER_LINEAR,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    };

    VK_CHECK_RETURN(vkCreateSampler(mLogicalDevice, &samplerCreateInfo, nullptr, &image.sampler));

    VkImageLayout displayLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
#ifdef SAF_VULKAN_COMPUTE
    if (createInfo.sharing == ImageSharing::Compute)
    {
        // Compute-shared images stay in GENERAL for their entire lifetime (its easier and seems to not have performance disadvantages)
        VK_CHECK_RETURN(transitionImageLayout(image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_NONE, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT));
        displayLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    else
#endif
    {
        // Other images are in SHADER_READ_ONLY between calls, which downloadImage relies on
        VK_CHECK_RETURN(transitionImageLayout(image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_NONE, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT));
    }

    image.descriptorSet = ImGui_ImplVulkan_AddTexture(image.imageView, displayLayout);

    return VK_SUCCESS;
}

void Application::destroyImage(ImageHandle image)
{
    const VulkanImage* vulkanImage = mImages.find(image);
    if (vulkanImage == nullptr)
    {
        return;
    }
#ifdef SAF_VULKAN_COMPUTE
    // A dispatch recorded this frame may use the image, and it is only submitted after onUpdate
    if (vulkanImage->sharing == ImageSharing::Compute && mComputeRecording)
    {
        reportMisuse("destroyImage() on a shared-with-compute image after getComputeCommandBuffer() in the same onUpdate.");
    }
#endif
    // The image may still be sampled by frames in flight.
    VK_CHECK(vkDeviceWaitIdle(mLogicalDevice));
    destroyImageResources(*vulkanImage);
    mImages.erase(image);
}

void Application::destroyImageResources(const VulkanImage& vulkanImage)
{
    // Handles may be null when called on a partially created image so we need to check it
    if (vulkanImage.descriptorSet != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkan_RemoveTexture(vulkanImage.descriptorSet);
    }

    vkDestroySampler(mLogicalDevice, vulkanImage.sampler, nullptr);
    vkDestroyImageView(mLogicalDevice, vulkanImage.imageView, nullptr);
    vkDestroyImage(mLogicalDevice, vulkanImage.image, nullptr);
    vkFreeMemory(mLogicalDevice, vulkanImage.deviceMemory, nullptr);

#ifdef SAF_CUDA_INTEROP
    if (vulkanImage.sharing == ImageSharing::Cuda)
    {
        if (vulkanImage.cudaTextureObject)
        {
            CUDA_CHECK(cudaDestroyTextureObject(vulkanImage.cudaTextureObject));
        }
        if (vulkanImage.cudaSurfaceObject)
        {
            CUDA_CHECK(cudaDestroySurfaceObject(vulkanImage.cudaSurfaceObject));
        }
        if (vulkanImage.cudaMipmappedImageArray)
        {
            CUDA_CHECK(cudaFreeMipmappedArray(vulkanImage.cudaMipmappedImageArray));
        }
        if (vulkanImage.cudaExternalImageMemory)
        {
            CUDA_CHECK(cudaDestroyExternalMemory(vulkanImage.cudaExternalImageMemory));
        }
    }
#endif
}

VkResult Application::uploadImage(ImageHandle image, const void* data, size_t size)
{
    const VulkanImage* found = mImages.find(image);
    if (found == nullptr)
    {
        return VK_ERROR_UNKNOWN;
    }
    const VulkanImage& vulkanImage = *found;

    // check size of data against image size
    const size_t imageSize = vulkanImage.width * vulkanImage.height * getFormatBytesPerPixel(vulkanImage.format);
    if (size != imageSize)
    {
        return VK_ERROR_UNKNOWN;
    }

    VK_CHECK_RETURN(waitForSharedImageWork(vulkanImage));

    bool sharedWithCompute = false;
#ifdef SAF_VULKAN_COMPUTE
    sharedWithCompute = vulkanImage.sharing == ImageSharing::Compute;
#endif
    // Compute-shared images are copied in GENERAL, which they never leave
    const VkImageLayout copyLayout = sharedWithCompute ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    return uploadThroughStagingBuffer(data, size, [&](VkCommandBuffer commandBuffer, VkBuffer stagingBuffer)
                                      {
                                        if (sharedWithCompute)
                                        {
                                            // this acts as a barrier
                                            transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, commandBuffer);
                                        }
                                        else
                                        {
                                            transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, commandBuffer);
                                        }

                                        const VkBufferImageCopy bufferImageCopy{
                                            .bufferOffset      = 0,
                                            .bufferRowLength   = 0,
                                            .bufferImageHeight = 0,
                                            .imageSubresource  = {
                                                 .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                                 .mipLevel       = 0,
                                                 .baseArrayLayer = 0,
                                                 .layerCount     = 1 },
                                            .imageOffset = { 0, 0, 0 },
                                            .imageExtent = { vulkanImage.width, vulkanImage.height, 1 }
                                        };

                                        vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, vulkanImage.image, copyLayout, 1, &bufferImageCopy);

                                        if (sharedWithCompute)
                                        {
                                            // this acts as a barrier
                                            transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, commandBuffer);
                                        }
                                        else
                                        {
                                            transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, commandBuffer);
                                        } });
}

VkResult Application::downloadImage(ImageHandle image, void* data, size_t size)
{
    const VulkanImage* found = mImages.find(image);
    if (found == nullptr)
    {
        return VK_ERROR_UNKNOWN;
    }
    const VulkanImage& vulkanImage = *found;

    // check size of data against image size
    const size_t imageSize = vulkanImage.width * vulkanImage.height * getFormatBytesPerPixel(vulkanImage.format);
    if (size != imageSize)
    {
        return VK_ERROR_UNKNOWN;
    }

    VK_CHECK_RETURN(waitForSharedImageWork(vulkanImage));

    bool sharedWithCompute = false;
#ifdef SAF_VULKAN_COMPUTE
    sharedWithCompute = vulkanImage.sharing == ImageSharing::Compute;
#endif
    // Compute-shared images are copied in GENERAL, which they never leave
    const VkImageLayout copyLayout = sharedWithCompute ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    return downloadThroughStagingBuffer(data, size, [&](VkCommandBuffer commandBuffer, VkBuffer stagingBuffer)
                                        {
                                        if (sharedWithCompute)
                                        {
                                            transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, commandBuffer);
                                        }
                                        else
                                        {
                                            transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, commandBuffer);
                                        }
                                        // Zero row length and image height: the buffer is tightly packed
                                        const VkBufferImageCopy bufferImageCopy{
                                            .bufferOffset      = 0,
                                            .bufferRowLength   = 0,
                                            .bufferImageHeight = 0,
                                            .imageSubresource  = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
                                            .imageOffset       = { 0, 0, 0 },
                                            .imageExtent       = { vulkanImage.width, vulkanImage.height, 1 }
                                        };

                                        vkCmdCopyImageToBuffer(commandBuffer, vulkanImage.image, copyLayout, stagingBuffer, 1, &bufferImageCopy);

                                        if (sharedWithCompute)
                                        {
                                            transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, commandBuffer);
                                        }
                                        else
                                        {
                                            transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, commandBuffer);
                                        } });
}

#ifdef SAF_CUDA_INTEROP
cudaSurfaceObject_t Application::getCudaSurface(ImageHandle image) const
{
    const VulkanImage* vulkanImage = mImages.find(image);
    if (vulkanImage == nullptr)
    {
        reportMisuse("Image " + std::to_string(image.id) + " does not exist.");
    }
    if (vulkanImage->sharing != ImageSharing::Cuda)
    {
        reportMisuse("Image " + std::to_string(image.id) + " was not created with ImageSharing::Cuda.");
    }
    return vulkanImage->cudaSurfaceObject;
}

cudaTextureObject_t Application::getCudaTexture(ImageHandle image) const
{
    const VulkanImage* vulkanImage = mImages.find(image);
    if (vulkanImage == nullptr)
    {
        reportMisuse("Image " + std::to_string(image.id) + " does not exist.");
    }
    if (vulkanImage->sharing != ImageSharing::Cuda)
    {
        reportMisuse("Image " + std::to_string(image.id) + " was not created with ImageSharing::Cuda.");
    }
    return vulkanImage->cudaTextureObject;
}
#endif

#ifdef SAF_VULKAN_COMPUTE
VkResult Application::createBuffer(const VulkanBufferCreateInfo& createInfo, BufferHandle& outBuffer)
{
    if (createInfo.size == 0)
    {
        return VK_ERROR_UNKNOWN;
    }

    VulkanBuffer buffer;
    buffer.size = createInfo.size;

    // CONCURRENT needs two distinct families; without async compute the buffer stays EXCLUSIVE to graphics
    const U32 computeQueueFamilies[] = { mQueueGCT.familyIndex, mQueueC.familyIndex };
    const VkBufferCreateInfo bufferCreateInfo{
        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size                  = createInfo.size,
        .usage                 = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode           = hasAsyncCompute() ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = hasAsyncCompute() ? 2u : 0u,
        .pQueueFamilyIndices   = computeQueueFamilies
    };

    const VkResult result = allocateBuffer(bufferCreateInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer.buffer, buffer.deviceMemory);
    if (result < 0)
    {
        destroyBufferResources(buffer);
        return result;
    }

    outBuffer = mBuffers.add(buffer);

    return VK_SUCCESS;
}

void Application::destroyBuffer(BufferHandle buffer)
{
    const VulkanBuffer* vulkanBuffer = mBuffers.find(buffer);
    if (vulkanBuffer == nullptr)
    {
        return;
    }
    // A dispatch recorded this frame may use the buffer, and it is only submitted after onUpdate
    if (mComputeRecording)
    {
        reportMisuse("destroyBuffer() after getComputeCommandBuffer() in the same onUpdate.");
    }
    // The buffer may still be used by work in flight.
    VK_CHECK(vkDeviceWaitIdle(mLogicalDevice));
    destroyBufferResources(*vulkanBuffer);
    mBuffers.erase(buffer);
}

void Application::destroyBufferResources(const VulkanBuffer& vulkanBuffer)
{
    vkDestroyBuffer(mLogicalDevice, vulkanBuffer.buffer, nullptr);
    vkFreeMemory(mLogicalDevice, vulkanBuffer.deviceMemory, nullptr);
}

VkResult Application::uploadBuffer(BufferHandle buffer, const void* data, size_t size)
{
    const VulkanBuffer* found = mBuffers.find(buffer);
    if (found == nullptr || size != found->size)
    {
        return VK_ERROR_UNKNOWN;
    }
    const VulkanBuffer& vulkanBuffer = *found;

    VK_CHECK_RETURN(waitForTimeline(mComputeTimeline, mComputeTimelineValue));

    return uploadThroughStagingBuffer(data, size, [&](VkCommandBuffer commandBuffer, VkBuffer stagingBuffer)
                                      {
                                        const VkBufferMemoryBarrier beforeCopy{
                                            .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                            .srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
                                            .dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .buffer              = vulkanBuffer.buffer,
                                            .offset              = 0,
                                            .size                = VK_WHOLE_SIZE
                                        };
                                        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &beforeCopy, 0, nullptr);

                                        const VkBufferCopy bufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
                                        vkCmdCopyBuffer(commandBuffer, stagingBuffer, vulkanBuffer.buffer, 1, &bufferCopy);

                                        const VkBufferMemoryBarrier afterCopy{
                                            .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                            .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
                                            .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .buffer              = vulkanBuffer.buffer,
                                            .offset              = 0,
                                            .size                = VK_WHOLE_SIZE
                                        };
                                        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &afterCopy, 0, nullptr); });
}

VkResult Application::downloadBuffer(BufferHandle buffer, void* data, size_t size)
{
    const VulkanBuffer* found = mBuffers.find(buffer);
    if (found == nullptr || size != found->size)
    {
        return VK_ERROR_UNKNOWN;
    }
    const VulkanBuffer& vulkanBuffer = *found;

    VK_CHECK_RETURN(waitForTimeline(mComputeTimeline, mComputeTimelineValue));

    return downloadThroughStagingBuffer(data, size, [&](VkCommandBuffer commandBuffer, VkBuffer stagingBuffer)
                                        {
                                        const VkBufferMemoryBarrier beforeCopy{
                                            .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                            .srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
                                            .dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .buffer              = vulkanBuffer.buffer,
                                            .offset              = 0,
                                            .size                = VK_WHOLE_SIZE
                                        };
                                        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &beforeCopy, 0, nullptr);

                                        const VkBufferCopy bufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
                                        vkCmdCopyBuffer(commandBuffer, vulkanBuffer.buffer, stagingBuffer, 1, &bufferCopy);

                                        const VkBufferMemoryBarrier afterCopy{
                                            .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                            .srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT,
                                            .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .buffer              = vulkanBuffer.buffer,
                                            .offset              = 0,
                                            .size                = VK_WHOLE_SIZE
                                        };
                                        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &afterCopy, 0, nullptr); });
}

VkCommandBuffer Application::getComputeCommandBuffer()
{
    if (mLayerPhase != LayerPhase::Update)
    {
        reportMisuse("getComputeCommandBuffer() called outside of Layer::onUpdate.");
    }

    // if alreadt recording current frame, just return, else reset, init and begin cb and relevant other things (such as timings, pools, etc.)
    const ComputeSlot& slot = mComputeSlots[mFrameSlotIndex];
    if (mComputeRecording)
    {
        return slot.commandBuffer;
    }

    VK_CHECK(waitForTimeline(mComputeTimeline, slot.timelineValue));
    readComputeTimings(mFrameSlotIndex);
    VK_CHECK(vkResetDescriptorPool(mLogicalDevice, slot.descriptorPool, 0));
    VK_CHECK(vkResetCommandBuffer(slot.commandBuffer, 0));

    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_CHECK(vkBeginCommandBuffer(slot.commandBuffer, &beginInfo));
    vkCmdResetQueryPool(slot.commandBuffer, slot.queryPool, 0, 2 * mMaxTimingPairs);

    mComputeRecording = true;
    return slot.commandBuffer;
}

VkResult Application::createComputePipeline(const ComputePipelineCreateInfo& createInfo, ComputePipelineHandle& outPipeline)
{
    VulkanComputePipeline pipeline;

    const VkResult result = createComputePipelineResources(pipeline, createInfo);
    if (result < 0)
    {
        destroyComputePipelineResources(pipeline);
        return result;
    }

    outPipeline = mComputePipelines.add(pipeline);

    return VK_SUCCESS;
}

VkResult Application::createComputePipelineResources(VulkanComputePipeline& pipeline, const ComputePipelineCreateInfo& createInfo)
{
    SpvReflectShaderModule module;
    if (spvReflectCreateShaderModule(createInfo.spirv.size_bytes(), createInfo.spirv.data(), &module) != SPV_REFLECT_RESULT_SUCCESS)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const VkResult reflectResult = reflectComputeShader(module, createInfo.entryPoint, pipeline.bindings, pipeline.pushConstantSize);

    // The device features that SPIR-V capabilities need; SAF enables only shaderInt16, shaderInt64, and shaderFloat64 for now, when the device supports them
    const VkPhysicalDeviceFeatures& enabledFeatures = mVulkanContext->getEnabledFeatures();
    const struct
    {
        SpvCapability capability;
        VkBool32 enabled;
        const char* feature;
    } capabilityFeatures[] = {
        { SpvCapabilityInt64, enabledFeatures.shaderInt64, "shaderInt64" },
        { SpvCapabilityInt16, enabledFeatures.shaderInt16, "shaderInt16" },
        { SpvCapabilityFloat64, enabledFeatures.shaderFloat64, "shaderFloat64" },
    };
    const char* missingFeature = nullptr;
    for (U32 i = 0; i < module.capability_count; ++i)
    {
        for (const auto& entry : capabilityFeatures)
        {
            if (module.capabilities[i].value == entry.capability && !entry.enabled)
            {
                missingFeature = entry.feature;
            }
        }
    }
    spvReflectDestroyShaderModule(&module);
    VK_CHECK_RETURN(reflectResult);
    if (missingFeature != nullptr)
    {
        std::cerr << "[vulkan] Error: The shader needs the device feature " << missingFeature << ", which is not enabled." << '\n';
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // One layout for each set from 0 to the highest used; unused sets in between get empty layouts
    U32 setCount = 0;
    for (const auto& binding : pipeline.bindings)
    {
        setCount = std::max(setCount, binding.set + 1);
    }
    std::vector<std::vector<VkDescriptorSetLayoutBinding>> setLayoutBindings(setCount);
    for (const auto& binding : pipeline.bindings)
    {
        setLayoutBindings[binding.set].push_back({ .binding = binding.binding, .descriptorType = binding.type, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr });
    }

    pipeline.setLayouts.resize(setCount, VK_NULL_HANDLE);
    for (U32 set = 0; set < setCount; ++set)
    {
        const VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo{
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<U32>(setLayoutBindings[set].size()),
            .pBindings    = setLayoutBindings[set].data()
        };
        VK_CHECK_RETURN(vkCreateDescriptorSetLayout(mLogicalDevice, &setLayoutCreateInfo, nullptr, &pipeline.setLayouts[set]));
    }

    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = pipeline.pushConstantSize
    };
    const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = setCount,
        .pSetLayouts            = pipeline.setLayouts.data(),
        .pushConstantRangeCount = pipeline.pushConstantSize > 0 ? 1u : 0u,
        .pPushConstantRanges    = &pushConstantRange
    };
    VK_CHECK_RETURN(vkCreatePipelineLayout(mLogicalDevice, &pipelineLayoutCreateInfo, nullptr, &pipeline.pipelineLayout));

    const VkShaderModuleCreateInfo shaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = createInfo.spirv.size_bytes(),
        .pCode    = createInfo.spirv.data()
    };
    VkShaderModule shaderModule;
    VK_CHECK_RETURN(vkCreateShaderModule(mLogicalDevice, &shaderModuleCreateInfo, nullptr, &shaderModule));

    const VkComputePipelineCreateInfo pipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
            .module              = shaderModule,
            .pName               = createInfo.entryPoint,
            .pSpecializationInfo = createInfo.specialization },
        .layout = pipeline.pipelineLayout
    };
    const VkResult result = vkCreateComputePipelines(mLogicalDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline.pipeline);
    vkDestroyShaderModule(mLogicalDevice, shaderModule, nullptr);
    VK_CHECK_RETURN(result);

    return VK_SUCCESS;
}

void Application::destroyComputePipeline(ComputePipelineHandle pipeline)
{
    const VulkanComputePipeline* computePipeline = mComputePipelines.find(pipeline);
    if (computePipeline == nullptr)
    {
        return;
    }
    // A dispatch recorded this frame may use the pipeline, and it is only submitted after onUpdate
    if (mComputeRecording)
    {
        reportMisuse("destroyComputePipeline() after getComputeCommandBuffer() in the same onUpdate.");
    }
    // The pipeline may still be used by work in flight.
    VK_CHECK(vkDeviceWaitIdle(mLogicalDevice));
    destroyComputePipelineResources(*computePipeline);
    mComputePipelines.erase(pipeline);
}

void Application::destroyComputePipelineResources(const VulkanComputePipeline& pipeline)
{
    vkDestroyPipeline(mLogicalDevice, pipeline.pipeline, nullptr);
    vkDestroyPipelineLayout(mLogicalDevice, pipeline.pipelineLayout, nullptr);
    for (const VkDescriptorSetLayout setLayout : pipeline.setLayouts)
    {
        vkDestroyDescriptorSetLayout(mLogicalDevice, setLayout, nullptr);
    }
}

VulkanComputePipeline& Application::findComputePipeline(ComputePipelineHandle pipeline)
{
    VulkanComputePipeline* computePipeline = mComputePipelines.find(pipeline);
    if (computePipeline == nullptr)
    {
        reportMisuse("Compute pipeline " + std::to_string(pipeline.id) + " does not exist.");
    }
    return *computePipeline;
}

ComputeBinding& Application::findComputeBinding(VulkanComputePipeline& pipeline, const BindingKey& key)
{
    if (key.name == nullptr)
    {
        // check for set and binding
        for (auto& computeBinding : pipeline.bindings)
        {
            if (computeBinding.set == key.set && computeBinding.binding == key.binding)
            {
                return computeBinding;
            }
        }
        reportMisuse("The shader has no binding at set " + std::to_string(key.set) + ", binding " + std::to_string(key.binding) + ".");
    }

    const char* name      = key.name;
    ComputeBinding* found = nullptr;
    bool unnamed          = false;
    // check by name (TODO: might not work all the time?)
    for (auto& binding : pipeline.bindings)
    {
        // A binding without a debug name never matches, not even the empty name
        unnamed |= binding.name.empty();
        if (!binding.name.empty() && binding.name == name)
        {
            if (found != nullptr)
            {
                reportMisuse("Binding name '" + Str(name) + "' is ambiguous across sets; bind by (set, binding).");
            }
            found = &binding;
        }
    }
    if (found == nullptr)
    {
        reportMisuse("The shader has no binding named '" + Str(name) + "'." + (unnamed ? " Some bindings have no debug name (glslc -O strips them); bind those by (set, binding)." : ""));
    }
    return *found;
}

const VulkanImage& Application::findSharedWithComputeImage(ImageHandle image) const
{
    const VulkanImage* vulkanImage = mImages.find(image);
    if (vulkanImage == nullptr)
    {
        reportMisuse("Image " + std::to_string(image.id) + " does not exist.");
    }
    if (vulkanImage->sharing != ImageSharing::Compute)
    {
        reportMisuse("Image " + std::to_string(image.id) + " was not created with ImageSharing::Compute.");
    }
    return *vulkanImage;
}

const VulkanBuffer& Application::findComputeBuffer(BufferHandle buffer) const
{
    const VulkanBuffer* vulkanBuffer = mBuffers.find(buffer);
    if (vulkanBuffer == nullptr)
    {
        reportMisuse("Buffer " + std::to_string(buffer.id) + " does not exist.");
    }
    return *vulkanBuffer;
}

void Application::bindImage(ComputeBinding& binding, ImageHandle image)
{
    if (binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
    {
        reportMisuse("Binding " + describeBinding(binding) + " is a storage buffer; an image was bound.");
    }
    // check if available - even though we do not care about the returned ressource at all
    findSharedWithComputeImage(image);
    binding.image  = image;
    binding.buffer = {};
}

void Application::bindBuffer(ComputeBinding& binding, BufferHandle buffer)
{
    if (binding.type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
    {
        reportMisuse("Binding " + describeBinding(binding) + " is an image; a buffer was bound.");
    }
    // check if available - even though we do not care about the returned ressource at all
    findComputeBuffer(buffer);
    binding.image  = {};
    binding.buffer = buffer;
}

void Application::bind(ComputePipelineHandle pipeline, const BindingKey& key, ImageHandle image)
{
    bindImage(findComputeBinding(findComputePipeline(pipeline), key), image);
}

void Application::bind(ComputePipelineHandle pipeline, const BindingKey& key, BufferHandle buffer)
{
    bindBuffer(findComputeBinding(findComputePipeline(pipeline), key), buffer);
}

void Application::setPushConstantData(ComputePipelineHandle pipeline, const void* data, U32 size)
{
    VulkanComputePipeline& computePipeline = findComputePipeline(pipeline);
    if (size != computePipeline.pushConstantSize)
    {
        reportMisuse("setPushConstants() got " + std::to_string(size) + " bytes; the shader's push constant block has " + std::to_string(computePipeline.pushConstantSize) + ".");
    }
    const auto* bytes = static_cast<const Byte*>(data);
    // copy so that this can be called with a temporary :D
    computePipeline.pushConstants.assign(bytes, bytes + size);
}

void Application::dispatch(ComputePipelineHandle pipeline, U32 groupsX, U32 groupsY, U32 groupsZ)
{
    // Begins the current slot's recording first: its pool is only safe to allocate from while the slot records
    const VkCommandBuffer commandBuffer = getComputeCommandBuffer();
    const ComputeSlot& slot             = mComputeSlots[mFrameSlotIndex];

    if (groupsX > mMaxComputeWorkGroupCount[0] || groupsY > mMaxComputeWorkGroupCount[1] || groupsZ > mMaxComputeWorkGroupCount[2])
    {
        reportMisuse("dispatch() group count exceeds maxComputeWorkGroupCount (" + std::to_string(mMaxComputeWorkGroupCount[0]) + ", " + std::to_string(mMaxComputeWorkGroupCount[1]) + ", " + std::to_string(mMaxComputeWorkGroupCount[2]) + ").");
    }

    const VulkanComputePipeline& computePipeline = findComputePipeline(pipeline);
    if (computePipeline.pushConstantSize > 0 && computePipeline.pushConstants.empty())
    {
        reportMisuse("dispatch() before setPushConstants() on a pipeline with a push constant block.");
    }

    std::vector<VkDescriptorSet> sets(computePipeline.setLayouts.size(), VK_NULL_HANDLE);
    if (!sets.empty())
    {
        // allocation cheap since we have per-frame pools
        const VkDescriptorSetAllocateInfo setAllocateInfo{
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = slot.descriptorPool,
            .descriptorSetCount = static_cast<U32>(sets.size()),
            .pSetLayouts        = computePipeline.setLayouts.data()
        };
        const VkResult result = vkAllocateDescriptorSets(mLogicalDevice, &setAllocateInfo, sets.data());
        if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL)
        {
            reportMisuse("The compute descriptor pool of this frame is exhausted; raise ApplicationSettings::maxComputeDescriptorSets.");
        }
        VK_CHECK(result);
    }

    // Reserved up front, so the pointers stored in the writes stay valid
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkWriteDescriptorSet> writes;
    imageInfos.reserve(computePipeline.bindings.size());
    bufferInfos.reserve(computePipeline.bindings.size());
    writes.reserve(computePipeline.bindings.size());

    for (const auto& binding : computePipeline.bindings)
    {
        VkWriteDescriptorSet write{
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = sets[binding.set],
            .dstBinding      = binding.binding,
            .descriptorCount = 1,
            .descriptorType  = binding.type
        };

        // atm we only have storage buffers and images - if we extend that further sometime this might get more complex.
        if (binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        {
            if (binding.buffer.id == 0)
            {
                reportMisuse("Binding " + describeBinding(binding) + " is not bound.");
            }
            const VulkanBuffer& buffer = findComputeBuffer(binding.buffer);
            bufferInfos.push_back({ .buffer = buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE });
            write.pBufferInfo = &bufferInfos.back();
        }
        else
        {
            if (binding.image.id == 0)
            {
                reportMisuse("Binding " + describeBinding(binding) + " is not bound.");
            }
            const VulkanImage& image = findSharedWithComputeImage(binding.image);
            imageInfos.push_back({ .sampler = image.sampler, .imageView = image.imageView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL });
            write.pImageInfo = &imageInfos.back();
        }

        writes.push_back(write);
    }

    vkUpdateDescriptorSets(mLogicalDevice, static_cast<U32>(writes.size()), writes.data(), 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline.pipeline);
    if (!sets.empty())
    {
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline.pipelineLayout, 0, static_cast<U32>(sets.size()), sets.data(), 0, nullptr);
    }
    if (!computePipeline.pushConstants.empty())
    {
        vkCmdPushConstants(commandBuffer, computePipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, computePipeline.pushConstantSize, computePipeline.pushConstants.data());
    }
    vkCmdDispatch(commandBuffer, groupsX, groupsY, groupsZ);
}

void Application::computeBarrier()
{
    const VkMemoryBarrier barrier{
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    };
    vkCmdPipelineBarrier(getComputeCommandBuffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Application::beginTiming(const char* name)
{
    const VkCommandBuffer commandBuffer = getComputeCommandBuffer();
    ComputeSlot& slot                   = mComputeSlots[mFrameSlotIndex];

    const U32 pairIndex = beginTimingPair(slot.timingPairs, name, TimingApi::Vulkan);
    if (mTimestampMask != 0)
    {
        vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, slot.queryPool, 2 * pairIndex); // events are stored [start0, end0, start1, end1, ....]
    }
}

void Application::endTiming(const char* name)
{
    const VkCommandBuffer commandBuffer = getComputeCommandBuffer();
    ComputeSlot& slot                   = mComputeSlots[mFrameSlotIndex];

    const U32 pairIndex = endTimingPair(slot.timingPairs, name);
    if (mTimestampMask != 0)
    {
        vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, slot.queryPool, 2 * pairIndex + 1); // events are stored [start0, end0, start1, end1, ....]
    }
}

void Application::readComputeTimings(U32 slotIndex)
{
    ComputeSlot& slot = mComputeSlots[slotIndex];
    if (slot.timingPairs.empty() || mTimestampMask == 0)
    {
        slot.timingPairs.clear();
        return;
    }

    // Only called once the slot's submission has finished, so the waiting (VK_QUERY_RESULT_WAIT_BIT) should return at once
    std::vector<U64> timestamps(2 * slot.timingPairs.size());
    VK_CHECK(vkGetQueryPoolResults(mLogicalDevice, slot.queryPool, 0, static_cast<U32>(timestamps.size()), timestamps.size() * sizeof(U64), timestamps.data(), sizeof(U64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

    std::vector<F32> milliseconds(slot.timingPairs.size());
    for (size_t i = 0; i < milliseconds.size(); ++i)
    {
        // Masked, so a counter wrapping between begin and end still gives the right difference (weird Vulkan Timing API stuff with the mask)
        const U64 ticks = (timestamps[2 * i + 1] - timestamps[2 * i]) & mTimestampMask;
        milliseconds[i] = static_cast<F32>(static_cast<F64>(ticks) * mTimestampPeriod * 1e-6); // VkPhysicalDeviceLimits::timestampPeriod is in nanoseconds per tick.
    }
    storeTimings(slot.timingPairs, milliseconds);
}
#endif

#ifdef SAF_CUDA_INTEROP
void Application::waitForVulkanCompletion(cudaExternalSemaphore_t semaphore, U64 value, cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams{};
    externalSemaphoreWaitParams.params.fence.value = value;

    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&semaphore, &externalSemaphoreWaitParams, 1, stream));
}

void Application::signalCudaCompletion(cudaExternalSemaphore_t semaphore, U64 value, cudaStream_t stream)
{
    cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams{};
    externalSemaphoreSignalParams.params.fence.value = value;

    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&semaphore, &externalSemaphoreSignalParams, 1, stream));
}

bool Application::hasCudaSharedImage() const
{
    return std::any_of(mImages.begin(), mImages.end(), [](const auto& image)
                       { return image.second.sharing == ImageSharing::Cuda; });
}

void Application::beginTiming(cudaStream_t stream, const char* name)
{
    if (mLayerPhase != LayerPhase::Update)
    {
        reportMisuse("beginTiming() on a CUDA stream called outside of Layer::onUpdate.");
    }

    CudaTimingSlot& slot = mCudaTimingSlots[mFrameSlotIndex];
    const U32 pairIndex  = beginTimingPair(slot.timingPairs, name, TimingApi::Cuda);
    CUDA_CHECK(cudaEventRecord(slot.events[2 * pairIndex], stream)); // events are stored [start0, end0, start1, end1, ....]
}

void Application::endTiming(cudaStream_t stream, const char* name)
{
    if (mLayerPhase != LayerPhase::Update)
    {
        reportMisuse("endTiming() on a CUDA stream called outside of Layer::onUpdate.");
    }

    CudaTimingSlot& slot = mCudaTimingSlots[mFrameSlotIndex];
    const U32 pairIndex  = endTimingPair(slot.timingPairs, name);
    CUDA_CHECK(cudaEventRecord(slot.events[2 * pairIndex + 1], stream)); // events are stored [start0, end0, start1, end1, ....]
}

void Application::readCudaTimings(U32 slotIndex, bool wait)
{
    CudaTimingSlot& slot = mCudaTimingSlots[slotIndex];
    for (size_t i = 0; i < 2 * slot.timingPairs.size(); ++i)
    {
        if (wait)
        {
            CUDA_CHECK(cudaEventSynchronize(slot.events[i]));
            continue;
        }

        const cudaError_t result = cudaEventQuery(slot.events[i]);
        if (result == cudaErrorNotReady)
        {
            return;
        }
        CUDA_CHECK(result);
    }

    std::vector<F32> milliseconds(slot.timingPairs.size());
    for (size_t i = 0; i < milliseconds.size(); ++i)
    {
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds[i], slot.events[2 * i], slot.events[2 * i + 1])); // events are stored [start0, end0, start1, end1, ....]
    }
    storeTimings(slot.timingPairs, milliseconds);
}
#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
U32 Application::beginTimingPair(std::vector<TimingPair>& pairs, const char* name, TimingApi api)
{
    U32 scope = 0;
    while (scope < mTimingScopes.size() && mTimingScopes[scope].name != name)
    {
        ++scope;
    }
    if (scope == mTimingScopes.size())
    {
        mTimingScopes.push_back({ .name = name, .api = api, .history = {} });
    }
    else if (mTimingScopes[scope].api != api)
    {
        reportMisuse("Timing scope '" + Str(name) + "' is used from both Vulkan and CUDA.");
    }

    for (const TimingPair& pair : pairs)
    {
        if (pair.scope == scope && pair.open)
        {
            reportMisuse("beginTiming() on timing scope '" + Str(name) + "', which is already open.");
        }
    }

    if (pairs.size() == mMaxTimingPairs)
    {
        reportMisuse("More than " + std::to_string(mMaxTimingPairs) + " timing pairs begun in one frame; raise ApplicationSettings::maxTimingPairs.");
    }

    pairs.push_back({ .scope = scope, .open = true });
    return static_cast<U32>(pairs.size() - 1);
}

U32 Application::endTimingPair(std::vector<TimingPair>& pairs, const char* name)
{
    for (U32 i = 0; i < pairs.size(); ++i)
    {
        if (pairs[i].open && mTimingScopes[pairs[i].scope].name == name)
        {
            pairs[i].open = false;
            return i;
        }
    }
    reportMisuse("endTiming() on timing scope '" + Str(name) + "' without a matching beginTiming().");
}

void Application::checkTimingPairsClosed(const std::vector<TimingPair>& pairs)
{
    for (const TimingPair& pair : pairs)
    {
        if (pair.open)
        {
            reportMisuse("Timing scope '" + mTimingScopes[pair.scope].name + "' is still open at the end of onUpdate.");
        }
    }
}

void Application::storeTimings(std::vector<TimingPair>& pairs, std::span<const F32> milliseconds)
{
    // One sum per scope, however many pairs it has
    std::vector<std::pair<U32, F32>> sums;
    for (size_t i = 0; i < pairs.size(); ++i)
    {
        const auto it = std::ranges::find(sums, pairs[i].scope, &std::pair<U32, F32>::first);
        if (it == sums.end())
        {
            sums.push_back({ pairs[i].scope, milliseconds[i] });
        }
        else
        {
            it->second += milliseconds[i];
        }
    }
    for (const auto& [scope, sum] : sums)
    {
        std::vector<F32>& history = mTimingScopes[scope].history;
        if (history.size() == timingHistoryLength)
        {
            history.erase(history.begin());
        }
        history.push_back(sum);
    }
    pairs.clear();
}

void Application::collectTimings()
{
    const U32 slotCount = mVulkanSwapchain->getFramesInFlight();

#ifdef SAF_VULKAN_COMPUTE
    // Oldest submission first, so a newer frame's result is stored last. A slot whose frame skipped the getter keeps
    // older pairs than its position suggests; timeline values grow with every submit, so they order exactly.
    U64 finishedValue = 0;
    VK_CHECK(vkGetSemaphoreCounterValue(mLogicalDevice, mComputeTimeline, &finishedValue));
    std::vector<U32> finishedSlots;
    for (U32 slotIndex = 0; slotIndex < slotCount; ++slotIndex)
    {
        const ComputeSlot& slot = mComputeSlots[slotIndex];
        if (!slot.timingPairs.empty() && slot.timelineValue <= finishedValue)
        {
            finishedSlots.push_back(slotIndex);
        }
    }
    std::sort(finishedSlots.begin(), finishedSlots.end(), [this](U32 a, U32 b)
              { return mComputeSlots[a].timelineValue < mComputeSlots[b].timelineValue; });
    for (const U32 slotIndex : finishedSlots)
    {
        readComputeTimings(slotIndex);
    }
#endif

#ifdef SAF_CUDA_INTEROP
    // Oldest slot first, the current one, so a newer frame's result is stored last
    for (U32 k = 0; k < slotCount; ++k)
    {
        // This frame's onUpdate reuses the current slot, so its events are waited for
        readCudaTimings((mFrameSlotIndex + k) % slotCount, k == 0);
    }
#endif
}
#endif

void Application::pushLayer(std::unique_ptr<Layer>&& layer)
{
    if (!mInitialized)
        return;

    if (mLayerPhase != LayerPhase::None)
    {
        mPendingLayerOps.push_back(std::move(layer));
        return;
    }

    VK_CHECK(vkQueueWaitIdle(mQueueGCT.queue));
    VK_CHECK(vkDeviceWaitIdle(mLogicalDevice));
    layer->onAttach(this);
    mLayerStack.emplace_back(std::move(layer));
}

U32 Application::findMemoryType(U32 memoryTypeBits, VkMemoryPropertyFlags propertyFlags)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memoryProperties);

    for (U32 i = 0; i < memoryProperties.memoryTypeCount; i++)
    {
        if ((memoryTypeBits & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags)
        {
            return i;
        }
    }

    VK_CHECK(VK_ERROR_UNKNOWN);
}

VkResult Application::allocateBuffer(const VkBufferCreateInfo& bufferCreateInfo, VkMemoryPropertyFlags propertyFlags, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VK_CHECK_RETURN(vkCreateBuffer(mLogicalDevice, &bufferCreateInfo, nullptr, &buffer));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(mLogicalDevice, buffer, &memoryRequirements);

    const VkMemoryAllocateInfo memoryAllocateInfo{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = memoryRequirements.size,
        .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, propertyFlags)
    };

    VK_CHECK_RETURN(vkAllocateMemory(mLogicalDevice, &memoryAllocateInfo, nullptr, &bufferMemory));
    VK_CHECK_RETURN(vkBindBufferMemory(mLogicalDevice, buffer, bufferMemory, 0));

    return VK_SUCCESS;
}

VkResult Application::uploadThroughStagingBuffer(const void* data, size_t size, const std::function<void(VkCommandBuffer, VkBuffer)>& recordCopy)
{
    VkBuffer stagingBuffer             = VK_NULL_HANDLE;
    VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;

    // In a lambda, so the staging buffer is freed below on every early return
    const VkResult result = [&]() -> VkResult
    {
        const VkBufferCreateInfo bufferCreateInfo{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = size,
            .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        VK_CHECK_RETURN(allocateBuffer(bufferCreateInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory));

        // Coherent memory: the write is visible to the copy at submit, so the mapping can end here
        void* mappedData;
        VK_CHECK_RETURN(vkMapMemory(mLogicalDevice, stagingBufferMemory, 0, size, 0, &mappedData));
        std::memcpy(mappedData, data, size);
        vkUnmapMemory(mLogicalDevice, stagingBufferMemory);

        return executeSingleTimeCommandBuffer([&](VkCommandBuffer commandBuffer)
                                              {
            recordCopy(commandBuffer, stagingBuffer);
            return VK_SUCCESS; });
    }();

    vkDestroyBuffer(mLogicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(mLogicalDevice, stagingBufferMemory, nullptr);
    return result;
}

VkResult Application::downloadThroughStagingBuffer(void* data, size_t size, const std::function<void(VkCommandBuffer, VkBuffer)>& recordCopy)
{
    VkBuffer stagingBuffer             = VK_NULL_HANDLE;
    VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;

    // In a lambda, so the staging buffer is freed below on every early return
    const VkResult result = [&]() -> VkResult
    {
        const VkBufferCreateInfo bufferCreateInfo{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = size,
            .usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        VK_CHECK_RETURN(allocateBuffer(bufferCreateInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory));

        void* mappedData;
        VK_CHECK_RETURN(vkMapMemory(mLogicalDevice, stagingBufferMemory, 0, size, 0, &mappedData));

        const VkResult copyResult = executeSingleTimeCommandBuffer([&](VkCommandBuffer commandBuffer)
                                                                   {
            recordCopy(commandBuffer, stagingBuffer);
            const VkBufferMemoryBarrier stagingToHost{
                .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask       = VK_ACCESS_HOST_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer              = stagingBuffer,
                .offset              = 0,
                .size                = VK_WHOLE_SIZE
            };
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &stagingToHost, 0, nullptr);
            return VK_SUCCESS; });

        if (copyResult >= 0)
        {
            std::memcpy(data, mappedData, size);
        }
        vkUnmapMemory(mLogicalDevice, stagingBufferMemory);
        return copyResult;
    }();

    vkDestroyBuffer(mLogicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(mLogicalDevice, stagingBufferMemory, nullptr);
    return result;
}

VkResult Application::waitForSharedImageWork(const VulkanImage& image)
{
#ifdef SAF_VULKAN_COMPUTE
    if (image.sharing == ImageSharing::Compute)
    {
        return waitForTimeline(mComputeTimeline, mComputeTimelineValue);
    }
#endif
#ifdef SAF_CUDA_INTEROP
    if (image.sharing == ImageSharing::Cuda)
    {
        return waitForTimeline(mCudaTimeline, mCudaTimelineValue);
    }
#endif
    (void)image;
    return VK_SUCCESS;
}

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
VkResult Application::waitForTimeline(VkSemaphore timeline, U64 value)
{
    const VkSemaphoreWaitInfo waitInfo{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores    = &timeline,
        .pValues        = &value
    };
    return vkWaitSemaphores(mLogicalDevice, &waitInfo, std::numeric_limits<U64>::max());
}
#endif

U32 Application::getFormatBytesPerPixel(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_UNORM:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    default:
        throw std::runtime_error("Unsupported format");
    }
}

VkResult Application::transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkCommandBuffer commandBuffer)
{
    const VkImageMemoryBarrier barrier{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask       = srcAccessMask,
        .dstAccessMask       = dstAccessMask,
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {
               .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
               .baseMipLevel   = 0,
               .levelCount     = 1,
               .baseArrayLayer = 0,
               .layerCount     = 1 }
    };

    if (commandBuffer == VK_NULL_HANDLE)
    {
        // execute single time command buffer
        VK_CHECK_RETURN(executeSingleTimeCommandBuffer([&](VkCommandBuffer cmdBuffer)
                                                       {
                                                           vkCmdPipelineBarrier(cmdBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                                                           return VK_SUCCESS; }));
    }
    else
    {
        vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    return VK_SUCCESS;
}

#ifdef SAF_CUDA_INTEROP
void* Application::getSemaphoreHandle(VkSemaphore semaphore)
{
#ifdef WIN32
    const VkSemaphoreGetWin32HandleInfoKHR getWin32HandleInfo{
        .sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
        .semaphore  = semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
    };
    HANDLE handle;
    VK_CHECK(vkGetSemaphoreWin32HandleKHR(mLogicalDevice, &getWin32HandleInfo, &handle));
    return reinterpret_cast<void*>(handle);
#else
    const VkSemaphoreGetFdInfoKHR getFdInfo{
        .sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
        .semaphore  = semaphore,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    I32 fd;
    VK_CHECK(vkGetSemaphoreFdKHR(mLogicalDevice, &getFdInfo, &fd));
    return reinterpret_cast<void*>(static_cast<uintptr_t>(fd));
#endif
}

void* Application::getMemoryHandle(VkDeviceMemory deviceMemory)
{
#ifdef WIN32
    const VkMemoryGetWin32HandleInfoKHR getWin32HandleInfo{
        .sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .memory     = deviceMemory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
    };
    HANDLE handle;
    VK_CHECK(vkGetMemoryWin32HandleKHR(mLogicalDevice, &getWin32HandleInfo, &handle));
    return reinterpret_cast<void*>(handle);
#else
    const VkMemoryGetFdInfoKHR getFdInfo{
        .sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory     = deviceMemory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
    };
    I32 fd;
    VK_CHECK(vkGetMemoryFdKHR(mLogicalDevice, &getFdInfo, &fd));
    return reinterpret_cast<void*>(static_cast<uintptr_t>(fd));
#endif
}

cudaChannelFormatDesc Application::getFormatDescriptor(VkFormat format)
{
    constexpr auto cudaCreateChannelDesc = []<typename T>() -> cudaChannelFormatDesc
    {
        cudaChannelFormatDesc desc{};
        if constexpr (std::is_same_v<T, UVec4>)
        {
            desc.x = 8;
            desc.y = 8;
            desc.z = 8;
            desc.w = 8;
            desc.f = cudaChannelFormatKindUnsigned;
        }
        else if constexpr (std::is_same_v<T, HVec4>)
        {
            desc.x = 16;
            desc.y = 16;
            desc.z = 16;
            desc.w = 16;
            desc.f = cudaChannelFormatKindFloat;
        }
        else if constexpr (std::is_same_v<T, Vec4>)
        {
            desc.x = 32;
            desc.y = 32;
            desc.z = 32;
            desc.w = 32;
            desc.f = cudaChannelFormatKindFloat;
        }

        return desc;
    };

    switch (format)
    {
    case VK_FORMAT_R8G8B8A8_UNORM:
        return cudaCreateChannelDesc.template operator()<UVec4>();
    case VK_FORMAT_B8G8R8A8_UNORM:
        return cudaCreateChannelDesc.template operator()<UVec4>();
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return cudaCreateChannelDesc.template operator()<HVec4>();
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return cudaCreateChannelDesc.template operator()<Vec4>();
    default:
        throw std::runtime_error("Unsupported format for CUDA interop");
    }
}

#endif
