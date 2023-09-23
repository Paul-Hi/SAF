/**
 * @file      application.cpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2023
 * @copyright Apache License 2.0
 */

#include "application.hpp"
#include <core/immediateSubmit.hpp>
#include <core/vulkanHelper.hpp>
#include <ui/guiStyle.hpp>
#include <ui/imguiBackend.hpp>

using namespace saf;

// This is really similar to imgui example to simplify things
#define UNLIMITED_FRAME_RATE

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#ifdef SAF_DEBUG
#define VULKAN_DEBUG_MESSENGER
#endif

// Data
static VkAllocationCallbacks *gAllocator = nullptr;
static VkInstance gInstance = VK_NULL_HANDLE;
static VkPhysicalDevice gPhysicalDevice = VK_NULL_HANDLE;
static VkDevice gDevice = VK_NULL_HANDLE;
static U32 gQueueFamily = static_cast<U32>(-1);
static VkQueue gQueue = VK_NULL_HANDLE;
#ifdef VULKAN_DEBUG_MESSENGER
static VkDebugUtilsMessengerEXT gDebugMessenger = VK_NULL_HANDLE;
#endif
static VkPipelineCache gPipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool gDescriptorPool = VK_NULL_HANDLE;

static VulkanContext gVulkanContext;
static int gMinImageCount = 2;
static bool gSwapChainRebuild = false;

static void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "[glfw] Error " << error << ": " << description << '\n';
}

#ifdef VULKAN_DEBUG_MESSENGER
static VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                         const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
    std::cerr << "[vulkan] Validation Layer: " << pCallbackData->pMessage << '\n';

    (void)(messageSeverity);
    (void)(messageType);
    (void)(pUserData);

    // Always return VK_FALSE
    return VK_FALSE;
}

static VkDebugUtilsMessengerCreateInfoEXT gDebugUtilsCreateInfo{
    VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
    nullptr,
    0,
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT // | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
    ,
    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
    debugUtilsCallback,
    nullptr};
#endif // VULKAN_DEBUG_MESSENGER

static bool isExtensionAvailable(const ImVector<VkExtensionProperties> &properties, const char *extension)
{
    for (const VkExtensionProperties &p : properties)
    {
        if (strcmp(p.extensionName, extension) == 0)
        {
            return true;
        }
    }
    return false;
}

static VkPhysicalDevice selectPhysicalDevice()
{
    U32 gpuCount;
    VkResult err = vkEnumeratePhysicalDevices(gInstance, &gpuCount, nullptr);
    checkVkResult(err);
    SAF_ASSERT(gpuCount > 0);

    ImVector<VkPhysicalDevice> gpus;
    gpus.resize(static_cast<I32>(gpuCount));
    err = vkEnumeratePhysicalDevices(gInstance, &gpuCount, gpus.Data);
    checkVkResult(err);

    for (VkPhysicalDevice &device : gpus)
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

static void setupVulkan(ImVector<const char *> instanceExtensions)
{
    VkResult err;

    // Create Vulkan Instance
    {
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

        U32 propertiesCount;
        ImVector<VkExtensionProperties> properties;
        vkEnumerateInstanceExtensionProperties(nullptr, &propertiesCount, nullptr);
        properties.resize(static_cast<I32>(propertiesCount));
        err = vkEnumerateInstanceExtensionProperties(nullptr, &propertiesCount, properties.Data);
        checkVkResult(err);

        if (isExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if (isExtensionAvailable(properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
        {
            instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

        // Enabling validation layers
#ifdef VULKAN_DEBUG_MESSENGER
        const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = layers;
        createInfo.pNext = static_cast<VkDebugUtilsMessengerCreateInfoEXT *>(&gDebugUtilsCreateInfo);
        instanceExtensions.push_back("VK_EXT_debug_utils");
#endif
        // Create Vulkan Instance
        createInfo.enabledExtensionCount = static_cast<U32>(instanceExtensions.Size);
        createInfo.ppEnabledExtensionNames = instanceExtensions.Data;
        err = vkCreateInstance(&createInfo, gAllocator, &gInstance);
        checkVkResult(err);

        // Setup the debug messenger
#ifdef VULKAN_DEBUG_MESSENGER
        auto vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(gInstance, "vkCreateDebugUtilsMessengerEXT");
        SAF_ASSERT(vkCreateDebugUtilsMessengerEXT != nullptr);
        err = vkCreateDebugUtilsMessengerEXT(gInstance, &gDebugUtilsCreateInfo, gAllocator, &gDebugMessenger);
        checkVkResult(err);
#endif
    }

    // Select Device
    gPhysicalDevice = selectPhysicalDevice();

    // Select graphics queue family
    {
        U32 queueCount;
        vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevice, &queueCount, nullptr);
        VkQueueFamilyProperties *queues = static_cast<VkQueueFamilyProperties *>(malloc(sizeof(VkQueueFamilyProperties) * queueCount));
        vkGetPhysicalDeviceQueueFamilyProperties(gPhysicalDevice, &queueCount, queues);
        for (U32 i = 0; i < queueCount; i++)
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                gQueueFamily = i;
                break;
            }
        free(queues);
        SAF_ASSERT(gQueueFamily != (U32)-1);
    }

    // Create Logical Device
    {
        ImVector<const char *> deviceExtensions;
        deviceExtensions.push_back("VK_KHR_swapchain");

        // Enumerate extensions
        U32 propertiesCount;
        ImVector<VkExtensionProperties> properties;
        vkEnumerateDeviceExtensionProperties(gPhysicalDevice, nullptr, &propertiesCount, nullptr);
        properties.resize(static_cast<I32>(propertiesCount));
        vkEnumerateDeviceExtensionProperties(gPhysicalDevice, nullptr, &propertiesCount, properties.Data);
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        if (isExtensionAvailable(properties, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME))
            deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

        const F32 queue_priority[] = {1.0f};
        VkDeviceQueueCreateInfo queue_info[1] = {};
        queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[0].queueFamilyIndex = gQueueFamily;
        queue_info[0].queueCount = 1;
        queue_info[0].pQueuePriorities = queue_priority;
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]);
        createInfo.pQueueCreateInfos = queue_info;
        createInfo.enabledExtensionCount = static_cast<U32>(deviceExtensions.Size);
        createInfo.ppEnabledExtensionNames = deviceExtensions.Data;
        err = vkCreateDevice(gPhysicalDevice, &createInfo, gAllocator, &gDevice);
        checkVkResult(err);
        vkGetDeviceQueue(gDevice, gQueueFamily, 0, &gQueue);
    }

    // Create Descriptor Pool
    {
        VkDescriptorPoolSize poolSizes[] =
            {
                {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1000 * ARRAYSIZE(poolSizes);
        poolInfo.poolSizeCount = static_cast<U32>(ARRAYSIZE(poolSizes));
        poolInfo.pPoolSizes = poolSizes;
        err = vkCreateDescriptorPool(gDevice, &poolInfo, gAllocator, &gDescriptorPool);
        checkVkResult(err);
    }
}

static void setupVulkanWindow(VulkanContext *context, VkSurfaceKHR surface, int width, int height)
{
    context->surface = surface;

    // Check for WSI support
    VkBool32 res;
    vkGetPhysicalDeviceSurfaceSupportKHR(gPhysicalDevice, gQueueFamily, context->surface, &res);
    if (res != VK_TRUE)
    {
        std::cerr << "Error no WSI support on physical device 0!" << '\n';
        exit(-1);
    }

    // Select Surface Format
    const VkFormat requestSurfaceImageFormat[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
    const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    context->surfaceFormat = vkSelectSurfaceFormat(gPhysicalDevice, context->surface, requestSurfaceImageFormat, static_cast<PtrSize>(ARRAYSIZE(requestSurfaceImageFormat)), requestSurfaceColorSpace);

    // Select Present Mode
#ifdef UNLIMITED_FRAME_RATE
    VkPresentModeKHR presentModes[] = {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR};
#else
    VkPresentModeKHR presentModes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
    context->presentMode = vkSelectPresentMode(gPhysicalDevice, context->surface, &presentModes[0], ARRAYSIZE(presentModes));

    // Create SwapChain, RenderPass, Framebuffer, etc.
    SAF_ASSERT(gMinImageCount >= 2);
    vkCreateOrResizeContext(gInstance, gPhysicalDevice, gDevice, context, gQueueFamily, gAllocator, width, height, static_cast<U32>(gMinImageCount));
}

static void cleanupVulkan()
{
    vkDestroyDescriptorPool(gDevice, gDescriptorPool, gAllocator);

#ifdef VULKAN_DEBUG_MESSENGER
    // Remove the debug report callback
    auto vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(gInstance, "vkDestroyDebugUtilsMessengerEXT");
    SAF_ASSERT(vkDestroyDebugUtilsMessengerEXT != nullptr);
    vkDestroyDebugUtilsMessengerEXT(gInstance, gDebugMessenger, gAllocator);
#endif // VULKAN_DEBUG_MESSENGER

    vkDestroyDevice(gDevice, gAllocator);
    vkDestroyInstance(gInstance, gAllocator);
}

static void cleanupVulkanWindow()
{
    vkDestroyContext(gInstance, gDevice, &gVulkanContext, gAllocator);
}

static void frameRender(VulkanContext *context, ImDrawData *draw_data)
{
    VkResult err;

    VkSemaphore imageAcquiredSemaphore = context->frameSemaphores[context->semaphoreIndex].imageAcquiredSemaphore;
    VkSemaphore renderCompleteSemaphore = context->frameSemaphores[context->semaphoreIndex].renderCompleteSemaphore;
    err = vkAcquireNextImageKHR(gDevice, context->swapchain, UINT64_MAX, imageAcquiredSemaphore, VK_NULL_HANDLE, &context->frameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    {
        gSwapChainRebuild = true;
        return;
    }
    checkVkResult(err);

    VulkanFrameData *fd = &context->frames[context->frameIndex];
    {
        err = vkWaitForFences(gDevice, 1, &fd->fence, VK_TRUE, UINT64_MAX); // wait indefinitely instead of periodically checking
        checkVkResult(err);

        err = vkResetFences(gDevice, 1, &fd->fence);
        checkVkResult(err);
    }
    {
        err = vkResetCommandPool(gDevice, fd->commandPool, 0);
        checkVkResult(err);
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(fd->commandBuffer, &info);
        checkVkResult(err);
    }
    {
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = context->renderPass;
        info.framebuffer = fd->framebuffer;
        info.renderArea.extent.width = static_cast<U32>(context->width);
        info.renderArea.extent.height = static_cast<U32>(context->height);
        info.clearValueCount = 1;
        info.pClearValues = &context->clearValue;
        vkCmdBeginRenderPass(fd->commandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    // Record dear imgui primitives into command buffer
    vkRenderImGuiDrawData(draw_data, fd->commandBuffer);

    // Submit command buffer
    vkCmdEndRenderPass(fd->commandBuffer);
    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &imageAcquiredSemaphore;
        info.pWaitDstStageMask = &wait_stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &fd->commandBuffer;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &renderCompleteSemaphore;

        err = vkEndCommandBuffer(fd->commandBuffer);
        checkVkResult(err);
        err = vkQueueSubmit(gQueue, 1, &info, fd->fence);
        checkVkResult(err);
    }
}

static void framePresent(VulkanContext *context)
{
    if (gSwapChainRebuild)
        return;
    VkSemaphore renderCompleteSemaphore = context->frameSemaphores[context->semaphoreIndex].renderCompleteSemaphore;
    VkPresentInfoKHR info = {};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &renderCompleteSemaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &context->swapchain;
    info.pImageIndices = &context->frameIndex;
    VkResult err = vkQueuePresentKHR(gQueue, &info);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
    {
        gSwapChainRebuild = true;
        return;
    }
    checkVkResult(err);
    context->semaphoreIndex = (context->semaphoreIndex + 1) % context->framesInFlight; // Now we can use the next set of semaphores
}

Application::Application(const ApplicationSettings &settings)
    : mName(settings.name), mWindowWidth(settings.windowWidth), mWindowHeight(settings.windowHeight), mClearColor(settings.clearColor), mRunning(true)
{
    initVulkanGLFW();
}

Application::~Application()
{
    VkResult err = vkDeviceWaitIdle(gDevice);
    checkVkResult(err);

    while (!mLayerStack.empty())
    {
        popLayer();
    }

    shutdownVulkanGLFW();
}

bool Application::initVulkanGLFW()
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        return false;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, mName.c_str(), nullptr, nullptr);
    if (!glfwVulkanSupported())
    {
        std::cerr << "GLFW: Vulkan Not Supported" << '\n';
        return false;
    }

    ImVector<const char *> extensions;
    U32 extensionsCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionsCount);
    for (U32 i = 0; i < extensionsCount; i++)
        extensions.push_back(glfwExtensions[i]);
    setupVulkan(extensions);

    // Create Window Surface
    VkSurfaceKHR surface;
    VkResult err = glfwCreateWindowSurface(gInstance, mWindow, gAllocator, &surface);
    checkVkResult(err);

    // Create Framebuffers
    I32 w, h;
    glfwGetFramebufferSize(mWindow, &w, &h);
    mVulkanContext = &gVulkanContext;
    setupVulkanWindow(mVulkanContext, surface, w, h);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;   // Enable Multi-Viewport / Platform Windows
    // io.ConfigViewportsNoAutoMerge = true;
    // io.ConfigViewportsNoTaskBarIcon = true;

    // Setup Dear ImGui style
    setupImGuiStyle();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform mWindows can look identical to regular ones.
    ImGuiStyle &style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(mWindow, true);
    VulkanInitInfo initInfo = {};
    initInfo.instance = gInstance;
    initInfo.physicalDevice = gPhysicalDevice;
    initInfo.device = gDevice;
    initInfo.queueFamily = gQueueFamily;
    initInfo.queue = gQueue;
    initInfo.pipelineCache = gPipelineCache;
    initInfo.descriptorPool = gDescriptorPool;
    initInfo.subpass = 0;
    initInfo.minImageCount = static_cast<U32>(gMinImageCount);
    initInfo.imageCount = mVulkanContext->framesInFlight;
    initInfo.msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.allocator = gAllocator;
    initInfo.checkVkResultFn = checkVkResult;
    vkInit(&initInfo, mVulkanContext->renderPass);

    mCommandPool = mVulkanContext->ressourceCommandPool;
    mCommandBuffer = mVulkanContext->ressourceCommandBuffer;

    // Upload Fonts
    {

        ImmediateSubmit::execute(gDevice, gQueue, mCommandPool, mCommandBuffer, [&](VkCommandBuffer cmd)
                                 { vkCreateImGuiFontsTexture(cmd); });

        vkDestroyImGuiFontUploadObjects();
    }

    mPhysicalDevice = gPhysicalDevice;
    mLogicalDevice = gDevice;
    mQueue = gQueue;

    return true;
}

void Application::shutdownVulkanGLFW()
{
    VkResult err = vkDeviceWaitIdle(gDevice);
    checkVkResult(err);
    vkShutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    cleanupVulkanWindow();
    cleanupVulkan();

    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

void Application::run()
{
    while (!glfwWindowShouldClose(mWindow) && mRunning)
    {
        glfwPollEvents();

        for (auto &layer : mLayerStack)
        {
            layer->onUpdate(this);
        }

        if (gSwapChainRebuild)
        {
            I32 width, height;
            glfwGetFramebufferSize(mWindow, &width, &height);
            if (width > 0 && height > 0)
            {
                vkSetMinImageCount(static_cast<U32>(gMinImageCount));
                vkCreateOrResizeContext(gInstance, gPhysicalDevice, gDevice, &gVulkanContext, gQueueFamily, gAllocator, width, height, static_cast<U32>(gMinImageCount));
                gVulkanContext.frameIndex = 0;
                gSwapChainRebuild = false;
            }
        }

        vkNewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Dockspace
        ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        static ImGuiDockNodeFlags dockNodeFlags = ImGuiDockNodeFlags_PassthruCentralNode;

        static ImGuiWindowFlags dockingWindowFlags;
        dockingWindowFlags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        dockingWindowFlags |= ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("DockSpace", nullptr, dockingWindowFlags);
        ImGui::PopStyleVar(3);

        // Real Dockspace
        ImGuiID dockspaceId = ImGui::GetID("SAFDockSpace");
        ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), dockNodeFlags);

        // Menu bar
        if (mMenubarCallback)
        {
            if (ImGui::BeginMenuBar())
            {
                mMenubarCallback();
                ImGui::EndMenuBar();
            }
        }

        for (auto &layer : mLayerStack)
        {
            layer->onUIRender();
        }

        ImGui::End();

        ImGui::Render();
        ImDrawData *mainDrawData = ImGui::GetDrawData();
        const bool mainMinimized = (mainDrawData->DisplaySize.x <= 0.0f || mainDrawData->DisplaySize.y <= 0.0f);
        mVulkanContext->clearValue.color.float32[0] = mClearColor.x() * mClearColor.w();
        mVulkanContext->clearValue.color.float32[1] = mClearColor.y() * mClearColor.w();
        mVulkanContext->clearValue.color.float32[2] = mClearColor.z() * mClearColor.w();
        mVulkanContext->clearValue.color.float32[3] = mClearColor.w();
        if (!mainMinimized)
        {
            frameRender(mVulkanContext, mainDrawData);
        }

        ImGuiIO &io = ImGui::GetIO();
        (void)io;
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        if (!mainMinimized)
        {
            framePresent(mVulkanContext);
        }
    }
}

void Application::close()
{
    mRunning = false;
}

void Application::popLayer()
{
    mLayerStack.back()->onDetach();
    mLayerStack.pop_back();
}
