/**
 * @file      application.cpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#include "application.hpp"
#include <GLFW/glfw3.h>
#include <app/settings.hpp>
#include <chrono>
#include <core/helpers.hpp>
#include <core/vulkanSwapchain.hpp>
#include <implot.h>
#include <implot3d.h>
#include <ui/guiStyle.hpp>
#include <ui/imguiBackend.hpp>

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

Application::Application(const ApplicationSettings& settings, bool loadFromPersistedSettings)
    : mName(settings.name)
    , mWindowWidth(settings.windowWidth)
    , mWindowHeight(settings.windowHeight)
    , mFontScale(settings.fontScale)
    , mTheme(settings.theme)
    , mClearColor(settings.clearColor)
    , mVSyncEnabled(settings.vSyncEnabled)
    , mRunning(true)
{
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
    init();
}

Application::~Application()
{
    VK_CHECK(vkDeviceWaitIdle(mLogicalDevice));

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
        std::cerr << "[GLFW] Failed to initialize GLFW." << '\n';
        return false;
    }

    if (!glfwVulkanSupported())
    {
        std::cerr << "[GLFW] Vulkan Not Supported." << '\n';
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
            { VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, {} },
            { VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, {} },
            { VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, {} },
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
        std::cerr << "[vulkan] Failed to create Vulkan context." << std::endl;
        return false;
    }

    mInstance       = mVulkanContext->getInstance();
    mPhysicalDevice = mVulkanContext->getPhysicalDevice();
    mLogicalDevice  = mVulkanContext->getLogicalDevice();

    mQueueGCT = mVulkanContext->getQueue(0);
    mQueueT   = mVulkanContext->getQueue(1);
    mQueueC   = mVulkanContext->getQueue(2);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE); // DPI aware
    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, mName.c_str(), nullptr, nullptr);

    if (!mWindow)
    {
        std::cerr << "[glfw] Failed to create GLFW window." << std::endl;
        return false;
    }

    glfwSetWindowSize(mWindow, mWindowWidth, mWindowHeight);

    // Create Window Surface
    if (glfwCreateWindowSurface(mInstance, mWindow, nullptr, &mSurface) != VK_SUCCESS)
    {
        std::cerr << "[glfw] Failed to create window surface." << std::endl;
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
        std::cerr << "[vulkan] Failed to create Vulkan swapchain." << std::endl;
        return false;
    }

    VkExtent2D swapchainExtent = { static_cast<U32>(mWindowWidth), static_cast<U32>(mWindowHeight) };
    if (mVulkanSwapchain->update(swapchainExtent, mVSyncEnabled) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to update swapchain." << std::endl;
        return false;
    }

    mWindowWidth  = swapchainExtent.width;
    mWindowHeight = swapchainExtent.height;

    // Create Command Pools
    const VkCommandPoolCreateInfo commandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = mQueueGCT.familyIndex
    };
    if (vkCreateCommandPool(mLogicalDevice, &commandPoolCreateInfo, nullptr, &mCommandPool) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create command pool." << std::endl;
        return false;
    }

    const VkCommandPoolCreateInfo transientCommandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = mQueueGCT.familyIndex
    };

    if (vkCreateCommandPool(mLogicalDevice, &transientCommandPoolCreateInfo, nullptr, &mTransientCommandPool) != VK_SUCCESS)
    {
        std::cerr << "[vulkan] Failed to create transient command pool." << std::endl;
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
        std::cerr << "[vulkan] Failed to allocate command buffer." << std::endl;
        return false;
    }

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
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;   // Enable Multi-Viewport / Platform Windows
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
        .DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE * 2,
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
    VK_CHECK(vkDeviceWaitIdle(mLogicalDevice));

    for (const auto& image : mImages)
    {
        destroyImageResources(image.second);
    }

    mImages.clear();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot3D::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    vkFreeCommandBuffers(mLogicalDevice, mCommandPool, static_cast<U32>(mCommandBuffers.size()), mCommandBuffers.data());

    vkDestroyCommandPool(mLogicalDevice, mCommandPool, nullptr);
    vkDestroyCommandPool(mLogicalDevice, mTransientCommandPool, nullptr);

    mVulkanSwapchain->destroy();
    vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
    mVulkanContext->destroy();

    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

bool Application::run()
{
    auto startTime = std::chrono::high_resolution_clock::now();
    const bool res = [&]()
    {
        const VkQueue queue = mQueueGCT.queue;

        while (!glfwWindowShouldClose(mWindow) && mRunning)
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            F32 dt           = std::chrono::duration<F32, std::chrono::seconds::period>(currentTime - startTime).count();
            startTime        = currentTime;

            glfwPollEvents();

#ifdef SAF_CUDA_INTEROP
            for (const auto& image : mImages)
            {
                if (image.second.sharedWithCuda)
                {
                    // transitionImageLayout(image.second.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_NONE, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
                    waitForVulkanCompletion(image.second.cudaExternalWaitSemaphore);
                }
            }
#endif

            for (auto& layer : mLayerStack)
            {
                layer->onUpdate(this, dt);
            }

#ifdef SAF_CUDA_INTEROP
            for (const auto& image : mImages)
            {
                if (image.second.sharedWithCuda)
                {
                    signalCudaCompletion(image.second.cudaExternalSignalSemaphore);
                    // transitionImageLayout(image.second.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_NONE, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                }
            }
#endif

            I32 width, height;
            glfwGetFramebufferSize(mWindow, &width, &height);

            while (width == 0 || height == 0)
            {
                glfwWaitEvents();
                glfwGetFramebufferSize(mWindow, &width, &height);
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
                    std::cerr << "[vulkan] Failed to update swapchain." << std::endl;
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

                if (mOnResizeCallback)
                {
                    mOnResizeCallback(mWindowWidth, mWindowHeight);
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
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

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

                    for (auto& layer : mLayerStack)
                    {
                        layer->onUIRender(this);
                    }

                    ImGui::End();

                    ImGui::Render();
                }

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

                    transitionImageLayout(mVulkanSwapchain->getCurrentImage().image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_NONE, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, commandBuffer);

                    vkCmdBeginRendering(commandBuffer, &renderingInfo);

                    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);

                    vkCmdEndRendering(commandBuffer);

                    transitionImageLayout(mVulkanSwapchain->getCurrentImage().image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, commandBuffer);
                }

                VK_CHECK_RETURN_BOOL(vkEndCommandBuffer(commandBuffer));

                const VkPipelineStageFlags waitStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
#ifdef SAF_CUDA_INTEROP
                // FIXME: Inefficient?
                std::vector<VkSemaphore> waitSemaphores, signalSemaphores;
                std::vector<VkPipelineStageFlags> waitStages;
                waitSemaphores.push_back(acquireSemaphore);
                waitStages.push_back(waitStageFlags);
                signalSemaphores.push_back(presentSemaphore);

                for (const auto& image : mImages)
                {
                    waitSemaphores.push_back(image.second.vkWaitSemaphore);
                    waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

                    signalSemaphores.push_back(image.second.vkSignalSemaphore);
                }

                const VkSubmitInfo submitInfo{
                    .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .waitSemaphoreCount   = static_cast<U32>(waitSemaphores.size()),
                    .pWaitSemaphores      = waitSemaphores.data(),
                    .pWaitDstStageMask    = waitStages.data(),
                    .commandBufferCount   = 1,
                    .pCommandBuffers      = &commandBuffer,
                    .signalSemaphoreCount = static_cast<U32>(signalSemaphores.size()),
                    .pSignalSemaphores    = signalSemaphores.data()
                };
#else
                const VkSubmitInfo submitInfo{
                    .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    .waitSemaphoreCount   = 1,
                    .pWaitSemaphores      = &acquireSemaphore,
                    .pWaitDstStageMask    = &waitStageFlags,
                    .commandBufferCount   = 1,
                    .pCommandBuffers      = &commandBuffer,
                    .signalSemaphoreCount = 1,
                    .pSignalSemaphores    = &presentSemaphore
                };
#endif

                VK_CHECK_RETURN_BOOL(vkQueueSubmit(queue, 1, &submitInfo, inFlightFence));

                ImGuiIO& io = ImGui::GetIO();
                (void)io;
                if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
                {
                    ImGui::UpdatePlatformWindows();
                    ImGui::RenderPlatformWindowsDefault();
                }

                mVulkanSwapchain->present();
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
    mLayerStack.back()->onDetach();
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

    immediateFunction(commandBuffer);

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
    image.width  = createInfo.width;
    image.height = createInfo.height;
    image.format = createInfo.format;
    image.handle = mNextImageHandle++;

    VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = createInfo.format,
        .extent        = { createInfo.width, createInfo.height, 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = createInfo.usage,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

#ifdef SAF_CUDA_INTEROP
    VkExternalMemoryImageCreateInfo externalMemoryImageCreateInfo{};
    if (createInfo.shareWithCUDA)
    {
        imageCreateInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;      // For CUDA read/write access
        imageCreateInfo.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // For reading back to CPU if needed
        imageCreateInfo.flags |= VK_IMAGE_CREATE_ALIAS_BIT;       // Allow aliasing for CUDA interop

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

    VK_CHECK_RETURN(vkCreateImage(mLogicalDevice, &imageCreateInfo, nullptr, &image.image));

#ifdef SAF_CUDA_INTEROP
    // Extra semaphores
    if (createInfo.shareWithCUDA)
    {
        VkSemaphoreCreateInfo semaphoreCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };

#ifdef WIN32
        WindowsSecurityAttributes winSecurityAttributes;

        VkExportSemaphoreWin32HandleInfoKHR exportSemaphoreWin32HandleInfoKHR{
            .sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR,
            .pAttributes = &winSecurityAttributes,
            .dwAccess    = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
            .name        = (LPCWSTR)NULL
        };
#endif
        VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo{
            .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR
        };

#ifdef WIN32
        exportSemaphoreCreateInfo.pNext       = &exportSemaphoreWin32HandleInfoKHR;
        exportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        exportSemaphoreCreateInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        semaphoreCreateInfo.pNext = &exportSemaphoreCreateInfo;

        VK_CHECK_RETURN(vkCreateSemaphore(mLogicalDevice, &semaphoreCreateInfo, nullptr, &image.vkWaitSemaphore));
        VK_CHECK_RETURN(vkCreateSemaphore(mLogicalDevice, &semaphoreCreateInfo, nullptr, &image.vkSignalSemaphore));

        // Signal once, since we update before rendering // NOTE Only solution without Timeline Semaphores
        VkSubmitInfo submitInfo{
            .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores    = &image.vkSignalSemaphore
        };

        VK_CHECK_RETURN(vkQueueSubmit(mQueueGCT.queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK_RETURN(vkQueueWaitIdle(mQueueGCT.queue));

        // import semaphores in CUDA
        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc{};

#ifdef WIN32
        externalSemaphoreHandleDesc.type                = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        externalSemaphoreHandleDesc.handle.win32.handle = getSemaphoreHandle(image.vkWaitSemaphore);
#else
        externalSemaphoreHandleDesc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
        externalSemaphoreHandleDesc.handle.fd = static_cast<I32>(reinterpret_cast<uintptr_t>(getSemaphoreHandle(image.vkWaitSemaphore)));
#endif

        CUDA_CHECK(cudaImportExternalSemaphore(&image.cudaExternalSignalSemaphore, &externalSemaphoreHandleDesc));

#ifdef WIN32
        externalSemaphoreHandleDesc.handle.win32.handle = getSemaphoreHandle(image.vkSignalSemaphore);
#else
        externalSemaphoreHandleDesc.handle.fd = static_cast<I32>(reinterpret_cast<uintptr_t>(getSemaphoreHandle(image.vkSignalSemaphore)));
#endif

        CUDA_CHECK(cudaImportExternalSemaphore(&image.cudaExternalWaitSemaphore, &externalSemaphoreHandleDesc));
    }
#endif

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
    const WindowsSecurityAttributes securityAttributes{
        .nLength        = sizeof(WindowsSecurityAttributes),
        .bInheritHandle = TRUE
    };

    const VkExportMemoryWin32HandleInfoKHR exportMemoryWin32HandleInfo{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR,
        .pAttributes = &securityAttributes,
        .dwAccess    = GENERIC_ALL
    };
#endif
    if (createInfo.shareWithCUDA)
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
    if (createInfo.shareWithCUDA)
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

    image.descriptorSet = ImGui_ImplVulkan_AddTexture(image.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    mImages.push_back({ image.handle, image });
    outImage = image.handle;

    return VK_SUCCESS;
}

void Application::destroyImage(ImageHandle image)
{
    for (auto it = mImages.begin(); it != mImages.end(); ++it)
    {
        if (it->first == image)
        {
            destroyImageResources(it->second);
            mImages.erase(it);
            return;
        }
    }
}

void Application::destroyImageResources(const VulkanImage& vulkanImage)
{
    ImGui_ImplVulkan_RemoveTexture(vulkanImage.descriptorSet);

    vkDestroySampler(mLogicalDevice, vulkanImage.sampler, nullptr);
    vkDestroyImageView(mLogicalDevice, vulkanImage.imageView, nullptr);
    vkDestroyImage(mLogicalDevice, vulkanImage.image, nullptr);
    vkFreeMemory(mLogicalDevice, vulkanImage.deviceMemory, nullptr);

#ifdef SAF_CUDA_INTEROP
    if (vulkanImage.sharedWithCuda)
    {
        CUDA_CHECK(cudaDestroyTextureObject(vulkanImage.cudaTextureObject));
        CUDA_CHECK(cudaDestroySurfaceObject(vulkanImage.cudaSurfaceObject));
        CUDA_CHECK(cudaFreeMipmappedArray(vulkanImage.cudaMipmappedImageArray));
        CUDA_CHECK(cudaDestroyExternalMemory(vulkanImage.cudaExternalImageMemory));

        vkDestroySemaphore(mLogicalDevice, vulkanImage.vkWaitSemaphore, nullptr);
        vkDestroySemaphore(mLogicalDevice, vulkanImage.vkSignalSemaphore, nullptr);

        CUDA_CHECK(cudaDestroyExternalSemaphore(vulkanImage.cudaExternalWaitSemaphore));
        CUDA_CHECK(cudaDestroyExternalSemaphore(vulkanImage.cudaExternalSignalSemaphore));
    }
#endif
}

VkResult Application::uploadImage(ImageHandle image, const void* data, size_t size)
{
    bool found = false;
    VulkanImage vulkanImage;
    for (const auto& img : mImages)
    {
        if (img.first == image)
        {
            vulkanImage = img.second;
            found       = true;
            break;
        }
    }
    if (!found)
    {
        return VK_ERROR_UNKNOWN;
    }

    // check size of data against image size
    const size_t imageSize = vulkanImage.width * vulkanImage.height * getFormatBytesPerPixel(vulkanImage.format);
    if (size != imageSize)
    {
        return VK_ERROR_UNKNOWN;
    }

    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    const VkBufferCreateInfo bufferCreateInfo{
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VK_CHECK_RETURN(vkCreateBuffer(mLogicalDevice, &bufferCreateInfo, nullptr, &stagingBuffer));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(mLogicalDevice, stagingBuffer, &memoryRequirements);

    const VkMemoryAllocateInfo memoryAllocateInfo{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = memoryRequirements.size,
        .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };

    VK_CHECK_RETURN(vkAllocateMemory(mLogicalDevice, &memoryAllocateInfo, nullptr, &stagingBufferMemory));
    VK_CHECK_RETURN(vkBindBufferMemory(mLogicalDevice, stagingBuffer, stagingBufferMemory, 0));

    // Copy data to staging buffer
    void* mappedData;
    VK_CHECK_RETURN(vkMapMemory(mLogicalDevice, stagingBufferMemory, 0, size, 0, &mappedData));
    std::memcpy(mappedData, data, size);
    vkUnmapMemory(mLogicalDevice, stagingBufferMemory);

    // Copy from staging buffer to image

    VK_CHECK_RETURN(executeSingleTimeCommandBuffer([&](VkCommandBuffer commandBuffer)
                                                   {

                                                       transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, commandBuffer);

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

                                                       vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, vulkanImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bufferImageCopy);

                                                       transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, commandBuffer);

                                                       return VK_SUCCESS; }));

    vkDestroyBuffer(mLogicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(mLogicalDevice, stagingBufferMemory, nullptr);

    return VK_SUCCESS;
}

VkResult Application::downloadImage(ImageHandle image, void* data, size_t size)
{
    bool found = false;
    VulkanImage vulkanImage;
    for (const auto& img : mImages)
    {
        if (img.first == image)
        {
            vulkanImage = img.second;
            found       = true;
            break;
        }
    }
    if (!found)
    {
        return VK_ERROR_UNKNOWN;
    }

    // check size of data against image size
    const size_t imageSize = vulkanImage.width * vulkanImage.height * getFormatBytesPerPixel(vulkanImage.format);
    if (size != imageSize)
    {
        return VK_ERROR_UNKNOWN;
    }

    // Create linear tiled image for copying
    VkImage linearImage;
    VkDeviceMemory linearImageMemory;

    const VkImageCreateInfo imageCreateInfo{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = vulkanImage.format,
        .extent        = { vulkanImage.width, vulkanImage.height, 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_LINEAR,
        .usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
    };

    VK_CHECK_RETURN(vkCreateImage(mLogicalDevice, &imageCreateInfo, nullptr, &linearImage));

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(mLogicalDevice, linearImage, &memoryRequirements);

    const VkMemoryAllocateInfo memoryAllocateInfo{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = memoryRequirements.size,
        .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };

    VK_CHECK_RETURN(vkAllocateMemory(mLogicalDevice, &memoryAllocateInfo, nullptr, &linearImageMemory));
    VK_CHECK_RETURN(vkBindImageMemory(mLogicalDevice, linearImage, linearImageMemory, 0));

    // Copy from image to linear image
    VK_CHECK_RETURN(executeSingleTimeCommandBuffer([&](VkCommandBuffer commandBuffer)
                                                   {
                                                       transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, commandBuffer);
                                                       transitionImageLayout(linearImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, commandBuffer);

                                                       const VkImageCopy imageCopy{
                                                           .srcSubresource = {
                                                               .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                                               .mipLevel       = 0,
                                                               .baseArrayLayer = 0,
                                                               .layerCount     = 1 },
                                                           .srcOffset      = { 0, 0, 0 },
                                                           .dstSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
                                                           .dstOffset      = { 0, 0, 0 },
                                                           .extent         = { vulkanImage.width, vulkanImage.height, 1 }
                                                       };

                                                       vkCmdCopyImage(commandBuffer, vulkanImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, linearImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);

                                                       transitionImageLayout(linearImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, commandBuffer);
                                                       transitionImageLayout(vulkanImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, commandBuffer);

                                                       return VK_SUCCESS; }));

    const VkImageSubresource subresource{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .arrayLayer = 0 };
    VkSubresourceLayout layout;
    vkGetImageSubresourceLayout(mLogicalDevice, linearImage, &subresource, &layout);

    void* mappedData;
    VK_CHECK_RETURN(vkMapMemory(mLogicalDevice, linearImageMemory, 0, VK_WHOLE_SIZE, 0, &mappedData));

    mappedData += layout.offset * sizeof(Byte);
    std::memcpy(data, mappedData, size);
    vkUnmapMemory(mLogicalDevice, linearImageMemory);

    vkFreeMemory(mLogicalDevice, linearImageMemory, nullptr);
    vkDestroyImage(mLogicalDevice, linearImage, nullptr);

    return VK_SUCCESS;
}

#ifdef SAF_CUDA_INTEROP
void Application::waitForVulkanCompletion(cudaExternalSemaphore_t semaphore, cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams{};

    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&semaphore, &externalSemaphoreWaitParams, 1, stream));
}

void Application::signalCudaCompletion(cudaExternalSemaphore_t semaphore, cudaStream_t stream)
{
    cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams{};

    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&semaphore, &externalSemaphoreSignalParams, 1, stream));
}
#endif

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
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR
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