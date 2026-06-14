/**
 * @file      vulkanContext.cpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#include "vulkanContext.hpp"

#include <core/helpers.hpp>

using namespace saf;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        const auto severity = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity)
        {
            if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                return "ERROR";
            if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
                return "WARNING";
            if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
                return "INFO";
            if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
                return "VERBOSE";
            return "UNKNOWN";
        };

        std::cerr << "[vulkan] Validation Layer: "
                  << severity(messageSeverity) << " (" << pCallbackData->pMessageIdName << ", ID " << pCallbackData->messageIdNumber << "): "
                  << pCallbackData->pMessage << std::endl;
    }

    (void)(pUserData);

    // Always return VK_FALSE
    return VK_FALSE;
}

VkResult VulkanContext::create(const ContextCreateInfo& createInfo)
{
    VK_CHECK_RETURN(volkInitialize());

    VK_CHECK_RETURN(createInstance(createInfo));
    VK_CHECK_RETURN(selectPhysicalDevice(createInfo));
    VK_CHECK_RETURN(createLogicalDevice(createInfo));

    if (createInfo.verbose)
    {
        VK_CHECK_RETURN(printVulkanVersion());
        VK_CHECK_RETURN(printGpus());
        VK_CHECK_RETURN(printQueues());
    }

    return VK_SUCCESS;
}

void VulkanContext::destroy()
{
    if (mLogicalDevice != VK_NULL_HANDLE)
    {
        vkDestroyDevice(mLogicalDevice, nullptr);
        mLogicalDevice = VK_NULL_HANDLE;
    }

    if (mDebugMessenger != VK_NULL_HANDLE)
    {
        vkDestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
        mDebugMessenger = VK_NULL_HANDLE;
    }

    if (mInstance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(mInstance, nullptr);
        mInstance = VK_NULL_HANDLE;
    }
}

VkResult VulkanContext::createInstance(const ContextCreateInfo& createInfo)
{
    const VkApplicationInfo applicationInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = createInfo.applicationName,
        .applicationVersion = VK_MAKE_API_VERSION(1, 0, 0, 0),
        .pEngineName        = "SAF Engine",
        .engineVersion      = VK_MAKE_API_VERSION(1, 0, 0, 0),
        .apiVersion         = createInfo.apiVersion,
    };

    std::vector<const char*> layers;
    std::vector<const char*> instanceExtensions = createInfo.instanceExtensions;

    if (createInfo.enableValidationLayers)
    {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

#ifdef SAF_CUDA_INTEROP
    if (createInfo.enableCudaInteroperability)
    {
        instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    }
#endif

    VK_CHECK_RETURN(checkInstanceExtensionSupport(instanceExtensions));

    const VkInstanceCreateInfo instanceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo        = &applicationInfo,
        .enabledLayerCount       = static_cast<U32>(layers.size()),
        .ppEnabledLayerNames     = layers.data(),
        .enabledExtensionCount   = static_cast<U32>(instanceExtensions.size()),
        .ppEnabledExtensionNames = instanceExtensions.data()
    };

    VK_CHECK_RETURN(vkCreateInstance(&instanceCreateInfo, nullptr, &mInstance));

    volkLoadInstance(mInstance);

    if (createInfo.enableValidationLayers)
    {
        const VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{
            .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback
        };

        VK_CHECK_RETURN(vkCreateDebugUtilsMessengerEXT(mInstance, &debugCreateInfo, nullptr, &mDebugMessenger));
    }

    return VK_SUCCESS;
}

VkResult VulkanContext::selectPhysicalDevice(const ContextCreateInfo& createInfo)
{
    U32 deviceCount = 0;
    VK_CHECK_RETURN(vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr));

    if (deviceCount == 0)
    {
        std::cerr << "Failed to find a device supporting Vulkan!" << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK_RETURN(vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data()));

    mPhysicalDevice = devices[0]; // Default to the first device
    for (const auto& device : devices)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            mPhysicalDevice = device;
            break;
        }
    }

    // check version
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &properties);
    if (properties.apiVersion < createInfo.apiVersion)
    {
        std::cerr << "Selected GPU does not support the required Vulkan API version!" << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

#ifdef SAF_CUDA_INTEROP
    if (createInfo.enableCudaInteroperability)
    {
        VkPhysicalDeviceIDProperties idProperties{};
        idProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        idProperties.pNext = NULL;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &idProperties;
        vkGetPhysicalDeviceProperties2(mPhysicalDevice, &properties2);
        VK_CHECK_RETURN(setupVulkanCudaInteropDevice(&(idProperties.deviceUUID[0])));
    }
#endif

    VK_CHECK_RETURN(selectQueueFamilies(createInfo));

    return VK_SUCCESS;
}

#ifdef SAF_CUDA_INTEROP
VkResult VulkanContext::setupVulkanCudaInteropDevice(const Byte* vkDeviceUUID)
{
    I32 deviceCount = 0;

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "[cuda] Error : No CUDA capable GPU found.\n";
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    I32 currentCuDevice   = 0;
    I32 devicesProhibited = 0;

    while (currentCuDevice < deviceCount)
    {
        cudaGetDeviceProperties(&deviceProp, currentCuDevice);

        if ((deviceProp.computeMode != cudaComputeModeProhibited))
        {
            int ret = std::memcmp(&deviceProp.uuid, vkDeviceUUID, VK_UUID_SIZE);
            if (ret == 0)
            {
                CUDA_CHECK(cudaSetDevice(currentCuDevice));
                CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, currentCuDevice));
                // std::cout << "Using GPU Device " << currentCuDevice << " " << deviceProp.name << " with capability " << deviceProp.major << "." << deviceProp.minor << '\n';
            }
        }
        else
        {
            devicesProhibited++;
        }

        currentCuDevice++;
    }

    if (devicesProhibited == deviceCount)
    {
        std::cerr << "[cuda] Error : No Vulkan-CUDA Interop capable GPU found.\n";
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return VK_SUCCESS;
}
#endif

VkResult VulkanContext::selectQueueFamilies(const ContextCreateInfo& createInfo)
{

    U32 queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, queueFamilies.data());

    std::unordered_map<U32, U32> queueFamilyUsage;
    for (U32 i = 0; i < queueFamilyCount; ++i)
    {
        queueFamilyUsage[i] = 0;
    }

    const auto findQueue = [&](VkQueueFlags requestedQueue)
    {
        // Do not use VK_QUEUE_GRAPHICS_BIT if not needed

        for (U32 j = 0; j < queueFamilyCount; ++j)
        {
            // exact match, unused queue family
            if ((queueFamilies[j].queueFlags & requestedQueue) == requestedQueue && queueFamilyUsage[j] == 0 &&
                ((requestedQueue & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
            {
                mQueues.push_back({ j, queueFamilyUsage[j] });
                queueFamilyUsage[j]++;
                return true;
            }
        }

        for (U32 j = 0; j < queueFamilyCount; ++j)
        {
            // exact match, allow reuse if queue count not exceeded
            if ((queueFamilies[j].queueFlags & requestedQueue) == requestedQueue &&
                queueFamilyUsage[j] < queueFamilies[j].queueCount &&
                ((requestedQueue & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
            {
                mQueues.push_back({ j, queueFamilyUsage[j] });
                queueFamilyUsage[j]++;
                return true;
            }
        }

        for (U32 j = 0; j < queueFamilyCount; ++j)
        {
            //  partial match, allow reuse if queue count not exceeded
            if ((queueFamilies[j].queueFlags & requestedQueue) && queueFamilyUsage[j] < queueFamilies[j].queueCount &&
                ((requestedQueue & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
            {
                mQueues.push_back({ j, queueFamilyUsage[j] });
                queueFamilyUsage[j]++;
                return true;
            }
        }

        for (U32 j = 0; j < queueFamilyCount; ++j)
        {
            //  partial match, allow reuse if queue count not exceeded, allow graphics queues
            if ((queueFamilies[j].queueFlags & requestedQueue) && queueFamilyUsage[j] < queueFamilies[j].queueCount)
            {
                mQueues.push_back({ j, queueFamilyUsage[j] });
                queueFamilyUsage[j]++;
                return true;
            }
        }

        return false;
    };

    for (const auto requestedQueue : createInfo.queues)
    {
        if (!findQueue(requestedQueue))
        {
            std::cerr << "Failed to find a suitable queue family for desired queue flags: " << requestedQueue << std::endl;
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    for (const auto& usage : queueFamilyUsage)
    {
        if (usage.second > 0)
        {
            mQueuePriorities.emplace_back(usage.second, 1.0f); // Same priority for all queues in a family
            mQueueCreateInfos.push_back(
                VkDeviceQueueCreateInfo{
                    .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .pNext            = nullptr,
                    .flags            = 0,
                    .queueFamilyIndex = usage.first,
                    .queueCount       = usage.second,
                    .pQueuePriorities = mQueuePriorities.back().data() });
        }
    }

    return VK_SUCCESS;
}

VkResult VulkanContext::createLogicalDevice(const ContextCreateInfo& createInfo)
{
    VkPhysicalDeviceVulkan13Features deviceFeatures13{
        .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .dynamicRendering = true,
    };

    VkPhysicalDeviceFeatures2 deviceFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &deviceFeatures13,
    };

    std::vector<const char*> deviceExtensions;
    deviceExtensions.reserve(createInfo.deviceExtensions.size());
    for (const auto& extension : createInfo.deviceExtensions)
    {
        deviceExtensions.push_back(extension.extensionName);
    }

    VK_CHECK_RETURN(checkDeviceExtensionSupport(deviceExtensions));

    const VkDeviceCreateInfo deviceCreateInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &deviceFeatures,
        .queueCreateInfoCount    = static_cast<U32>(mQueueCreateInfos.size()),
        .pQueueCreateInfos       = mQueueCreateInfos.data(),
        .enabledExtensionCount   = static_cast<U32>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data()
    };

    VK_CHECK_RETURN(vkCreateDevice(mPhysicalDevice, &deviceCreateInfo, nullptr, &mLogicalDevice));

    volkLoadDevice(mLogicalDevice);

    for (auto& queue : mQueues)
    {
        vkGetDeviceQueue(mLogicalDevice, queue.familyIndex, queue.queueIndex, &queue.queue);
    }

    return VK_SUCCESS;
}

VkResult VulkanContext::checkInstanceExtensionSupport(const std::vector<const char*>& requiredExtensions)
{
    U32 propertiesCount;
    std::vector<VkExtensionProperties> properties;
    VK_CHECK_RETURN(vkEnumerateInstanceExtensionProperties(nullptr, &propertiesCount, nullptr));
    properties.resize(static_cast<I32>(propertiesCount));
    VK_CHECK_RETURN(vkEnumerateInstanceExtensionProperties(nullptr, &propertiesCount, properties.data()));
    for (const char* requiredExtension : requiredExtensions)
    {
        bool found = false;
        for (const VkExtensionProperties& property : properties)
        {
            if (strcmp(requiredExtension, property.extensionName) == 0)
            {
                found = true;
                break;
            }
        }
        if (!found)
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

VkResult VulkanContext::checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions)
{
    U32 propertiesCount;
    std::vector<VkExtensionProperties> properties;
    VK_CHECK_RETURN(vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &propertiesCount, nullptr));
    properties.resize(static_cast<I32>(propertiesCount));
    VK_CHECK_RETURN(vkEnumerateDeviceExtensionProperties(mPhysicalDevice, nullptr, &propertiesCount, properties.data()));
    for (const char* requiredExtension : requiredExtensions)
    {
        bool found = false;
        for (const VkExtensionProperties& property : properties)
        {
            if (strcmp(requiredExtension, property.extensionName) == 0)
            {
                found = true;
                break;
            }
        }
        if (!found)
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

VkResult VulkanContext::printVulkanVersion()
{
    U32 version;
    VK_CHECK_RETURN(vkEnumerateInstanceVersion(&version));

    std::cout << "Vulkan API Version: " << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "." << VK_VERSION_PATCH(version) << std::endl;

    return VK_SUCCESS;
}

static std::string getVendorName(uint32_t vendorID)
{
    static const std::unordered_map<uint32_t, std::string> vendorMap = {
        { 0x1002, "AMD" },
        { 0x1010, "ImgTec" },
        { 0x10DE, "NVIDIA" },
        { 0x13B5, "ARM" },
        { 0x5143, "Qualcomm" },
        { 0x8086, "INTEL" }
    };

    auto it = vendorMap.find(vendorID);
    return it != vendorMap.end() ? it->second : "Unknown Vendor";
}

static std::string getDeviceType(uint32_t deviceType)
{
    static const std::unordered_map<uint32_t, std::string> deviceTypeMap = {
        { VK_PHYSICAL_DEVICE_TYPE_OTHER, "Other" },
        { VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, "Integrated GPU" },
        { VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, "Discrete GPU" },
        { VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, "Virtual GPU" },
        { VK_PHYSICAL_DEVICE_TYPE_CPU, "CPU" }
    };

    auto it = deviceTypeMap.find(deviceType);
    return it != deviceTypeMap.end() ? it->second : "Unknown";
}

static std::string getVersionString(uint32_t version)
{
    return std::to_string(VK_VERSION_MAJOR(version)) + "."   //
           + std::to_string(VK_VERSION_MINOR(version)) + "." //
           + std::to_string(VK_VERSION_PATCH(version));
}

static void printPhysicalDeviceProperties(const VkPhysicalDeviceProperties& properties)
{
    std::cout << "Device Name: " << properties.deviceName << std::endl;
    std::cout << "Vendor: " << getVendorName(properties.vendorID) << " (0x" << std::hex << properties.vendorID << std::dec << ")" << std::endl;
    std::cout << "Device Type: " << getDeviceType(properties.deviceType) << std::endl;
    std::cout << "API Version: " << getVersionString(properties.apiVersion) << std::endl;
    std::cout << "Driver Version: " << getVersionString(properties.driverVersion) << std::endl;
}

VkResult VulkanContext::printGpus()
{
    uint32_t deviceCount = 0;
    VK_CHECK_RETURN(vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr));

    std::vector<VkPhysicalDevice> gpus(deviceCount);
    VK_CHECK_RETURN(vkEnumeratePhysicalDevices(mInstance, &deviceCount, gpus.data()));

    std::cout << "Available Vulkan GPUs: " << deviceCount << std::endl;

    uint32_t usedGpuIndex = 0;
    for (uint32_t d = 0; d < deviceCount; d++)
    {
        if (gpus[d] == mPhysicalDevice)
        {
            usedGpuIndex = d;
        }

        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(gpus[d], &properties);

        std::cout << " - " << d << ") " << properties.deviceName << std::endl;
    }

    if (mPhysicalDevice == VK_NULL_HANDLE)
    {
        std::cout << "No compatible GPU." << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::cout << "Using GPU " << usedGpuIndex << ":\n";

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(mPhysicalDevice, &properties);
    printPhysicalDeviceProperties(properties);

    return VK_SUCCESS;
}

VkResult VulkanContext::printQueues()
{
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(mPhysicalDevice, &queueFamilyCount, queueFamilies.data());

    std::cout << "Available Queue Families: " << queueFamilyCount << std::endl;

    for (uint32_t i = 0; i < queueFamilyCount; i++)
    {
        const auto flags = string_VkQueueFlags(queueFamilies[i].queueFlags);
        std::cout << " - " << i << ") " << queueFamilies[i].queueCount << " queues : " << flags << std::endl;
    }

    std::cout << "Used queues:\n";

    for (const auto& queue : mQueues)
    {
        std::cout << " - family: " << queue.familyIndex << ", index: " << queue.queueIndex << std::endl;
    }

    return VK_SUCCESS;
}
