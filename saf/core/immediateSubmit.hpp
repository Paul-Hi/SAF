/**
 * @file      immediateSubmit.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2023
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef IMMEDIATE_SUBMIT_HPP
#define IMMEDIATE_SUBMIT_HPP

namespace saf
{
    /// @brief Class to immediatly execute any vulkan command.
    /// @details Inserts commands into command buffer and flushes it directly.
    class ImmediateSubmit
    {
    public:
        /**
         * @brief Executes any function including vulkan commands immediately.
         * @param[in] logicalDevice The VkDevice.
         * @param[in] queue The VkQueue.
         * @param[in] commandPool The VkCommandPool.
         * @param[in] commandBuffer The VkCommandBuffer.
         * @param[in] immediateFunction A function pointer to the commands to execute immediately.
         */
        static void execute(VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, const std::function<void(VkCommandBuffer)> &immediateFunction)
        {
            VkResult err = vkResetCommandPool(logicalDevice, commandPool, 0);
            checkVkResult(err);
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            err = vkBeginCommandBuffer(commandBuffer, &beginInfo);
            checkVkResult(err);

            immediateFunction(commandBuffer);

            VkSubmitInfo endInfo = {};
            endInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            endInfo.commandBufferCount = 1;
            endInfo.pCommandBuffers = &commandBuffer;
            err = vkEndCommandBuffer(commandBuffer);
            checkVkResult(err);
            err = vkQueueSubmit(queue, 1, &endInfo, VK_NULL_HANDLE);
            checkVkResult(err);

            err = vkDeviceWaitIdle(logicalDevice);
            checkVkResult(err);
        }
    };

} // namespace saf

#endif // IMMEDIATE_SUBMIT_HPP
