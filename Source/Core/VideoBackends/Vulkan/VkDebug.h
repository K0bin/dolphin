// Copyright 2016 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string_view>

#include "Common/CommonTypes.h"
#include "VideoBackends/Vulkan/Constants.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/Constants.h"
#include "VideoBackends/Vulkan/StagingBuffer.h"

#define EMIT_VK_DEBUG_MARKER(CMD, AUX0, AUX1, AUX2, AUX3) \
  if (g_command_buffer_mgr->GetDebug().IsEnabled()) [[unlikely]] \
    g_command_buffer_mgr->GetDebug().EmitMarker(CMD, AUX0, AUX1, AUX2, AUX3);

namespace Vulkan
{
enum class VkDebugCommand {
    Draw,
    DrawIndexed,
    Dispatch,
    BoundingBoxRead,
    BoundingBoxWrite,
    DisableQuery,
    ResetQuery,
    ReadbackQuery,
    ClearRegion,
    TransitionToLayout,
    TransitionToComputeLayout,
    StagingCopyFromTexture,
    StagingCopyToTexture,
    StagingCopyRectFromTexture,
};

struct VkDebugMarker
{
    u64 sequence_number;
    VkDebugCommand cmd;
    std::array<u64, 4> aux;
};

class VkDebug
{
public:
    VkDebug();
    void EmitMarker(VkDebugCommand cmd, u64 aux0, u64 aux1, u64 aux2, u64 aux3);
    void EnsureStagingBufferCapacity();
    void Reset();
    const VkDebugMarker* FindFault();
    void PrintFault();
    bool IsEnabled() const {
        return m_enabled;
    }

private:
    bool m_enabled;
    u64 m_sequence_number = 0;
    u64 m_staging_buffer_index = 0;
    u64 m_staging_buffer_offset = 0;
    std::vector<std::unique_ptr<StagingBuffer>> m_buffers;
    std::vector<VkDebugMarker> m_markers;
    std::unique_ptr<StagingBuffer> m_readback_buffer;
};
}
