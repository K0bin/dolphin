// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include "Common/CommonTypes.h"
#include "Common/SPSCQueue.h"
#include "Core/HW/GPFifo.h"

class PointerWrap;

namespace Fifo
{
void Init();
void Shutdown();
void Prepare();  // Must be called from the CPU thread.
void DoState(PointerWrap& f);
void PauseAndLock(bool doLock, bool unpauseOnUnlock);

// Used for diagnostics.
enum class SyncGPUReason
{
  Other,
  Wraparound,
  EFBPoke,
  PerfQuery,
  BBox,
  Swap,
  AuxSpace,
};

// In single core mode, this runs the GPU for a single slice.
// In dual core mode, this synchronizes with the GPU thread.
void SyncGPUForRegisterAccess();

void PushFifoAuxBuffer(const void* ptr, size_t size);
void* PopFifoAuxBuffer(size_t size);

void FlushGpu();
void RunGpu();
void GpuMaySleep();
void RunGpuLoop();
void ExitGpuLoop();
void EmulatorState(bool running);
bool AtBreakpoint();
void ResetVideoBuffer();

struct FifoChunk
{
    std::vector<u8> data;
    std::unordered_map<u32, u32> display_list_offsets;

    FifoChunk() = default;

    FifoChunk(const FifoChunk& other) = delete;

    FifoChunk(FifoChunk&& other) noexcept
    {
      data = std::move(other.data);
      display_list_offsets = std::move(other.display_list_offsets);
    }

    FifoChunk& operator = (FifoChunk&& other) noexcept
    {
      data = std::move(other.data);
      display_list_offsets = std::move(other.display_list_offsets);
      return *this;
    }

    void Reset()
    {
      data.clear();
      display_list_offsets.clear();
    }

    void CopyFrom(const u8* src, u32 length)
    {
      u32 additional_capacity = length;
      // Pad for SIMD overreads
      additional_capacity += 4;

      if (data.capacity() >= data.size() + additional_capacity)
        return;

      if (data.empty())
        additional_capacity = std::max(GPFifo::GATHER_PIPE_SIZE * 32, additional_capacity);

      data.reserve(additional_capacity);

      u32 old_size = data.size();
      data.resize(data.size() + length);
      memcpy(data.data() + old_size, src, length);
    }

    bool IsEmpty() const
    {
      return data.empty();
    }

    u32 Length() const
    {
      return data.size();
    }

    u8* Ptr()
    {
      return data.data();
    }
};

class FifoThread
{
public:
  void Flush();
  void CopyFrom(const u8* src, u32 length);

private:
  FifoChunk m_write_chunk;

  Common::SPSCQueue<FifoChunk, false> m_submit_queue;
  Common::SPSCQueue<FifoChunk, false> m_reuse_queue;
};

}  // namespace Fifo
