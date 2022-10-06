// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <vector>
#include <unordered_map>
#include "Common/Assert.h"
#include "Common/CommonTypes.h"
#include "Common/SPSCQueue.h"
#include "Core/HW/GPFifo.h"
#include "VideoCommon/DataReader.h"
#include "Statistics.h"

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
    std::unordered_map<u32, u32> memory_offsets;
    std::vector<u8> aux_data;

    FifoChunk() = default;

    FifoChunk(const FifoChunk& other) = delete;

    FifoChunk(FifoChunk&& other) noexcept
    {
      data = std::move(other.data);
      memory_offsets = std::move(other.memory_offsets);
      aux_data = std::move(other.aux_data);
    }

    FifoChunk& operator = (FifoChunk&& other) noexcept
    {
      data = std::move(other.data);
      memory_offsets = std::move(other.memory_offsets);
      aux_data = std::move(other.aux_data);
      return *this;
    }

    void Reset()
    {
      data.clear();
      memory_offsets.clear();
      aux_data.clear();
    }

    void PushFifoData(const u8* src, u32 length)
    {
      ADDSTAT(g_stats.this_frame.fifo_data_copied, length);
      DEBUG_ASSERT(data.empty());

      // Pad for SIMD overreads
      data.resize(length + 4);
      data.resize(length);
      memcpy(data.data(), src, length);
    }

    void CopyAuxData(u32 guest_address, const u8* src, u32 length)
    {
      ADDSTAT(g_stats.this_frame.aux_data_copied, length);
      u32 old_size = aux_data.size();
      aux_data.resize(aux_data.size() + length);
      memcpy(aux_data.data() + old_size, src, length);
      memory_offsets.insert(std::make_pair(guest_address, old_size));
    }

    u8* AuxData(u32 guest_address)
    {
      auto iter = memory_offsets.find(guest_address);
      if (iter == memory_offsets.end()) [[unlikely]]
        return nullptr;

      DEBUG_ASSERT(aux_data.size() > iter->second);

      return aux_data.data() + iter->second;
    }

    bool IsEmpty() const
    {
      return data.empty();
    }

    DataReader FifoReader()
    {
      return DataReader(data.data(), data.data() + data.size());
    }
};

class FifoThreadContext
{
public:
  void Flush();

  bool IsEmpty() const
  {
    return m_submit_queue.Empty();
  }

  bool PopReadChunk()
  {
    if (!m_read_chunk.IsEmpty()) [[likely]]
      m_reuse_queue.Push(std::move(m_read_chunk));

    return m_submit_queue.Pop(m_read_chunk);
  }

  FifoChunk& WriteChunk()
  {
    return m_write_chunk;
  }

  FifoChunk& ReadChunk()
  {
    return m_read_chunk;
  }

private:
  FifoChunk m_write_chunk;
  FifoChunk m_read_chunk;

  Common::SPSCQueue<FifoChunk, true> m_submit_queue;
  Common::SPSCQueue<FifoChunk, true> m_reuse_queue;
};

extern FifoThreadContext g_fifo_thread;

}  // namespace Fifo
