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
#include "Common/Align.h"

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

void WakeGPU();
void RunGpu();
void RunGpuLoop();
void ExitGpuLoop();
void EmulatorState(bool running);
bool AtBreakpoint();
void ResetVideoBuffer();

struct FifoEntry {
    u32 start;
    u32 length;
};

// One FifoChunk is equivalent to one invocation of RunGPUonCPU
struct FifoChunk
{
    std::vector<u8> data;
    std::unordered_map<u32, u32> memory_offsets;
    std::vector<u8> aux_data;
    std::vector<FifoEntry> fifo_entries;
    u32 fifo_index = 0;
    bool pull_requests_before_execution = false;

    FifoChunk() = default;

    FifoChunk(const FifoChunk& other) = delete;

    FifoChunk(FifoChunk&& other) noexcept
    {
      data = std::move(other.data);
      memory_offsets = std::move(other.memory_offsets);
      aux_data = std::move(other.aux_data);
      fifo_entries = std::move(other.fifo_entries);
      fifo_index = other.fifo_index;
      pull_requests_before_execution = other.pull_requests_before_execution;
    }

    FifoChunk& operator = (FifoChunk&& other) noexcept
    {
      data = std::move(other.data);
      memory_offsets = std::move(other.memory_offsets);
      aux_data = std::move(other.aux_data);
      fifo_entries = std::move(other.fifo_entries);
      fifo_index = other.fifo_index;
      pull_requests_before_execution = other.pull_requests_before_execution;
      return *this;
    }

    void Reset()
    {
      data.clear();
      memory_offsets.clear();
      aux_data.clear();
      fifo_index = 0;
      pull_requests_before_execution = false;
    }

    void PushFifoData(const u8* src, u32 length)
    {
      ADDSTAT(g_stats.this_frame.fifo_data_copied, length);

      u32 offset = 0;
      if (!fifo_entries.empty())
      {
        const FifoEntry& last = fifo_entries.back();
        offset = Common::AlignUp(last.start + last.length + 4, 8);
      }

      fifo_entries.push_back(FifoEntry{static_cast<u32>(offset), length});

      // Pad for SIMD overreads
      u32 aligned_end = Common::AlignUp(offset + length + 4, 8);
      if (aligned_end > data.capacity()) [[unlikely]]
        data.resize(aligned_end);

      memcpy(data.data() + offset, src, length);
      memset(data.data() + offset + length, 0, aligned_end - offset + length);
    }

    void CopyAuxData(u32 guest_address, const u8* src, u32 length)
    {
      ADDSTAT(g_stats.this_frame.aux_data_copied, length);
      u32 offset = Common::AlignUp(aux_data.size(), 8);
      u32 aligned_end = Common::AlignUp(offset + length, 8);
      if (aligned_end > aux_data.capacity()) [[unlikely]]
        aux_data.resize(aligned_end);

      memcpy(aux_data.data() + offset, src, length);

      // zero the padding
      if (aligned_end - offset + length) [[likely]]
        memset(aux_data.data() + offset + length, 0, aligned_end - offset + length);

      memory_offsets.insert(std::make_pair(guest_address, offset));
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

    bool PullAsyncRequestsBefore() const
    {
      return pull_requests_before_execution;
    }

    void MarkNeedsAsyncRequestsPull()
    {
      pull_requests_before_execution = true;
    }

    DataReader NextFifoReader()
    {
      if (fifo_index >= fifo_entries.size())
        return DataReader(nullptr, nullptr);

      const FifoEntry& entry = fifo_entries[fifo_index++];
      return DataReader(data.data() + entry.start, data.data() + entry.start + entry.length);
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
