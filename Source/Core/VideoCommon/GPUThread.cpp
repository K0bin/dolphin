//
// Created by robin on 10/9/22.
//

#include "Common/BlockingLoop.h"
#include "Core/System.h"
#include "VideoCommon/AsyncRequests.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/FramebufferManager.h"
#include "Core/Host.h"
#include "GPUThread.h"
#include "AsyncRequests.h"

namespace GPUThread {

    static Common::BlockingLoop s_gpu_mainloop;
    
    static FifoThreadContext s_fifo_context;

    // Sync with the GPU thread
    static std::mutex s_gpu_frame_mutex;
    static std::condition_variable s_gpu_frame_condvar;
    static u64 s_cpu_frame_number = 0;
    static std::atomic<u64> s_gpu_frame_number = 0;

    void Init() {
      s_gpu_mainloop.Prepare();
    }

    void Exit() {
      // Terminate GPU thread loop
      s_gpu_mainloop.Stop(s_gpu_mainloop.kNonBlock);
    }

    bool IsActive() {
      return s_gpu_mainloop.IsRunning();
    }

    void FlushFifoChunk() {
      if (FifoWriteChunk().IsEmpty())
        return;

      if (s_cpu_frame_number > s_gpu_frame_number.load() + 1)
      {
        WARN_LOG_FMT(VIDEO, "Syncing! GPU Frame: {}, CPU frame: {}", s_gpu_frame_number.load(), s_cpu_frame_number);

        // TODO: this dead locks here

        // TODO: memory reads in BPStructs.cpp

        std::unique_lock lock(s_gpu_frame_mutex);
        s_gpu_frame_condvar.wait(lock, [&] { Wake(); return s_cpu_frame_number <= s_gpu_frame_number.load() + 1; });
      }

      AsyncRequests::Event e;
      e.type = AsyncRequests::Event::PROCESS_CHUNK;
      e.time = 0;
      new (&e.process_chunk.chunk) FifoChunk(std::move(FifoWriteChunk()));
      AsyncRequests::GetInstance()->PushEvent(std::move(e));
      s_fifo_context.PopWriteChunk();
    }

    void Wake()
    {
      s_gpu_mainloop.Wakeup();
    }

    void BumpGPUFrame() {
      WARN_LOG_FMT(VIDEO, "GPU Frame: {}", s_gpu_frame_number.load());
      std::lock_guard lock(s_gpu_frame_mutex);
      s_gpu_frame_number++;
      s_gpu_frame_condvar.notify_one();
    }

    void BumpCPUFrame()
    {
      WARN_LOG_FMT(VIDEO, "CPU Frame: {}", s_cpu_frame_number);
      s_cpu_frame_number++;
    }

    void ProcessGPUChunk(FifoChunk&& chunk) {
      s_fifo_context.PushReadChunk(std::move(chunk));
      u32 cycles = 0;
      DataReader reader;
      do {
        reader = s_fifo_context.ReadChunk().NextFifoReader();
        OpcodeDecoder::RunFifo<false>(
                reader, &cycles);
      } while (reader.size() != 0);
    }

    FifoChunk& FifoWriteChunk()
    {
      return s_fifo_context.WriteChunk();
    }

    FifoChunk& FifoReadChunk()
    {
      return s_fifo_context.ReadChunk();
    }

    // Description: Main FIFO update loop
    // Purpose: Keep the Core HW updated about the CPU-GPU distance
    void Run() {
      AsyncRequests::GetInstance()->SetPassthrough(false);

      s_gpu_mainloop.Run(
              [] {
                  while (!AsyncRequests::GetInstance()->IsQueueEmpty()) {
                    // Run events from the CPU thread.
                    AsyncRequests::GetInstance()->PullEvents();
                  }

                  // The fifo is empty, and it's unlikely we will get any more work in the near future.
                  // Make sure VertexManager finishes drawing any primitives it has stored in its buffer.
                  g_vertex_manager->Flush();
                  g_framebuffer_manager->RefreshPeekCache();
                  s_gpu_mainloop.AllowSleep();
              },
              100);

      AsyncRequests::GetInstance()->SetPassthrough(true);
    }

    void FifoThreadContext::PopWriteChunk() {
      FifoChunk new_chunk;
      {
        std::lock_guard free_list_lock(m_free_list_mutex);

        if (m_free_list.empty()) [[unlikely]] {
          std::lock_guard free_list_b_lock(m_free_list_mutex_b);
          std::swap(m_free_list, m_free_list_b);
        }

        if (!m_free_list.empty()) [[likely]] {
          new_chunk = std::move(m_free_list.back());
          m_free_list.pop_back();
        } else {
          INCSTAT(g_stats.num_fifo_chunks);
        }
      }

      new_chunk.Reset();
      m_write_chunk = std::move(new_chunk);
    }

    FifoChunk::~FifoChunk()
    {
      free(data);
      free(aux_data);
    }

    FifoChunk::FifoChunk(FifoChunk&& other) noexcept
    {
      data = other.data;
      other.data = nullptr;
      memory_offsets = std::move(other.memory_offsets);
      aux_data = other.aux_data;
      other.aux_data = nullptr;
      data_capacity = other.data_capacity;
      other.data_capacity = 0;
      aux_data_capacity = other.aux_data_capacity;
      other.aux_data_capacity = 0;
      fifo_entries = std::move(other.fifo_entries);
      fifo_index = other.fifo_index;
      other.fifo_index = 0;
      aux_data_length = other.aux_data_length;
      other.aux_data_length = 0;
    }

    FifoChunk& FifoChunk::operator=(FifoChunk &&other) noexcept
    {
      data = other.data;
      other.data = nullptr;
      memory_offsets = std::move(other.memory_offsets);
      aux_data = other.aux_data;
      other.aux_data = nullptr;
      data_capacity = other.data_capacity;
      other.data_capacity = 0;
      aux_data_capacity = other.aux_data_capacity;
      other.aux_data_capacity = 0;
      fifo_entries = std::move(other.fifo_entries);
      fifo_index = other.fifo_index;
      other.fifo_index = 0;
      aux_data_length = other.aux_data_length;
      other.aux_data_length = 0;
      return *this;
    }

    void FifoChunk::Reset()
    {
      memory_offsets.clear();
      fifo_entries.clear();
      fifo_index = 0;
      aux_data_length = 0;
    }

    void FifoChunk::PushFifoData(const u8 *src, u32 length)
    {
      ADDSTAT(g_stats.this_frame.fifo_data_copied, length);

      u32 offset = 0;
      if (!fifo_entries.empty()) {
        const FifoEntry &last = fifo_entries.back();
        offset = Common::AlignUp(last.start + last.length + 4, 8);
      }

      fifo_entries.push_back(FifoEntry{static_cast<u32>(offset), length});

      // Pad for SIMD overreads
      u32 aligned_end = Common::AlignUp(offset + length + 4, 8);
      if (aligned_end > data_capacity) [[unlikely]] {
        data_capacity = aligned_end;
        data = static_cast<u8 *>(realloc(data, data_capacity));
      }

      memcpy(data + offset, src, length);
      memset(data + offset + length, 0, aligned_end - (offset + length));
    }

    void FifoChunk::CopyAuxData(u32 guest_address, const u8 *src, u32 length)
    {
      ADDSTAT(g_stats.this_frame.aux_data_copied, length);
      u32 offset = Common::AlignUp(aux_data_length, 8);
      u32 aligned_end = Common::AlignUp(offset + length, 8);
      if (aligned_end > aux_data_capacity) [[unlikely]] {
        aux_data_capacity = aligned_end;
        aux_data = static_cast<u8 *>(realloc(aux_data, aux_data_capacity));
      }

      memcpy(aux_data + offset, src, length);

      // zero the padding
      if (aligned_end - offset + length)
        [[likely]]
                memset(aux_data + offset + length, 0, aligned_end - (offset + length));

      memory_offsets.insert(std::make_pair(guest_address, offset));
      aux_data_length = aligned_end;
    }

    u8* FifoChunk::AuxData(u32 guest_address)
    {
      auto iter = memory_offsets.find(guest_address);
      if (iter == memory_offsets.end()) [[unlikely]]
        return nullptr;

      DEBUG_ASSERT(aux_data_length > iter->second);

      return aux_data + iter->second;
    }

    void FifoThreadContext::PushReadChunk(FifoChunk&& chunk)
    {
      if (!m_read_chunk.IsEmpty()) [[likely]]
      {
        std::lock_guard lock(m_free_list_mutex_b);
        m_free_list_b.push_back(std::move(m_read_chunk));
      }
      m_read_chunk = std::move(chunk);
    }
}