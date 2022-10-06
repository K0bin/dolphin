// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/Fifo.h"

#include <atomic>
#include <cstring>

#include "Common/Assert.h"
#include "Common/BlockingLoop.h"
#include "Common/ChunkFile.h"
#include "Common/Event.h"
#include "Common/FPURoundMode.h"
#include "Common/MemoryUtil.h"
#include "Common/MsgHandler.h"

#include "Core/Config/MainSettings.h"
#include "Core/ConfigManager.h"
#include "Core/CoreTiming.h"
#include "Core/HW/GPFifo.h"
#include "Core/HW/Memmap.h"
#include "Core/Host.h"
#include "Core/System.h"

#include "VideoCommon/AsyncRequests.h"
#include "VideoCommon/CPMemory.h"
#include "VideoCommon/CommandProcessor.h"
#include "VideoCommon/DataReader.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/OpcodeDecoding.h"
#include "VideoCommon/VertexLoaderManager.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoBackendBase.h"

namespace Fifo
{
static constexpr u32 FIFO_SIZE = 2 * 1024 * 1024;
static constexpr int GPU_TIME_SLOT_SIZE = 1000;

static Common::BlockingLoop s_gpu_mainloop;

static Common::Flag s_emu_running_state;

static CoreTiming::EventType* s_event_sync_gpu;

// STATE_TO_SAVE
static u8* s_video_buffer;
static u8* s_video_buffer_read_ptr;
static std::atomic<u8*> s_video_buffer_write_ptr;

static std::atomic<int> s_sync_ticks;
static bool s_syncing_suspended;

static std::optional<size_t> s_config_callback_id = std::nullopt;
static float s_config_sync_gpu_overclock = 0.0f;

static void RefreshConfig()
{
  s_config_sync_gpu_overclock = Config::Get(Config::MAIN_SYNC_GPU_OVERCLOCK);
}

void DoState(PointerWrap& p)
{
  p.DoArray(s_video_buffer, FIFO_SIZE);
  u8* write_ptr = s_video_buffer_write_ptr;
  p.DoPointer(write_ptr, s_video_buffer);
  s_video_buffer_write_ptr = write_ptr;
  p.DoPointer(s_video_buffer_read_ptr, s_video_buffer);

  p.Do(s_sync_ticks);
  p.Do(s_syncing_suspended);
}

void PauseAndLock(bool doLock, bool unpauseOnUnlock)
{
  if (doLock)
  {
    EmulatorState(false);

    if (!Core::System::GetInstance().IsDualCoreMode())
      return;

    s_gpu_mainloop.WaitYield(std::chrono::milliseconds(100), Host_YieldToUI);
  }
  else
  {
    if (unpauseOnUnlock)
      EmulatorState(true);
  }
}

void Init()
{
  if (!s_config_callback_id)
    s_config_callback_id = Config::AddConfigChangedCallback(RefreshConfig);
  RefreshConfig();

  // Padded so that SIMD overreads in the vertex loader are safe
  s_video_buffer = static_cast<u8*>(Common::AllocateMemoryPages(FIFO_SIZE + 4));
  ResetVideoBuffer();
  if (Core::System::GetInstance().IsDualCoreMode())
    s_gpu_mainloop.Prepare();
  s_sync_ticks.store(0);
}

void Shutdown()
{
  if (s_gpu_mainloop.IsRunning())
    PanicAlertFmt("FIFO shutting down while active");

  Common::FreeMemoryPages(s_video_buffer, FIFO_SIZE + 4);
  s_video_buffer = nullptr;
  s_video_buffer_write_ptr = nullptr;
  s_video_buffer_read_ptr = nullptr;

  if (s_config_callback_id)
  {
    Config::RemoveConfigChangedCallback(*s_config_callback_id);
    s_config_callback_id = std::nullopt;
  }
}

// May be executed from any thread, even the graphics thread.
// Created to allow for self shutdown.
void ExitGpuLoop()
{
  // This should break the wait loop in CPU thread
  CommandProcessor::fifo.bFF_GPReadEnable.store(0, std::memory_order_relaxed);
  FlushGpu();

  // Terminate GPU thread loop
  s_emu_running_state.Set();
  s_gpu_mainloop.Stop(s_gpu_mainloop.kNonBlock);
}

void EmulatorState(bool running)
{
  s_emu_running_state.Set(running);
  if (running)
    s_gpu_mainloop.Wakeup();
  else
    s_gpu_mainloop.AllowSleep();
}

// Description: RunGpuLoop() sends data through this function.
static void ReadDataFromFifo(u32 readPtr)
{
  if (GPFifo::GATHER_PIPE_SIZE >
      static_cast<size_t>(s_video_buffer + FIFO_SIZE - s_video_buffer_write_ptr))
  {
    const size_t existing_len = s_video_buffer_write_ptr - s_video_buffer_read_ptr;
    if (GPFifo::GATHER_PIPE_SIZE > static_cast<size_t>(FIFO_SIZE - existing_len))
    {
      PanicAlertFmt("FIFO out of bounds (existing {} + new {} > {})", existing_len,
                    GPFifo::GATHER_PIPE_SIZE, FIFO_SIZE);
      return;
    }
    memmove(s_video_buffer, s_video_buffer_read_ptr, existing_len);
    s_video_buffer_write_ptr = s_video_buffer + existing_len;
    s_video_buffer_read_ptr = s_video_buffer;
  }
  // Copy new video instructions to s_video_buffer for future use in rendering the new picture
  Memory::CopyFromEmu(s_video_buffer_write_ptr, readPtr, GPFifo::GATHER_PIPE_SIZE);
  s_video_buffer_write_ptr += GPFifo::GATHER_PIPE_SIZE;
}

void ResetVideoBuffer()
{
  s_video_buffer_read_ptr = s_video_buffer;
  s_video_buffer_write_ptr = s_video_buffer;
}

// Description: Main FIFO update loop
// Purpose: Keep the Core HW updated about the CPU-GPU distance
void RunGpuLoop()
{
  AsyncRequests::GetInstance()->SetEnable(true);
  AsyncRequests::GetInstance()->SetPassthrough(false);

  s_gpu_mainloop.Run(
      [] {
        // Run events from the CPU thread.
        AsyncRequests::GetInstance()->PullEvents();

        // Do nothing while paused
        if (!s_emu_running_state.IsSet())
          return;

        while (g_fifo_thread.PopReadChunk())
        {
          u32 cycles = 0;
          OpcodeDecoder::RunFifo<false>(
                  g_fifo_thread.ReadChunk().FifoReader(), &cycles);
        }

        // The fifo is empty, and it's unlikely we will get any more work in the near future.
        // Make sure VertexManager finishes drawing any primitives it has stored in its buffer.
        g_vertex_manager->Flush();
        g_framebuffer_manager->RefreshPeekCache();
        AsyncRequests::GetInstance()->PullEvents();
      },
      100);

  AsyncRequests::GetInstance()->SetEnable(false);
  AsyncRequests::GetInstance()->SetPassthrough(true);
}

void FlushGpu()
{
  if (!Core::System::GetInstance().IsDualCoreMode())
    return;

  s_gpu_mainloop.Wait();
}

void GpuMaySleep()
{
  s_gpu_mainloop.AllowSleep();
}

bool AtBreakpoint()
{
  CommandProcessor::SCPFifoStruct& fifo = CommandProcessor::fifo;
  return fifo.bFF_BPEnable.load(std::memory_order_relaxed) &&
         (fifo.CPReadPointer.load(std::memory_order_relaxed) ==
          fifo.CPBreakpoint.load(std::memory_order_relaxed));
}

void RunGpu()
{
    s_syncing_suspended = false;
    CoreTiming::ScheduleEvent(GPU_TIME_SLOT_SIZE, s_event_sync_gpu, GPU_TIME_SLOT_SIZE);
}

static int RunGpuOnCpu(int ticks)
{
  CommandProcessor::SCPFifoStruct& fifo = CommandProcessor::fifo;
  bool reset_simd_state = false;
  int available_ticks = int(ticks * s_config_sync_gpu_overclock) + s_sync_ticks.load();
  while (fifo.bFF_GPReadEnable.load(std::memory_order_relaxed) &&
         fifo.CPReadWriteDistance.load(std::memory_order_relaxed) && !AtBreakpoint() &&
         available_ticks >= 0)
  {
    if (!reset_simd_state)
    {
      FPURoundMode::SaveSIMDState();
      FPURoundMode::LoadDefaultSIMDState();
      reset_simd_state = true;
    }
    ReadDataFromFifo(fifo.CPReadPointer.load(std::memory_order_relaxed));
    u32 cycles = 0;
    if (Core::System::GetInstance().IsDualCoreMode())
    {
      // Send FIFO to data to the worker and run preprocess
      u8* start_ptr = s_video_buffer_read_ptr;
      s_video_buffer_read_ptr = OpcodeDecoder::RunFifo<true>(
              DataReader(s_video_buffer_read_ptr, s_video_buffer_write_ptr), &cycles);
      g_fifo_thread.WriteChunk().PushFifoData(start_ptr, s_video_buffer_write_ptr - start_ptr);
      g_fifo_thread.Flush();
      s_gpu_mainloop.Wakeup();
    }
    else
    {
      s_video_buffer_read_ptr = OpcodeDecoder::RunFifo<false>(
              DataReader(s_video_buffer_read_ptr, s_video_buffer_write_ptr), &cycles);
    }
    available_ticks -= cycles;

    if (fifo.CPReadPointer.load(std::memory_order_relaxed) ==
        fifo.CPEnd.load(std::memory_order_relaxed))
    {
      fifo.CPReadPointer.store(fifo.CPBase.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
    }
    else
    {
      fifo.CPReadPointer.fetch_add(GPFifo::GATHER_PIPE_SIZE, std::memory_order_relaxed);
    }

    fifo.CPReadWriteDistance.fetch_sub(GPFifo::GATHER_PIPE_SIZE, std::memory_order_relaxed);
  }

  CommandProcessor::SetCPStatusFromGPU();

  if (reset_simd_state)
  {
    FPURoundMode::LoadSIMDState();
  }

  // Discard all available ticks as there is nothing to do any more.
  s_sync_ticks.store(std::min(available_ticks, 0));

  // If the GPU is idle, drop the handler.
  if (available_ticks >= 0)
    return -1;

  // Always wait at least for GPU_TIME_SLOT_SIZE cycles.
  return -available_ticks + GPU_TIME_SLOT_SIZE;
}

static void SyncGPUCallback(u64 ticks, s64 cyclesLate)
{
  ticks += cyclesLate;
  int next = RunGpuOnCpu((int)ticks);

  s_syncing_suspended = next < 0;
  if (!s_syncing_suspended)
    CoreTiming::ScheduleEvent(next, s_event_sync_gpu, next);
}

void SyncGPUForRegisterAccess()
{
  RunGpuOnCpu(GPU_TIME_SLOT_SIZE);
}

// Initialize GPU - CPU thread syncing, this gives us a deterministic way to start the GPU thread.
void Prepare()
{
  s_event_sync_gpu = CoreTiming::RegisterEvent("SyncGPUCallback", SyncGPUCallback);
  s_syncing_suspended = true;
}

void FifoThreadContext::Flush()
{
  // TODO: synchronize at some point, so we're not running too far ahead.

  if (m_write_chunk.IsEmpty())
    return;

  FifoChunk new_chunk;
  if (!m_reuse_queue.Pop(new_chunk)) {
    INCSTAT(g_stats.num_fifo_chunks);
  }

  new_chunk.Reset();
  FifoChunk submit_chunk = std::exchange(m_write_chunk, std::move(new_chunk));
  m_submit_queue.Push(std::move(submit_chunk));
}

FifoThreadContext g_fifo_thread;

}  // namespace Fifo
