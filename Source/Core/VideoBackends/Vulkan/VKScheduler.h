// Copyright 2022 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
// Adapted from the yuzu Emulator Project
// which in turn is adapted from DXVK

#pragma once

#include <cstddef>
#include <queue>

#include "CommandBufferManager.h"
#include "Common/Align.h"
#include "Common/Assert.h"

namespace Vulkan
{
class Scheduler
{
  class Command
  {
  public:
    virtual ~Command() = default;

    virtual void Execute(CommandBufferManager* cmdbuf) = 0;

    Command* GetNext() const { return next; }

    void SetNext(Command* next_) { next = next_; }

  private:
    Command* next = nullptr;
  };

  template <typename T>
  class alignas(16) TypedCommand final : public Command
  {
  public:
    explicit TypedCommand(T&& command_) : command{std::move(command_)} {}

    ~TypedCommand() override = default;

    TypedCommand(TypedCommand&&) = delete;

    TypedCommand& operator=(TypedCommand&&) = delete;

    void Execute(CommandBufferManager* cmdbuf) override { command(cmdbuf); }

  private:
    T command;
  };

  class CommandChunk final
  {
  public:
    void ExecuteAll(CommandBufferManager* cmdbuf);

    template <typename T>
    bool Record(T& command)
    {
      using FuncType = TypedCommand<T>;
      static_assert(sizeof(FuncType) < sizeof(data), "Lambda is too large");

      command_offset = Common::AlignUp(command_offset, alignof(FuncType));
      if (command_offset > sizeof(data) - sizeof(FuncType)) [[unlikely]]
      {
        return false;
      }
      Command* const current_last = last;
      last = new (data.data() + command_offset) FuncType(std::move(command));

      if (current_last) [[likely]]
      {
        current_last->SetNext(last);
      }
      else
      {
        first = last;
      }
      command_offset += sizeof(FuncType);
      return true;
    }

    bool Empty() const { return command_offset == 0; }

  private:
    Command* first = nullptr;
    Command* last = nullptr;

    size_t command_offset = 0;
    alignas(64) std::array<u8, 0x8000> data{};
  };

public:
  Scheduler();
  ~Scheduler();

  bool Initialize();

  void Flush();
  void SyncWorker();

  void Shutdown();

  template <typename T>
  void Record(T&& command)
  {
#ifdef VULKAN_DISABLE_THREADING
    command(m_command_buffer_manager.get());
    return;
#endif

    if (m_chunk->Record(command)) [[likely]]
      return;

    Flush();
    bool succeeded = m_chunk->Record(command);
    ASSERT(succeeded);
  }

  u64 GetCompletedFenceCounter() const
  {
    return m_command_buffer_manager->GetCompletedFenceCounter();
  }

  u64 GetCurrentFenceCounter() const
  {
    return m_current_fence_counter.load(std::memory_order_acquire);
  }

  void WaitForFenceCounter(u64 counter);
  void SynchronizeSubmissionThread();

  bool CheckLastPresentFail() { return m_command_buffer_manager->CheckLastPresentFail(); }
  VkResult GetLastPresentResult() const { return m_command_buffer_manager->GetLastPresentResult(); }
  bool CheckLastPresentDone() { return m_command_buffer_manager->CheckLastPresentDone(); }

  void SubmitCommandBuffer(bool submit_on_worker_thread, bool wait_for_completion,
                           VkSwapchainKHR present_swap_chain = VK_NULL_HANDLE,
                           uint32_t present_image_index = 0xFFFFFFFF);

private:
  void WorkerThread();
  void AcquireNewChunk();

  std::unique_ptr<CommandBufferManager> m_command_buffer_manager;

  std::unique_ptr<CommandChunk> m_chunk;

  std::thread m_worker;
  std::unique_ptr<Common::BlockingLoop> m_submit_loop;

  std::atomic<u64> m_current_fence_counter = 1;

  std::queue<std::unique_ptr<CommandChunk>> m_work_queue;
  std::mutex m_work_mutex;

  std::vector<std::unique_ptr<CommandChunk>> m_chunk_reserve;
  std::mutex m_reserve_mutex;
};

extern std::unique_ptr<Scheduler> g_scheduler;
}  // namespace Vulkan
