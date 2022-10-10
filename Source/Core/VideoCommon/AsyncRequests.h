// Copyright 2015 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#include "Common/CommonTypes.h"
#include "Common/Flag.h"

#include "VideoCommon/GPUThread.h"

struct EfbPokeData;
class PointerWrap;

class AsyncRequests
{
public:
  struct Event
  {
    Event() {
      // anything but PROCESS_CHUNK
      type = SYNC_EVENT;
    }
    ~Event() {
      if (type == PROCESS_CHUNK)
        process_chunk.chunk.~FifoChunk();
    }

    Event(Event&& other) noexcept
    {
      time = other.time;

      switch (other.type)
      {
        case SYNC_EVENT:
          break;
        case PROCESS_CHUNK:
            new (&process_chunk.chunk) GPUThread::FifoChunk(std::move(other.process_chunk.chunk));
          break;
        case EFB_PEEK_COLOR:
        case EFB_PEEK_Z:
          efb_peek = other.efb_peek;
          break;
        case EFB_POKE_COLOR:
        case EFB_POKE_Z:
          efb_poke = other.efb_poke;
          break;
        case SWAP_EVENT:
          swap_event = other.swap_event;
          break;
        case BBOX_READ:
          bbox = other.bbox;
          break;
        case FIFO_RESET:
          fifo_reset = other.fifo_reset;
          break;
        case PERF_QUERY:
          perf_query = other.perf_query;
          break;
        case DO_SAVE_STATE:
          do_save_state = other.do_save_state;
          break;
      }
      type = other.type;
    }

    Event& operator=(Event &&other) noexcept
    {
      time = other.time;

      switch (other.type)
      {
        case SYNC_EVENT:
          break;
        case PROCESS_CHUNK:
          if (type == PROCESS_CHUNK)
            process_chunk.chunk = std::move(other.process_chunk.chunk);
          else
            new (&process_chunk.chunk) GPUThread::FifoChunk(std::move(other.process_chunk.chunk));
          break;
        case EFB_PEEK_COLOR:
        case EFB_PEEK_Z:
          efb_peek = other.efb_peek;
          break;
        case EFB_POKE_COLOR:
        case EFB_POKE_Z:
          efb_poke = other.efb_poke;
          break;
        case SWAP_EVENT:
          swap_event = other.swap_event;
          break;
        case BBOX_READ:
          bbox = other.bbox;
          break;
        case FIFO_RESET:
          fifo_reset = other.fifo_reset;
          break;
        case PERF_QUERY:
          perf_query = other.perf_query;
          break;
        case DO_SAVE_STATE:
          do_save_state = other.do_save_state;
          break;
      }
      type = other.type;

      return *this;
    }

    enum Type
    {
      EFB_POKE_COLOR,
      EFB_POKE_Z,
      EFB_PEEK_COLOR,
      EFB_PEEK_Z,
      SWAP_EVENT,
      BBOX_READ,
      FIFO_RESET,
      PERF_QUERY,
      DO_SAVE_STATE,
      SYNC_EVENT,
      PROCESS_CHUNK,
    } type;
    u64 time;

    union
    {
      struct
      {
        u16 x;
        u16 y;
        u32 data;
      } efb_poke;

      struct
      {
        u16 x;
        u16 y;
        u32* data;
      } efb_peek;

      struct
      {
        u32 xfbAddr;
        u32 fbWidth;
        u32 fbStride;
        u32 fbHeight;
      } swap_event;

      struct
      {
        int index;
        u16* data;
      } bbox;

      struct
      {
      } fifo_reset;

      struct
      {
      } perf_query;

      struct
      {
      } sync;

      struct
      {
          GPUThread::FifoChunk chunk;
      } process_chunk;

      struct
      {
        PointerWrap* p;
      } do_save_state;
    };
  };

  AsyncRequests();

  void PullEvents()
  {
    if (!m_empty.IsSet())
      PullEventsInternal();
  }
  void PushEvent(Event&& event, bool blocking = false);
  void WaitForEmptyQueue();
  void SetPassthrough(bool enable);
  bool IsQueueEmpty();

  static AsyncRequests* GetInstance() { return &s_singleton; }

private:
  void PullEventsInternal();
  void HandleEvent(Event&& e);

  static AsyncRequests s_singleton;

  Common::Flag m_empty;
  std::queue<Event> m_queue;
  std::mutex m_mutex;
  std::condition_variable m_cond;

  bool m_wake_me_up_again = false;
  bool m_enable = false;
  bool m_passthrough = true;

  std::vector<EfbPokeData> m_merged_efb_pokes;
};
