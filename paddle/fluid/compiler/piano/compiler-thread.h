/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_COMPILER_COMPILER_THREAD_H
#define PADDLE_COMPILER_COMPILER_THREAD_H

#include <memory>
#include <vector>

namespace paddle {
namespace piano {

class Arena;
class CompilerContext;

class CompilerThread {
  
 public:
  ~CompilerThread();

  static CompilerThread* Current() {
    return t_;
  }
  
  void Run();

  CompilerContext* SetCompilerContext(CompilerContext* cc) {
    CompilerContext* prev_cc = cc_;
    cc_ = cc;
    return prev_cc;
  }

  CompilerContext& GetCompilerContext() {
    return *cc_;
  }

 private:
  CompilerContext* cc_;
  static thread_local CompilerThread* t_;
};


class CompilerContext {
 public:
  CompilerContext(CompilerThread* thread,
                  bool is_jit) {
    prev_cc_ = thread->SetCompilerContext(this);
  }

  ~CompilerContext() {
    thread()->SetCompilerContext(prev_cc_);
  }

  CompilerThread* thread() { return t_; }
  Arena *arena() const { return arena_; }
 private:
  Arena *arena_ = nullptr;
  CompilerThread* t_;
  CompilerContext* prev_cc_;
};

class ThreadPool {
 public:
  class Task : public std::vector<Task> {
   protected:
    Task() {}

   public:
    virtual ~Task() {}
    virtual void Run() = 0;

   private:
    Task(const Task&);                                          \
    void operator=(const Task&);
  };

  explicit ThreadPool(int max_pool_size = 0);
  virtual ~ThreadPool();

  // Runs a task on the thread pool.
  template <typename T, typename... Args>
  bool Run(Args&&... args) {
    return RunImpl(std::unique_ptr<Task>(new T(std::forward<Args>(args)...)));
  }

 private:
  class Worker : public std::vector<Worker> {
   public:
    explicit Worker(ThreadPool* pool);
    void StartThread();

   private:
    friend class ThreadPool;
    static void Main(void* args);

    ThreadPool* pool_;
    CompilerThread* thread_;
    bool is_blocked_ = false;

    Worker(const Worker&);                                          \
    void operator=(const Worker&);
  };

 private:
  bool RunImpl(std::unique_ptr<Task> task);
  void WorkerLoop(Worker* worker);

  Worker* ScheduleTask(std::unique_ptr<Task> task);

  bool shutting_down_ = false;
  std::vector<Worker> running_workers_;
  std::vector<Worker> idle_workers_;
  std::vector<Task> tasks_;
  int max_pool_size_ = 0;

  ThreadPool(const ThreadPool&);                                          \
  void operator=(const ThreadPool&);
};

}
}

#endif // PADDLE_COMPILER_COMPILER_THREAD_H