#ifndef PADDLE_COMPILER_COMPILER_THREAD_H
#define PADDLE_COMPILER_COMPILER_THREAD_H

#include <memory>
#include <vector>

namespace paddle {
namespace piano {

class Arena;
class CompilerThread;

class CompilerState {
 public:
  CompilerState(CompilerThread* thread,
                bool is_jit) {
    previous_ = thread->SetCompilerState(this);
  }

  ~CompilerState() {
    thread()->SetCompilerState(previous_);
  }

  CompilerThread* thread() { return thread_; }
 private:
  CompilerThread* thread_;
  CompilerState* previous_;
};


class CompilerThread {
  
 public:
  ~CompilerThread();

  static CompilerThread* Current() {
    return current_thread_;
  }
  
  void Run();

  CompilerState* SetCompilerState(CompilerState* state) {
    CompilerState* previous = compiler_state_;
    compiler_state_ = state;
    return previous;
  }

  CompilerState& compiler_state() {
    return *compiler_state_;
  }

  Arena *arena() const { return arena_; }
 private:
  Arena *arena_ = nullptr;
  CompilerState* compiler_state_;
  static thread_local CompilerThread* current_thread_;
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