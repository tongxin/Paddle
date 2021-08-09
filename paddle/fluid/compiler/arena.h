#ifndef PADDLE_COMPILER_ARENA_H
#define PADDLE_COMPILER_ARENA_H

class Arena
{
 private:
  Arena();
  ~Arena();

 public:
  template<typename DataType>
  inline DataType* Alloc(const size_t len);

  template <class DataType>
  inline DataType* Realloc(DataType* old_array,
                           const size_t old_len,
                           const size_t new_len);
};

#endif // PADDLE_COMPILER_ARENA_H