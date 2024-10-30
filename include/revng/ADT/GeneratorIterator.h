#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstddef>
#include <type_traits>

#include "revng/Support/Debug.h"
#include "revng/Support/Generator.h"
#include "revng/Support/IRHelpers.h"

// Define a custom concept for restricting the T type
template<typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>
                            and std::is_copy_constructible_v<T>
                            and std::is_copy_assignable_v<T>;

// We restrict the `T` type so that is a trivially copyable object
template<TriviallyCopyable T>
class GeneratorIterator {
public:
  // We need the specification of the following `iterator_traits` so that
  // `llvm::GraphWriter` doesn't complain in trying to perform `std::distance`
  // over the `child_begin`/`child_end` traits
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  using iterator_category = typename std::forward_iterator_tag;

public:
  using inner_iterator = cppcoro::generator<T>::iterator;

public:
  bool IsEnd = false;
  mutable bool IsDead = false;
  mutable cppcoro::generator<T> Coroutine;
  inner_iterator Begin;
  inner_iterator End;

private:
  // The `SnapshotContent` `std::optional` field is used when constructing a
  // snapshotted iterator, i.e., a special iterator state which can only be
  // dereferenced (this is used in order to provide a post-increment operator,
  // which would need to perform a copy of the main iterator object, operation
  // that would violate the unicity of the ownership of the `Coroutine` object)
  std::optional<T> SnapshotContent;

public:
  // Constructor for building a end sentinel iterator
  GeneratorIterator() : IsEnd(true) {}

  // Standard constructor providing an input `Coroutine`, which is used during
  // the increment of the iterator
  GeneratorIterator(cppcoro::generator<T> &&Coroutine) :
    IsEnd(false),
    Coroutine(std::move(Coroutine)),
    Begin(this->Coroutine.begin()),
    End(this->Coroutine.end()) {}

private:
  // Constructor used for a snapshotted iterator, which should only be invoked
  // in the post increment, so this is private
  GeneratorIterator(T &Snapshot) : SnapshotContent(Snapshot) {}

public:
  // Copy constructor which uses the copy assignment
  GeneratorIterator(const GeneratorIterator &Other) { *this = Other; }

  // Copy assignment. The semantics of this is non-trivial and appear counter
  // intuitive. We are constrained by the existence of the private `Coroutine`
  // field, which owns the state of the iteration progression, and that cannot
  // be copied. However, due to the behavior of the llvm `iterator_range` class,
  // we need to perform a copy of the iterator, a copy in which only the LHS of
  // the result survives (the RHS is an immediately discarded temporary).
  // Therefore we implement a copy constructor that takes into consideration
  // this behavior, and which leaves the RHS in a `IsDead` state, which is
  // asserted as first thing in any method of the `GeneratorIterator`, in order
  // to ensure that it is never used.
  GeneratorIterator &operator=(const GeneratorIterator &Other) {
    IsEnd = Other.IsEnd;
    Other.IsDead = true;
    Coroutine = std::move(Other.Coroutine);
    Begin = Other.Begin;
    End = Other.End;
    return *this;
  }

  // We rely on the default copy construct and assignment

public:
  bool operator==(const GeneratorIterator &Other) const {
    // We should not invoke any operation on an iterator instance left in the
    // `IsDead` state by a copy operation
    revng_assert(not IsDead);
    revng_assert(not Other.IsDead);

    // Verify that we are not in the `Snapshot` state, before attempting a
    // comparison
    revng_assert(not SnapshotContent.has_value());

    // In the comparison operator, we need to explicitly handle the sentinel
    // iterator state, both for the situation where `this` is a sentinel
    // iterator, and when `Other` is a sentinel iterator. If both the iterator
    // objects are not sentinel, we compare the internal iterators
    if (IsEnd)
      return Other.Begin == Other.End;
    if (Other.IsEnd)
      return Begin == End;
    else
      return Begin == Other.Begin;
  }

  GeneratorIterator &operator++() {
    // We should not invoke any operation on an iterator instance left in the
    // `IsDead` state by a copy operation
    revng_assert(not IsDead);

    // Verify that we are not in the `Snapshot` state
    revng_assert(not SnapshotContent.has_value());

    ++Begin;
    return *this;
  }

  GeneratorIterator operator++(int) {
    // We should not invoke any operation on an iterator instance left in the
    // `IsDead` state by a copy operation
    revng_assert(not IsDead);

    // Verify that we are not in the `Snapshot` state
    revng_assert(not SnapshotContent.has_value());

    // We return an iterator in the `Snapshot` state, so it can be dereferenced
    // and T obtained, as required by the contract of the postincrement operator
    // for a forward iterator, but we do not perform a full copy that would
    // create problems due to the uniqueness of the internal `Coroutine`
    GeneratorIterator Ret(*Begin);
    ++*this;
    return Ret;
  }

  T operator*() const {
    // We should not invoke any operation on an iterator instance left in the
    // `IsDead` state by a copy operation
    revng_assert(not IsDead);

    if (SnapshotContent.has_value()) {
      return *SnapshotContent;
    }

    return *Begin;
  }
};
