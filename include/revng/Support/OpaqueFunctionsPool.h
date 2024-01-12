#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ModRef.h"

#include "revng/ADT/Concepts.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

template<typename T>
concept PointerToLLVMTypeOrDerived = std::derived_from<std::remove_pointer_t<T>,
                                                       llvm::Type>;

template<typename... KeyTypes>
class OpaqueFunctionsPool {
private:
  llvm::Module &M;
  const bool PurgeOnDestruction;
  std::map<std::tuple<KeyTypes...>, llvm::Function *> Pool;
  llvm::AttributeList AttributeSets;
  llvm::MemoryEffects MemoryEffects = llvm::MemoryEffects::none();
  FunctionTags::TagsSet Tags;

public:
  OpaqueFunctionsPool(llvm::Module *M, bool PurgeOnDestruction) :
    M(*M), PurgeOnDestruction(PurgeOnDestruction) {}

  ~OpaqueFunctionsPool() {
    if (PurgeOnDestruction) {
      for (auto &[Key, F] : Pool) {
        revng_assert(F->use_begin() == F->use_end());
        eraseFromParent(F);
      }
    }
  }

public:
  void addFnAttribute(llvm::Attribute::AttrKind Kind) {
    using namespace llvm;
    AttributeSets = AttributeSets.addFnAttribute(M.getContext(), Kind);
  }

  void setMemoryEffects(const llvm::MemoryEffects &NewMemoryEffects) {
    MemoryEffects = NewMemoryEffects;
  }

  void setTags(const FunctionTags::TagsSet &Tags) { this->Tags = Tags; }

public:
  auto begin() const { return Pool.begin(); }
  auto end() const { return Pool.end(); }

public:
  void record(KeyT Key, llvm::Function *F) {
    auto It = Pool.find(Key);
    if (It == Pool.end())
      Pool[Key] = F;
    else
      revng_assert(It->second == F);
  }

public:
  llvm::Function *get(std::tuple<KeyTypes...> Key,
                      llvm::FunctionType *FT,
                      const llvm::Twine &Name = {}) {
    using namespace llvm;

    Function *F = nullptr;
    auto It = Pool.find(Key);
    if (It != Pool.end()) {
      F = It->second;
    } else {
      F = Function::Create(FT, GlobalValue::ExternalLinkage, Name, &M);
      F->setAttributes(AttributeSets);
      F->setMemoryEffects(MemoryEffects);
      Tags.set(F);
      Pool.insert(It, { Key, F });
    }

    // Ensure the function we're returning is as expected
    revng_assert(F->getFunctionType() == FT);

    return F;
  }

  llvm::Function *get(std::tuple<KeyTypes...> Key,
                      llvm::Type *ReturnType = nullptr,
                      llvm::ArrayRef<llvm::Type *> Arguments = {},
                      const llvm::Twine &Name = {}) {
    using namespace llvm;
    if (ReturnType == nullptr)
      ReturnType = Type::getVoidTy(M.getContext());

    return get(Key, FunctionType::get(ReturnType, Arguments, false), Name);
  }

private:
  template<int N, typename... Types>
  using NthType = std::tuple_element<N, std::tuple<Types...>>::type;

  template<std::size_t N>
  using NthTypeLikeInKey = std::remove_pointer_t<NthType<N, KeyTypes...>>;

  template<bool ReturnIsKey,
           std::size_t... ArgumentIndicesInFunctionPrototype,
           std::size_t... Indices>
  static void
  initializeArgumentTypesInKey(llvm::Function &F,
                               std::tuple<KeyTypes...> &Key,
                               const std::index_sequence<Indices...> &) {

    const auto AssignArgument =
      [&Key, &F]<std::size_t ArgumentIndexInKey,
                 std::size_t ArgumentIndexInFunctionType>() {
        using TypeLike = NthTypeLikeInKey<ArgumentIndexInKey>;
        std::get<ArgumentIndexInKey>(Key) = llvm::cast<
          TypeLike>(F.getArg(ArgumentIndexInFunctionType)->getType());
      };

    constexpr std::size_t InitialIndexForArgumentsInKey = ReturnIsKey ? 1 : 0;
    ((AssignArgument
        .template operator()<InitialIndexForArgumentsInKey + Indices,
                             ArgumentIndicesInFunctionPrototype>()),
     ...);
  }

public:
  /// Initialize the pool with all the functions in M that match the tag TheTag,
  /// using the return type and/or the argument types as key, according to the
  /// what specified in the template parameters.
  template<bool ReturnTypeIsKey, std::size_t... IndicesOfKeyArgumentTypes>
  void initializeFromType(const FunctionTags::Tag &TheTag)
    requires((std::derived_from<std::remove_pointer_t<KeyTypes>, llvm::Type>)
             and ...)
  {
    for (llvm::Function &F : TheTag.functions(&M)) {

      std::tuple<KeyTypes...> Key;

      if constexpr (ReturnTypeIsKey) {
        auto *RetType = F.getFunctionType()->getReturnType();
        using TypeLike = std::remove_pointer_t<NthType<0, KeyTypes...>>;
        std::get<0>(Key) = cast<TypeLike>(RetType);
      }

      auto PackIndices = std::make_index_sequence<
        sizeof...(IndicesOfKeyArgumentTypes)>();
      if constexpr (PackIndices.size() != 0) {
        initializeArgumentTypesInKey<ReturnTypeIsKey,
                                     IndicesOfKeyArgumentTypes...>(F,
                                                                   Key,
                                                                   PackIndices);
      }

      recordUnchecked(Key, &F);
    }
  }

  /// Initialize the pool with all the functions in M that match the tag TheTag,
  /// using the return type as key.
  void initializeFromReturnType(const FunctionTags::Tag &TheTag) {
    return initializeFromType<true>(TheTag);
  }

  /// Initialize the pool with all the functions in M that match the tag TheTag,
  /// using the type of the Nth argument as key.
  template<std::size_t N>
  void initializeFromNthArgType(const FunctionTags::Tag &TheTag) {
    return initializeFromType<false, N>(TheTag);
  }

  /// Initialize the pool with all the functions in M that match the tag TheTag,
  /// using the type of the ArgNo-th argument as key.
  void initializeFromName(const FunctionTags::Tag &TheTag)
    requires(std::tuple_size_v<std::tuple<KeyTypes...>> > 0
             and std::is_same_v<NthType<0, KeyTypes...>, std::string>)
  {
      record(F.getName().str(), &F);
    for (llvm::Function &F : TheTag.functions(&M))
  }
};
