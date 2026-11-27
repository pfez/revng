#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>
#include <cstdint>
#include <iterator>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

/// Describes the set of features supported by a dialect of C.
/// \note Each option must also be listed in `CDialectOptions`.
struct CDialect {
  /// Conversions between incompatible pointer types (e.g. int and float) are
  /// considered implicit.
  bool ImplicitIncompatiblePointerConversions = false;

  /// Conversions between pointers to integer types with different signedness
  /// (e.g. signed int and unsigned int) are considered implicit
  bool ImplicitPointerSignConversions = false;

  /// Conversions between pointer types where qualifiers are discarded are
  /// considered implicit.
  bool ImplicitQualifierDiscardingConversions = false;

  /// Conversions from pointers-to-void to pointers-to-object-type are
  /// considered implicit.
  bool ImplicitVoidPointerConversions = false;

  /// Returns the strictest dialect, where no option is enabled. This is the
  /// dialect assumed by Clift when a module does not specify one.
  [[nodiscard]] static constexpr CDialect strict() { return {}; }

  /// Returns the most permissive dialect, where every option is enabled
  [[nodiscard]] static constexpr CDialect permissive();

  [[nodiscard]] friend bool operator==(const CDialect &,
                                       const CDialect &) = default;
};

/// Describes an option of `CDialect`, so that code handling every option in
/// the same way (hashing, MLIR syntax, pipe configuration) can iterate over
/// `CDialectOptions` instead of naming each option.
struct CDialectOption {
  /// The name of the option in the MLIR syntax of `CDialectAttr`
  llvm::StringLiteral Name;

  /// The key of the option in the configuration of the pipeline
  llvm::StringLiteral ConfigurationKey;

  bool CDialect::*Member = nullptr;
};

inline constexpr CDialectOption CDialectOptions[] = {
  { "ImplicitIncompatiblePointerConversions",
    "implicit-incompatible-pointer-conversions",
    &CDialect::ImplicitIncompatiblePointerConversions },
  { "ImplicitPointerSignConversions",
    "implicit-pointer-sign-conversions",
    &CDialect::ImplicitPointerSignConversions },
  { "ImplicitQualifierDiscardingConversions",
    "implicit-qualifier-discarding-conversions",
    &CDialect::ImplicitQualifierDiscardingConversions },
  { "ImplicitVoidPointerConversions",
    "implicit-void-pointer-conversions",
    &CDialect::ImplicitVoidPointerConversions },
};

// Every field of CDialect must be listed in CDialectOptions exactly once
static_assert(sizeof(CDialect) == std::size(CDialectOptions) * sizeof(bool));
static_assert([] {
  for (size_t I = 0; I < std::size(CDialectOptions); ++I)
    for (size_t J = I + 1; J < std::size(CDialectOptions); ++J)
      if (CDialectOptions[I].Member == CDialectOptions[J].Member)
        return false;
  return true;
}());

constexpr CDialect CDialect::permissive() {
  CDialect Result;
  for (const CDialectOption &Option : CDialectOptions)
    Result.*Option.Member = true;
  return Result;
}

[[nodiscard]] inline llvm::hash_code hash_value(const CDialect &Dialect) {
  uint64_t EnabledOptions = 0;
  for (auto [Index, Option] : llvm::enumerate(CDialectOptions))
    EnabledOptions |= uint64_t(Dialect.*Option.Member) << Index;
  return llvm::hash_value(EnabledOptions);
}
