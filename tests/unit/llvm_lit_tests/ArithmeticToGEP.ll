;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -arithmetic-to-gep -S -o - | FileCheck %s

; CHECK-LABEL: define i64 @a
; CHECK: [[GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 1
; CHECK: [[CAST:%[a-zA-Z0-9]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK: ret i64 [[CAST]]
define i64 @a (ptr %arg) {
  %intptr = ptrtoint ptr %arg to i64
  %with_offset = add i64 %intptr, 1
  ret i64 %with_offset
}

; CHECK-LABEL: define i64 @b
; CHECK: [[GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 2
; CHECK: [[CAST:%[a-zA-Z0-9]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK: ret i64 [[CAST]]
define i64 @b (ptr %arg) {
  %intptr = ptrtoint ptr %arg to i64
  %a = inttoptr i64 %intptr to ptr
  %b = ptrtoint ptr %a to i64
  %with_offset = add i64 %b, 2
  ret i64 %with_offset
}

; CHECK-LABEL: define i64 @c
; CHECK: [[GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 3
; CHECK: [[CAST:%[a-zA-Z0-9]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK: ret i64 [[CAST]]
define i64 @c (ptr %arg) {
  %intptr = ptrtoint ptr %arg to i64
  %with_offset = add i64 %intptr, 3
  %a = inttoptr i64 %with_offset to ptr
  %b = ptrtoint ptr %a to i64
  ret i64 %b
}

; CHECK-LABEL: define ptr @d
; CHECK: [[GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 4
; CHECK: ret ptr [[GEP]]
define ptr @d (ptr %arg) {
  %intptr = ptrtoint ptr %arg to i64
  %with_offset = add i64 %intptr, 4
  %ptr_result = inttoptr i64 %with_offset to ptr
  ret ptr %ptr_result
}

; CHECK-LABEL: define ptr @e
; CHECK: [[ORIGINAL_GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 5
; CHECK: [[NEW_GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr [[ORIGINAL_GEP]], i64 6
; CHECK: ret ptr [[NEW_GEP]]
define ptr @e (ptr %arg) {
  %gep = getelementptr i8, ptr %arg, i64 5
  %intptr = ptrtoint ptr %gep to i64
  %with_offset = add i64 %intptr, 6
  %ptr_result = inttoptr i64 %with_offset to ptr
  ret ptr %ptr_result
}

; CHECK-LABEL: define ptr @f
; CHECK: [[ORIGINAL_GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 7
; CHECK: [[NEW_GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr [[ORIGINAL_GEP]], i64 8
; CHECK: ret ptr [[NEW_GEP]]
define ptr @f (ptr %arg) {
  %intptr = ptrtoint ptr %arg to i64
  %with_offset = add i64 %intptr, 7
  %ptr_with_offset = inttoptr i64 %with_offset to ptr
  %gep = getelementptr i8, ptr %ptr_with_offset, i64 8
  ret ptr %gep
}

; CHECK-LABEL: define i64 @w
; CHECK-NEXT [[WITH_OFFSET:%[a-zA-Z0-9]+]]  = add i64 %arg, 1
; CHECK-NEXT ret i54 [[WITH_OFFSET:%[a-zA-Z0-9]+]]
define i64 @w (i64 %arg) !revng.pointers !5 {
  %with_offset = add i64 %arg, 1
  ret i64 %with_offset
}

; CHECK-LABEL: define i64 @x
; CHECK: ret i64 [[INT]]
define i64 @x (i64 %arg) !revng.pointers !2 {
  %with_offset = add i64 %arg, 1
  ret i64 %with_offset
}

define i64 @y (i64 %arg) !revng.pointers !4 {
  %with_offset = add i64 %arg, 1
  ret i64 %with_offset
}

; CHECK-LABEL: define i64 @z
; CHECK: ret i64 [[INT]]
define i64 @z (i64 %arg) !revng.pointers !5 {
  %ptr = call i64 @x(i64 %arg)
  %with_offset = add i64 %ptr, 1
  ret i64 %with_offset
}



; CHECK-LABEL: define i64 @a_revng_pointers
; CHECK: [[PTR:%[a-zA-Z0-9]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr [[PTR]], i64 1
; CHECK: [[INT:%[a-zA-Z0-9]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK: ret i64 [[INT]]
define i64 @a_revng_pointers (i64 %arg) !revng.pointers !3 {
  %with_offset = add i64 %arg, 1
  ret i64 %with_offset
}

; CHECK-LABEL: define i64 @b_revng_pointers
; CHECK: [[PTR:%[a-zA-Z0-9]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr [[PTR]], i64 1
; CHECK: [[INT:%[a-zA-Z0-9]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK: ret i64 [[INT]]
define i64 @b_revng_pointers (i64 %arg) !revng.pointers !3 {
  %a = inttoptr i64 %arg to ptr
  %b = ptrtoint ptr %a to i64
  %with_offset = add i64 %b, 2
  ret i64 %with_offset
}

; COM: ; CHECK-LABEL: define i64 @c
; COM: ; CHECK: [[GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 3
; COM: ; CHECK: [[CAST:%[a-zA-Z0-9]+]] = ptrtoint ptr [[GEP]] to i64
; COM: ; CHECK: ret i64 [[CAST]]
; COM: define i64 @c (ptr %arg) {
; COM:   %intptr = ptrtoint ptr %arg to i64
; COM:   %with_offset = add i64 %intptr, 3
; COM:   %a = inttoptr i64 %with_offset to ptr
; COM:   %b = ptrtoint ptr %a to i64
; COM:   ret i64 %b
; COM: }
; COM: 
; COM: ; CHECK-LABEL: define ptr @d
; COM: ; CHECK: [[GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 4
; COM: ; CHECK: ret ptr [[GEP]]
; COM: define ptr @d (ptr %arg) {
; COM:   %intptr = ptrtoint ptr %arg to i64
; COM:   %with_offset = add i64 %intptr, 4
; COM:   %ptr_result = inttoptr i64 %with_offset to ptr
; COM:   ret ptr %ptr_result
; COM: }
; COM: 
; COM: ; CHECK-LABEL: define ptr @e
; COM: ; CHECK: [[ORIGINAL_GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 5
; COM: ; CHECK: [[NEW_GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr [[ORIGINAL_GEP]], i64 6
; COM: ; CHECK: ret ptr [[NEW_GEP]]
; COM: define ptr @e (ptr %arg) {
; COM:   %gep = getelementptr i8, ptr %arg, i64 5
; COM:   %intptr = ptrtoint ptr %gep to i64
; COM:   %with_offset = add i64 %intptr, 6
; COM:   %ptr_result = inttoptr i64 %with_offset to ptr
; COM:   ret ptr %ptr_result
; COM: }
; COM: 
; COM: ; CHECK-LABEL: define ptr @f
; COM: ; CHECK: [[ORIGINAL_GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr %arg, i64 7
; COM: ; CHECK: [[NEW_GEP:%[a-zA-Z0-9]+]] = getelementptr i8, ptr [[ORIGINAL_GEP]], i64 8
; COM: ; CHECK: ret ptr [[NEW_GEP]]
; COM: define ptr @f (ptr %arg) {
; COM:   %intptr = ptrtoint ptr %arg to i64
; COM:   %with_offset = add i64 %intptr, 7
; COM:   %ptr_with_offset = inttoptr i64 %with_offset to ptr
; COM:   %gep = getelementptr i8, ptr %ptr_with_offset, i64 8
; COM:   ret ptr %gep
; COM: }


!0 = !{ i1 false }
!1 = !{ i1 true }
!2 = !{ !1, !0 }
!3 = !{ !1, !1 }
!4 = !{ !0, !1 }
!5 = !{ !0, !0 }
!10000 = !{}
