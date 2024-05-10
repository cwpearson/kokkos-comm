//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include "KokkosComm.hpp"

namespace {
TEST(Waitall, 1) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks (" << size << " provided)";
  }

  Kokkos::View<double *> sv("sv", 100);
  Kokkos::View<double *> rv("rv", 100);
  Kokkos::DefaultExecutionSpace space;

  if (0 == rank) {
    auto r1 = KokkosComm::isend(space, sv, 1, 0, MPI_COMM_WORLD);
    auto r2 = KokkosComm::isend(space, sv, 1, 0, MPI_COMM_WORLD);

    // FIXME: use kokkos-tools to ensure at most one fence is here
    KokkosComm::wait_all(r1, r2);

    // FIXME: use kokkos-tools to ensure at most one fence is here
    KokkosComm::wait_all({r1, r2});
  } else if (1 == rank) {
    auto r1 = KokkosComm::irecv(space, rv, 0, 0, MPI_COMM_WORLD);
    auto r2 = KokkosComm::irecv(space, rv, 0, 0, MPI_COMM_WORLD);

    // FIXME: use kokkos-tools to ensure at most one fence is here
    KokkosComm::wait_all(r1, r2);

    // FIXME: use kokkos-tools to ensure at most one fence is here
    KokkosComm::wait_all({r1, r2});
  }


}
}  // namespace
