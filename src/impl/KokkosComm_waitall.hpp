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

#pragma once

#include "KokkosComm_concepts.hpp"

namespace KokkosComm::Impl {


        template <KokkosExecutionSpace ExecSpace>
        struct Fences {

            std::vector<ExecSpace> toFence;


            void add_req(const KokkosComm::Req<ExecSpace> &req) {

                const std::optional<ExecSpace> &maybeReqSpace = req.space_instance();

                if (maybeReqSpace) {
                    const ExecSpace &reqSpace = *maybeReqSpace;
                    if (std::find(toFence.begin(), toFence.end(), reqSpace) == toFence.end()) {
                        toFence.push_back(reqSpace);
                    }
                }

            }

            void fence() const {
                for (const auto &space : toFence) {
                    space.fence();
                }
            }

        };


    template <KokkosExecutionSpace ExecSpace>
    void wait_all_f(Fences<ExecSpace> &fs, const KokkosComm::Req<ExecSpace> &first) {
        fs.add_req(first);
    }

    template <KokkosExecutionSpace ExecSpace, typename... Reqs>
    void wait_all_f(Fences<ExecSpace> &fs, const KokkosComm::Req<ExecSpace> &first, const Reqs&... rest) {
        fs.add_req(first);
        wait_all_f(fs, rest...);
    }

    // FIXME: constrain Reqs to be KokkosComm::Req<ExecSpace> and all the same type
    template <typename... Reqs>
    void wait_all(const Reqs&... reqs) {
        using ExecSpace = std::tuple_element_t<0, std::tuple<Reqs...>>::execution_space;
        Fences<ExecSpace> fs;
        wait_all_f(fs, reqs...);
        fs.fence();
    };


    template <typename Req>
    void wait_all(std::initializer_list<Req> l) {
        using execution_space = Req::execution_space;
        Fences<execution_space> fs;
        for (auto &e : l) {
            fs.add_req(e);
        }
        fs.fence();
    }

}