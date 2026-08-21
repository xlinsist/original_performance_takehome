"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self, enable_pauses: bool = False):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.enable_pauses = enable_pauses

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        def slot_reads_writes(engine, slot):
            reads = set()
            writes = set()
            has_mem_read = False
            has_mem_write = False

            if engine == "alu":
                _, dest, a1, a2 = slot
                writes.add(dest)
                reads.update((a1, a2))
            elif engine == "load":
                op = slot[0]
                if op == "load":
                    _, dest, addr = slot
                    writes.add(dest)
                    reads.add(addr)
                    has_mem_read = True
                elif op == "load_offset":
                    _, dest, addr, offset = slot
                    writes.add(dest + offset)
                    reads.add(addr + offset)
                    has_mem_read = True
                elif op == "vload":
                    _, dest, addr = slot
                    writes.update(dest + i for i in range(VLEN))
                    reads.add(addr)
                    has_mem_read = True
                elif op == "const":
                    _, dest, _ = slot
                    writes.add(dest)
            elif engine == "store":
                op = slot[0]
                if op == "store":
                    _, addr, src = slot
                    reads.update((addr, src))
                    has_mem_write = True
                elif op == "vstore":
                    _, addr, src = slot
                    reads.add(addr)
                    reads.update(src + i for i in range(VLEN))
                    has_mem_write = True
            elif engine == "flow":
                op = slot[0]
                if op == "select":
                    _, dest, cond, a, b = slot
                    writes.add(dest)
                    reads.update((cond, a, b))
                elif op == "add_imm":
                    _, dest, a, _ = slot
                    writes.add(dest)
                    reads.add(a)
                elif op == "vselect":
                    _, dest, cond, a, b = slot
                    writes.update(dest + i for i in range(VLEN))
                    reads.update(cond + i for i in range(VLEN))
                    reads.update(a + i for i in range(VLEN))
                    reads.update(b + i for i in range(VLEN))
                elif op == "trace_write":
                    _, val = slot
                    reads.add(val)
                elif op == "cond_jump":
                    _, cond, _ = slot
                    reads.add(cond)
                elif op == "cond_jump_rel":
                    _, cond, _ = slot
                    reads.add(cond)
                elif op == "jump_indirect":
                    _, addr = slot
                    reads.add(addr)
                elif op == "coreid":
                    _, dest = slot
                    writes.add(dest)
            elif engine == "valu":
                op = slot[0]
                if op == "vbroadcast":
                    _, dest, src = slot
                    writes.update(dest + i for i in range(VLEN))
                    reads.add(src)
                elif op == "multiply_add":
                    _, dest, a, b, c = slot
                    writes.update(dest + i for i in range(VLEN))
                    reads.update(a + i for i in range(VLEN))
                    reads.update(b + i for i in range(VLEN))
                    reads.update(c + i for i in range(VLEN))
                else:
                    _, dest, a1, a2 = slot
                    writes.update(dest + i for i in range(VLEN))
                    reads.update(a1 + i for i in range(VLEN))
                    reads.update(a2 + i for i in range(VLEN))

            return reads, writes, has_mem_read, has_mem_write

        instrs = []
        bundle = {}
        bundle_reads = set()
        bundle_writes = set()
        bundle_has_mem_write = False

        def flush_bundle():
            nonlocal bundle, bundle_reads, bundle_writes, bundle_has_mem_write
            if bundle:
                instrs.append(bundle)
            bundle = {}
            bundle_reads = set()
            bundle_writes = set()
            bundle_has_mem_write = False

        for engine, slot in slots:
            reads, writes, has_mem_read, has_mem_write = slot_reads_writes(engine, slot)
            engine_slots = bundle.get(engine, [])

            conflicts = False
            if len(engine_slots) >= SLOT_LIMITS[engine]:
                conflicts = True
            elif reads & bundle_writes:
                conflicts = True
            elif writes & bundle_writes:
                conflicts = True

            if conflicts:
                flush_bundle()

            bundle.setdefault(engine, []).append(slot)
            bundle_reads.update(reads)
            bundle_writes.update(writes)
            bundle_has_mem_write = bundle_has_mem_write or has_mem_write

        flush_bundle()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_vhash(self, val_addr, tmp1_addr, tmp2_addr, stage_consts):
        slots = []
        for op1, op2, op3, val1_vec, val3_vec, mul_vec in stage_consts:
            slots.append(("valu", (op1, tmp1_addr, val_addr, val1_vec)))
            if mul_vec is not None:
                slots.append(
                    ("valu", ("multiply_add", val_addr, val_addr, mul_vec, tmp1_addr))
                )
            else:
                slots.append(("valu", (op3, tmp2_addr, val_addr, val3_vec)))
                slots.append(("valu", (op2, val_addr, tmp1_addr, tmp2_addr)))
        return slots

    def build_depth1_select(self, active_groups, vec_three, shallow_node_vec2, shallow_node_diff12):
        slots = []
        for vec_group, _ in active_groups:
            slots.append(("valu", ("<", vec_group["tmp1"], vec_group["idx"], vec_three)))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["tmp3"],
                        vec_group["tmp1"],
                        shallow_node_diff12,
                        shallow_node_vec2,
                    ),
                )
            )
        return slots

    def build_depth2_select(
        self,
        active_groups,
        vec_thresh,
        vec_six,
        shallow_node_vec4,
        shallow_node_diff34,
        shallow_node_vec6,
        shallow_node_diff56,
        scalar_five,
        scalar_seven,
    ):
        slots = []
        slots.append(("valu", ("vbroadcast", vec_thresh, scalar_five)))
        for vec_group, _ in active_groups:
            slots.append(("valu", ("<", vec_group["tmp1"], vec_group["idx"], vec_six)))
        for vec_group, _ in active_groups:
            slots.append(("valu", ("<", vec_group["tmp2"], vec_group["idx"], vec_thresh)))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["tmp3"],
                        vec_group["tmp2"],
                        shallow_node_diff34,
                        shallow_node_vec4,
                    ),
                )
            )
        slots.append(("valu", ("vbroadcast", vec_thresh, scalar_seven)))
        for vec_group, _ in active_groups:
            slots.append(("valu", ("<", vec_group["tmp2"], vec_group["idx"], vec_thresh)))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["tmp2"],
                        vec_group["tmp2"],
                        shallow_node_diff56,
                        shallow_node_vec6,
                    ),
                )
            )
        for vec_group, _ in active_groups:
            slots.append(("valu", ("-", vec_group["tmp3"], vec_group["tmp3"], vec_group["tmp2"])))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["tmp3"],
                        vec_group["tmp1"],
                        vec_group["tmp3"],
                        vec_group["tmp2"],
                    ),
                )
            )
        return slots

    def build_depth3_select(
        self,
        active_groups,
        vec_one,
        vec_two,
        vec_four,
        shallow_node_vecs,
        shallow_node_diffs,
    ):
        slots = []
        for vec_group, _ in active_groups:
            slots.append(("valu", ("&", vec_group["tmp1"], vec_group["idx"], vec_one)))
        for vec_group, _ in active_groups:
            slots.append(("valu", ("<", vec_group["tmp1"], vec_group["tmp1"], vec_one)))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["tmp3"],
                        vec_group["tmp1"],
                        shallow_node_diffs["78"],
                        shallow_node_vecs[8],
                    ),
                )
            )
        for vec_group, _ in active_groups:
            slots.append(("valu", ("&", vec_group["tmp2"], vec_group["idx"], vec_two)))
        for vec_group, _ in active_groups:
            slots.append(("valu", ("<", vec_group["tmp2"], vec_group["tmp2"], vec_one)))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["node_val"],
                        vec_group["tmp1"],
                        shallow_node_diffs["910"],
                        shallow_node_vecs[10],
                    ),
                )
            )
        for vec_group, _ in active_groups:
            slots.append(("valu", ("-", vec_group["tmp3"], vec_group["tmp3"], vec_group["node_val"])))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["tmp3"],
                        vec_group["tmp2"],
                        vec_group["tmp3"],
                        vec_group["node_val"],
                    ),
                )
            )
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["node_val"],
                        vec_group["tmp1"],
                        shallow_node_diffs["1112"],
                        shallow_node_vecs[12],
                    ),
                )
            )
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["tmp1"],
                        vec_group["tmp1"],
                        shallow_node_diffs["1314"],
                        shallow_node_vecs[14],
                    ),
                )
            )
        for vec_group, _ in active_groups:
            slots.append(("valu", ("-", vec_group["node_val"], vec_group["node_val"], vec_group["tmp1"])))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["node_val"],
                        vec_group["tmp2"],
                        vec_group["node_val"],
                        vec_group["tmp1"],
                    ),
                )
            )
        for vec_group, _ in active_groups:
            slots.append(("valu", ("&", vec_group["tmp2"], vec_group["idx"], vec_four)))
        for vec_group, _ in active_groups:
            slots.append(("valu", ("<", vec_group["tmp2"], vec_group["tmp2"], vec_one)))
        for vec_group, _ in active_groups:
            slots.append(("valu", ("-", vec_group["tmp3"], vec_group["tmp3"], vec_group["node_val"])))
        for vec_group, _ in active_groups:
            slots.append(
                (
                    "valu",
                    (
                        "multiply_add",
                        vec_group["node_val"],
                        vec_group["tmp2"],
                        vec_group["tmp3"],
                        vec_group["node_val"],
                    ),
                )
            )
        return slots

    def build_generic_quad_load_batches(self, quad_groups):
        batches = []
        for offset in range(VLEN):
            batches.append(
                [("load", ("load_offset", quad_groups[0]["tmp3"], quad_groups[0]["tmp3"], offset)),
                 ("load", ("load_offset", quad_groups[1]["tmp3"], quad_groups[1]["tmp3"], offset))]
            )
            batches.append(
                [("load", ("load_offset", quad_groups[2]["tmp3"], quad_groups[2]["tmp3"], offset)),
                 ("load", ("load_offset", quad_groups[3]["tmp3"], quad_groups[3]["tmp3"], offset))]
            )
        return batches

    def build_generic_quad_valu_batches(self, quad_groups, hash_stage_consts, vec_one, vec_two, do_update: bool):
        batches = []
        batches.append(
            [("valu", ("^", vec_group["val"], vec_group["val"], vec_group["tmp3"])) for vec_group in quad_groups]
        )
        for op1, op2, op3, val1_vec, val3_vec, mul_vec in hash_stage_consts:
            batches.append([("valu", (op1, vec_group["tmp1"], vec_group["val"], val1_vec)) for vec_group in quad_groups])
            if mul_vec is not None:
                batches.append(
                    [
                        ("valu", ("multiply_add", vec_group["val"], vec_group["val"], mul_vec, vec_group["tmp1"]))
                        for vec_group in quad_groups
                    ]
                )
            else:
                batches.append([("valu", (op3, vec_group["tmp2"], vec_group["val"], val3_vec)) for vec_group in quad_groups])
                batches.append(
                    [("valu", (op2, vec_group["val"], vec_group["tmp1"], vec_group["tmp2"])) for vec_group in quad_groups]
                )
        if do_update:
            batches.append([("valu", ("&", vec_group["tmp1"], vec_group["val"], vec_one)) for vec_group in quad_groups])
            batches.append(
                [
                    (
                        "valu",
                        (
                            "multiply_add",
                            vec_group["idx"],
                            vec_group["idx"],
                            vec_two,
                            vec_group["tmp1"],
                        ),
                    )
                    for vec_group in quad_groups
                ]
            )
        return batches

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Mixed SIMD implementation with harness-aware specialization.
        """
        forest_values_base = 7
        inp_values_base = 7 + n_nodes + batch_size

        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        if self.enable_pauses:
            self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        unroll = 32

        vec_groups = []
        for group in range(unroll):
            vec_groups.append(
                {
                    "idx": self.alloc_scratch(f"vec_idx_{group}", VLEN),
                    "val": self.alloc_scratch(f"vec_val_{group}", VLEN),
                    "tmp1": self.alloc_scratch(f"vec_tmp1_{group}", VLEN),
                    "tmp2": self.alloc_scratch(f"vec_tmp2_{group}", VLEN),
                    "tmp3": self.alloc_scratch(f"vec_tmp3_{group}", VLEN),
                }
            )

        vec_one = self.alloc_scratch("vec_one", VLEN)
        vec_two = self.alloc_scratch("vec_two", VLEN)
        vec_three = self.alloc_scratch("vec_three", VLEN)
        vec_thresh = self.alloc_scratch("vec_thresh", VLEN)
        vec_six = self.alloc_scratch("vec_six", VLEN)
        forest_root_val = self.alloc_scratch("forest_root_val")
        vec_forest_root_val = self.alloc_scratch("vec_forest_root_val", VLEN)

        forest_values_base_addr = self.scratch_const(forest_values_base)
        scalar_five = self.scratch_const(5)
        scalar_seven = self.scratch_const(7)

        vector_constants = [
            (vec_one, one_const),
            (vec_two, two_const),
            (vec_three, self.scratch_const(3)),
            (vec_six, self.scratch_const(6)),
        ]
        for dest, src in vector_constants:
            self.add("valu", ("vbroadcast", dest, src))
        self.add("load", ("load", forest_root_val, forest_values_base_addr))
        self.add("valu", ("vbroadcast", vec_forest_root_val, forest_root_val))

        forest_node_odd = self.alloc_scratch("forest_node_odd")
        forest_node_even = self.alloc_scratch("forest_node_even")
        forest_node_diff = self.alloc_scratch("forest_node_diff")
        vec_forest_node_2 = self.alloc_scratch("vec_forest_node_2", VLEN)
        vec_forest_diff_12 = self.alloc_scratch("vec_forest_diff_12", VLEN)
        vec_forest_node_4 = self.alloc_scratch("vec_forest_node_4", VLEN)
        vec_forest_diff_34 = self.alloc_scratch("vec_forest_diff_34", VLEN)
        vec_forest_node_6 = self.alloc_scratch("vec_forest_node_6", VLEN)
        vec_forest_diff_56 = self.alloc_scratch("vec_forest_diff_56", VLEN)
        self.add("load", ("load", forest_node_odd, self.scratch_const(8)))
        self.add("load", ("load", forest_node_even, self.scratch_const(9)))
        self.add("alu", ("-", forest_node_diff, forest_node_odd, forest_node_even))
        self.add("valu", ("vbroadcast", vec_forest_node_2, forest_node_even))
        self.add("valu", ("vbroadcast", vec_forest_diff_12, forest_node_diff))
        self.add("load", ("load", forest_node_odd, self.scratch_const(10)))
        self.add("load", ("load", forest_node_even, self.scratch_const(11)))
        self.add("alu", ("-", forest_node_diff, forest_node_odd, forest_node_even))
        self.add("valu", ("vbroadcast", vec_forest_node_4, forest_node_even))
        self.add("valu", ("vbroadcast", vec_forest_diff_34, forest_node_diff))
        self.add("load", ("load", forest_node_odd, self.scratch_const(12)))
        self.add("load", ("load", forest_node_even, self.scratch_const(13)))
        self.add("alu", ("-", forest_node_diff, forest_node_odd, forest_node_even))
        self.add("valu", ("vbroadcast", vec_forest_node_6, forest_node_even))
        self.add("valu", ("vbroadcast", vec_forest_diff_56, forest_node_diff))

        hash_stage_consts = []
        hash_stage_setup = [None] * len(HASH_STAGES)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val1_scalar = self.scratch_const(val1)
            val1_vec = self.alloc_scratch(f"hash_stage_{hi}_val1", VLEN)
            setup_slots = [("valu", ("vbroadcast", val1_vec, val1_scalar))]
            mul_vec = None
            val3_vec = None
            if op2 == "+" and op3 == "<<":
                mul_scalar = self.scratch_const(1 << val3)
                mul_vec = self.alloc_scratch(f"hash_stage_{hi}_mul", VLEN)
                setup_slots.append(("valu", ("vbroadcast", mul_vec, mul_scalar)))
            else:
                val3_scalar = self.scratch_const(val3)
                val3_vec = self.alloc_scratch(f"hash_stage_{hi}_val3", VLEN)
                setup_slots.append(("valu", ("vbroadcast", val3_vec, val3_scalar)))
            hash_stage_setup[hi] = setup_slots
            hash_stage_consts.append((op1, op2, op3, val1_vec, val3_vec, mul_vec))
        for stage_setup in hash_stage_setup:
            for engine, slot in stage_setup:
                self.add(engine, slot)

        for i in range(0, batch_size, unroll * VLEN):
            active_groups = []
            for group in range(unroll):
                group_i = i + group * VLEN
                if group_i >= batch_size:
                    continue
                active_groups.append((vec_groups[group], self.scratch_const(inp_values_base + group_i)))

            for vec_group, val_addr in active_groups:
                body.append(("load", ("vload", vec_group["val"], val_addr)))

            for round in range(rounds):
                start_idx_zero = round % (forest_height + 1) == 0
                depth_since_reset = round % (forest_height + 1)
                needs_idx_update = round + 1 < rounds and (round + 1) % (forest_height + 1) != 0
                if start_idx_zero:
                    pass
                elif depth_since_reset == 1:
                    body.extend(self.build_depth1_select(active_groups, vec_three, vec_forest_node_2, vec_forest_diff_12))
                elif depth_since_reset == 2:
                    body.extend(
                        self.build_depth2_select(
                            active_groups,
                            vec_thresh,
                            vec_six,
                            vec_forest_node_4,
                            vec_forest_diff_34,
                            vec_forest_node_6,
                            vec_forest_diff_56,
                            scalar_five,
                            scalar_seven,
                        )
                    )
                else:
                    generic_groups = [vec_group for vec_group, _ in active_groups]
                    if len(generic_groups) >= 4:
                        quads = [generic_groups[i : i + 4] for i in range(0, len(generic_groups) // 4 * 4, 4)]
                        tail_groups = generic_groups[len(quads) * 4 :]
                        load_batches = [self.build_generic_quad_load_batches(quad) for quad in quads]
                        valu_batches = [
                            self.build_generic_quad_valu_batches(quad, hash_stage_consts, vec_one, vec_two, needs_idx_update)
                            for quad in quads
                        ]
                        for vec_group in quads[0]:
                            body.append(("valu", ("+", vec_group["tmp3"], vec_group["idx"], vec_six)))
                        load0_start = 0
                        for quad in quads[1:]:
                            body.extend([("valu", ("+", vec_group["tmp3"], vec_group["idx"], vec_six)) for vec_group in quad])
                            if load0_start < len(load_batches[0]):
                                body.extend(load_batches[0][load0_start])
                                load0_start += 1
                        for batch in load_batches[0][load0_start:]:
                            body.extend(batch)
                        for quad_i, quad_valu_batches in enumerate(valu_batches[:-1]):
                            next_load_batches = load_batches[quad_i + 1]
                            for batch_i, valu_batch in enumerate(quad_valu_batches):
                                body.extend(valu_batch)
                                if batch_i < len(next_load_batches):
                                    body.extend(next_load_batches[batch_i])
                        for batch in valu_batches[-1]:
                            body.extend(batch)
                        if tail_groups:
                            for vec_group in tail_groups:
                                body.append(("valu", ("+", vec_group["tmp3"], vec_group["idx"], vec_six)))
                            for offset in range(VLEN):
                                for vec_group in tail_groups:
                                    body.append(("load", ("load_offset", vec_group["tmp3"], vec_group["tmp3"], offset)))
                            for vec_group in tail_groups:
                                body.append(("valu", ("^", vec_group["val"], vec_group["val"], vec_group["tmp3"])))
                            for op1, op2, op3, val1_vec, val3_vec, mul_vec in hash_stage_consts:
                                for vec_group in tail_groups:
                                    body.append(("valu", (op1, vec_group["tmp1"], vec_group["val"], val1_vec)))
                                if mul_vec is not None:
                                    for vec_group in tail_groups:
                                        body.append(
                                            (
                                                "valu",
                                                ("multiply_add", vec_group["val"], vec_group["val"], mul_vec, vec_group["tmp1"]),
                                            )
                                        )
                                else:
                                    for vec_group in tail_groups:
                                        body.append(("valu", (op3, vec_group["tmp2"], vec_group["val"], val3_vec)))
                                    for vec_group in tail_groups:
                                        body.append(("valu", (op2, vec_group["val"], vec_group["tmp1"], vec_group["tmp2"])))
                            if needs_idx_update:
                                for vec_group in tail_groups:
                                    body.append(("valu", ("&", vec_group["tmp1"], vec_group["val"], vec_one)))
                                for vec_group in tail_groups:
                                    body.append(
                                        (
                                            "valu",
                                            (
                                                "multiply_add",
                                                vec_group["idx"],
                                                vec_group["idx"],
                                                vec_two,
                                                vec_group["tmp1"],
                                            ),
                                        )
                                    )
                        continue
                    else:
                        for vec_group in generic_groups:
                            body.append(("valu", ("+", vec_group["tmp3"], vec_group["idx"], vec_six)))
                        for offset in range(VLEN):
                            for vec_group in generic_groups:
                                body.append(("load", ("load_offset", vec_group["tmp3"], vec_group["tmp3"], offset)))

                for vec_group, _ in active_groups:
                    body.append(
                        (
                            "valu",
                            (
                                "^",
                                vec_group["val"],
                                vec_group["val"],
                                vec_forest_root_val if start_idx_zero else vec_group["tmp3"],
                            ),
                        )
                    )

                for op1, op2, op3, val1_vec, val3_vec, mul_vec in hash_stage_consts:
                    for vec_group, _ in active_groups:
                        body.append(("valu", (op1, vec_group["tmp1"], vec_group["val"], val1_vec)))
                    if mul_vec is not None:
                        for vec_group, _ in active_groups:
                            body.append(
                                (
                                    "valu",
                                    ("multiply_add", vec_group["val"], vec_group["val"], mul_vec, vec_group["tmp1"]),
                                )
                            )
                    else:
                        for vec_group, _ in active_groups:
                            body.append(("valu", (op3, vec_group["tmp2"], vec_group["val"], val3_vec)))
                        for vec_group, _ in active_groups:
                            body.append(("valu", (op2, vec_group["val"], vec_group["tmp1"], vec_group["tmp2"])))

                if needs_idx_update:
                    for vec_group, _ in active_groups:
                        body.append(("valu", ("&", vec_group["tmp1"], vec_group["val"], vec_one)))
                    for vec_group, _ in active_groups:
                        if start_idx_zero:
                            body.append(("valu", ("+", vec_group["idx"], vec_group["tmp1"], vec_two)))
                        else:
                            body.append(
                                (
                                    "valu",
                                    (
                                        "multiply_add",
                                        vec_group["idx"],
                                        vec_group["idx"],
                                        vec_two,
                                        vec_group["tmp1"],
                                    ),
                                )
                            )

            for vec_group, val_addr in active_groups:
                body.append(("store", ("vstore", val_addr, vec_group["val"])))

        setup_slots = []
        for instr in self.instrs:
            for engine, slots in instr.items():
                for slot in slots:
                    setup_slots.append((engine, slot))

        self.instrs = self.build(setup_slots)
        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        if self.enable_pauses:
            self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder(enable_pauses=True)
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
