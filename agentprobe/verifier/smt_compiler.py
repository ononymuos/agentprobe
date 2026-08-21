"""
agentprobe.verifier.smt_compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compiles runtime declarative schemas and invariant rules into
First-Order Logic formulas solved by Microsoft Z3 in <1ms.
"""

import z3  # type: ignore


class SMTInvariantVerifier:
    def __init__(self) -> None:
        self.solver = z3.Solver()
        self.solver.set("timeout", 50)  # 50ms strict bound

    def verify_filesystem_safety(
        self,
        command_type: str,
        target_path: str,
        sandbox_root: str,
        is_read_only: bool
    ) -> tuple[bool, str | None]:
        """
        Symbolically proves whether a filesystem operation violates sandbox invariants.
        """
        self.solver.reset()

        cmd = z3.StringVal(command_type)
        path = z3.StringVal(target_path)
        root = z3.StringVal(sandbox_root)

        # Invariant 1: Path must be contained within sandbox root
        is_subpath = z3.PrefixOf(root, path)

        # Invariant 2: Write operations forbidden if session is read-only
        is_write = z3.Or(cmd == z3.StringVal("write"), cmd == z3.StringVal("delete"))
        write_violation = z3.And(z3.BoolVal(is_read_only), is_write)

        # Invariant 3: Parent traversal forbidden
        has_parent_traversal = z3.Contains(path, z3.StringVal(".."))

        # Safety Predicate
        safety_condition = z3.And(is_subpath, z3.Not(write_violation), z3.Not(has_parent_traversal))

        # We assert the negation to search for a counterexample
        self.solver.add(z3.Not(safety_condition))
        check_result = self.solver.check()

        if check_result == z3.sat:
            return False, f"SMT Invariant Violation: Target path '{target_path}' breaks sandbox bounds or read-only policy."
        elif check_result == z3.unsat:
            return True, None
        else:
            return False, "SMT Solver Timeout: Verification failed to prove safety."
