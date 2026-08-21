import random
import string

import pytest

from agentprobe.verifier.smt_compiler import SMTInvariantVerifier


def test_fuzz_smt_compiler():
    verifier = SMTInvariantVerifier()
    commands = ["read", "write", "delete", "execute"]

    for _ in range(1000):
        cmd = random.choice(commands)
        # Generate random paths
        path_len = random.randint(1, 50)
        path = "/".join("".join(random.choices(string.ascii_letters, k=5)) for _ in range(path_len))
        if random.random() > 0.5:
            path = "/workspace/" + path

        is_readonly = random.choice([True, False])

        try:
            valid, msg = verifier.verify_filesystem_safety(cmd, path, "/workspace", is_readonly)
            assert isinstance(valid, bool)
        except Exception as e:
            pytest.fail(f"SMT Verifier crashed on fuzzing input: {e}")
