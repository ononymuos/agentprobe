import pytest
z3 = pytest.importorskip("z3")
from agentprobe.verifier.smt_compiler import SMTInvariantVerifier

def test_smt_compiler_valid():
    verifier = SMTInvariantVerifier()
    valid, err = verifier.verify_filesystem_safety(
        command_type="read",
        target_path="/workspace/test.txt",
        sandbox_root="/workspace",
        is_read_only=False
    )
    assert valid
    assert err is None

def test_smt_compiler_parent_traversal():
    verifier = SMTInvariantVerifier()
    valid, err = verifier.verify_filesystem_safety(
        command_type="read",
        target_path="/workspace/../etc/passwd",
        sandbox_root="/workspace",
        is_read_only=False
    )
    assert not valid
    assert "Violation" in err

def test_smt_compiler_read_only_write():
    verifier = SMTInvariantVerifier()
    valid, err = verifier.verify_filesystem_safety(
        command_type="write",
        target_path="/workspace/test.txt",
        sandbox_root="/workspace",
        is_read_only=True
    )
    assert not valid
