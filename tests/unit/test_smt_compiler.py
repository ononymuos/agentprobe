from agentprobe.verifier.smt_compiler import SMTInvariantVerifier


def test_smt_compiler_safe():
    verifier = SMTInvariantVerifier()
    valid, msg = verifier.verify_filesystem_safety("read", "/workspace/data.txt", "/workspace", False)
    assert valid is True

def test_smt_compiler_unsafe_traversal():
    verifier = SMTInvariantVerifier()
    valid, msg = verifier.verify_filesystem_safety("read", "/workspace/../etc/passwd", "/workspace", False)
    assert valid is False
    assert msg is not None and "breaks sandbox bounds" in msg

def test_smt_compiler_readonly_violation():
    verifier = SMTInvariantVerifier()
    valid, msg = verifier.verify_filesystem_safety("write", "/workspace/data.txt", "/workspace", True)
    assert valid is False
    assert msg is not None and "breaks sandbox bounds or read-only policy" in msg
