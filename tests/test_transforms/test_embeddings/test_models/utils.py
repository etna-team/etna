def check_logged_loss(log_file: str, n_lines_expected: int):
    """Check that model loss is logged into the file."""
    with open(log_file, "r") as in_file:
        lines = in_file.readlines()
        print(lines)
        assert len(lines) == n_lines_expected
        for i, line in enumerate(lines):
            assert f"Epoch {i}: loss=" in line
