class CommandError(Exception):
    """A command failed. Mirrors Rust's `Err(String)`: the message is shown to the user verbatim."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status
