from crewai_tools import tools

@tools.tool("Test Tool")
def test_function(input_str: str) -> str:
    """A simple test function."""
    return f"Test successful: {input_str}"

print("Successfully imported 'tools' from 'crewai_tools' and defined a test tool using tools.tool.")