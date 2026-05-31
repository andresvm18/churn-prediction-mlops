import re

MAX_LEN = 88  # Maximum line length allowed


def wrap_line(line):
    # Return line as-is if it's already short enough
    if len(line) <= MAX_LEN:
        return [line]

    # Basic split: commas and parentheses
    parts = re.split(r"(?<=,)|(?=\()|(?<=\))", line)

    new_lines = []
    current = ""

    # Build new lines by adding parts until max length is reached
    for part in parts:
        if len(current + part) <= MAX_LEN:
            current += part
        else:
            new_lines.append(current.rstrip())  # Save current line
            current = "    " + part.lstrip()  # Start new line with indent

    # Add the last line if there's anything left
    if current:
        new_lines.append(current)

    return new_lines


def fix_file(path):
    # Read the original file
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_content = []

    # Process each line, wrapping long ones
    for line in lines:
        if len(line) > MAX_LEN:
            new_content.extend(wrap_line(line))
        else:
            new_content.append(line.rstrip("\n"))

    # Write back the fixed content
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_content))


if __name__ == "__main__":
    # Fix the Streamlit app file
    file = "./app/streamlit_app.py"
    fix_file(file)
    print(f"Fixed long lines in {file}")
