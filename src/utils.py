def format_cleaning_plan_string(plan: str) -> str:
    lines = plan.split("\n")
    formatted_lines = []
    for line in lines:
        if line.startswith("###"):
            formatted_lines.append(f"**{line[4:]}**")  # Convert heading
        elif line.startswith("-"):
            formatted_lines.append(f"• {line[2:]}")  # Convert bullet points
        elif ":" in line:
            key, value = map(str.strip, line.split(":", 1))
            formatted_lines.append(f"**{key}:** {value}")
        else:
            formatted_lines.append(line)
    return "\n".join(formatted_lines)
