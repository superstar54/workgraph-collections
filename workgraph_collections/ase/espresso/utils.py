import re
import html


def parse_pp_inputfile(xml_string):
    # Decode HTML entities
    xml_string = html.unescape(xml_string)

    # Extract the PP_INPUTFILE section
    pp_inputfile_match = re.search(
        r"<PP_INPUTFILE>(.*?)</PP_INPUTFILE>", xml_string, re.DOTALL
    )
    if not pp_inputfile_match:
        return None

    pp_inputfile_content = pp_inputfile_match.group(1).strip()

    # Split the section into lines and parse
    lines = pp_inputfile_content.split("\n")
    config_data = {}
    current_section = None
    pseudo_potential_test_cards = ""

    for line in lines:
        line = line.strip()
        if line.startswith("&"):
            # Start of a new section
            current_section = line[1:].split()[
                0
            ]  # Removes '&' and splits to get the section name
            config_data[current_section] = {}
        elif line.startswith("/"):
            # End of a section
            current_section = None
        elif current_section and "=" in line:
            # Within a section, parse parameters
            key_values = line.split(",")
            for key_value in key_values:
                key_value_pair = key_value.split("=")
                if len(key_value_pair) == 2:
                    key, value = key_value_pair[0].strip(), key_value_pair[
                        1
                    ].strip().strip("'")
                    config_data[current_section][key] = value
        elif line.isdigit():
            # Start recording pseudo potential cards
            pseudo_potential_test_cards += line + "\n"
        elif line:
            # Continuing the pseudo potential cards
            pseudo_potential_test_cards += line + "\n"

    return {
        "config_data": config_data,
        "pseudo_potential_test_cards": pseudo_potential_test_cards.strip(),
    }
