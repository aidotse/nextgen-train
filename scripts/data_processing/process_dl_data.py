import json
import os
import xml.etree.ElementTree as ET
import zipfile

from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PRESCRIPTION_DATA_PATH = os.path.join(DATA_DIR, "dailymed_prescription", "unzipped_data", "prescription")


def process_section(section_element, parent_title, namespace):
    """
    Recursively processes a section and all its nested sub-sections.
    """
    results = []

    # --- 1. Process the current section ---
    current_title_element = section_element.find("hl7:title", namespace)
    current_title = (
        current_title_element.text.strip()
        if current_title_element is not None and current_title_element.text
        else "NO_TITLE"
    )

    # Combine parent and current titles for full context
    full_title = f"{parent_title} - {current_title}" if parent_title else current_title

    text_element = section_element.find("hl7:text", namespace)
    text_content = ""
    if text_element is not None:
        # Join all text content within the <text> tag
        text_content = " ".join(node.text.strip() for node in text_element.iter() if node.text and node.text.strip())

    # Add the current section's data if it has text
    if text_content:
        results.append({"title": full_title, "text": text_content})

    # --- 2. Find and process any nested sub-sections ---
    nested_sections = section_element.findall("./hl7:component/hl7:section", namespace)
    for sub_section in nested_sections:
        # The magic happens here: we pass the current section's full title
        # as the parent title for the next level down.
        results.extend(process_section(sub_section, full_title, namespace))

    return results


def parse_spl_xml(xml_content):
    """
    Parses the XML to extract the drug name and all sections recursively.
    """
    try:
        root = ET.fromstring(xml_content)  # noqa: S314
        namespace = {"hl7": "urn:hl7-org:v3"}

        drug_name_element = root.find("hl7:title", namespace)
        drug_name = (
            drug_name_element.text.strip()
            if drug_name_element is not None and drug_name_element.text
            else "UNKNOWN_DRUG"
        )

        structured_body = root.find(".//hl7:structuredBody", namespace)
        if structured_body is None:
            return {"drug_name": drug_name, "sections": []}

        all_sections_data = []
        # Find only the top-level sections to start the recursion
        top_level_sections = structured_body.findall("./hl7:component/hl7:section", namespace)

        for section in top_level_sections:
            # Start the recursive processing with no parent title
            all_sections_data.extend(process_section(section, "", namespace))

        return {"drug_name": drug_name, "sections": all_sections_data}

    except ET.ParseError:
        return None


if __name__ == "__main__":
    if not os.path.exists(PRESCRIPTION_DATA_PATH):
        print(f"Error: Directory {PRESCRIPTION_DATA_PATH} not found.")
        exit(1)

    zip_files = [f for f in os.listdir(PRESCRIPTION_DATA_PATH) if f.endswith(".zip")]

    output_filename = os.path.join(DATA_DIR, "parsed_drug_data_recursive.jsonl")
    print(f"Starting to process {len(zip_files)} zip files...")

    # We process the files and write to the output file in a single loop
    # to avoid holding all the data in memory.
    processed_count = 0
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(output_filename, "w") as f_out:
        for zip_filename in tqdm(zip_files, desc="Processing drug labels"):
            zip_filepath = os.path.join(PRESCRIPTION_DATA_PATH, zip_filename)
            try:
                with zipfile.ZipFile(zip_filepath, "r") as zf:
                    xml_filenames = [name for name in zf.namelist() if name.endswith(".xml")]
                    if not xml_filenames:
                        print(f"Warning: No XML file found in {zip_filename}")
                        continue
                    xml_filename = xml_filenames[0]
                    with zf.open(xml_filename) as xml_file:
                        xml_content = xml_file.read()
                        parsed_data = parse_spl_xml(xml_content)
                        if parsed_data and parsed_data["sections"]:
                            entry = {
                                "source_file": zip_filename,
                                "drug_name": parsed_data["drug_name"],
                                "sections": parsed_data["sections"],
                            }
                            f_out.write(json.dumps(entry) + "\n")
                            processed_count += 1
            except Exception as e:
                print(f"Warning: Could not process {zip_filename}. Error: {e}")

    print(f"\nProcessing complete. Extracted data from {processed_count} files.")
    print(f"Data saved to '{output_filename}'.")
