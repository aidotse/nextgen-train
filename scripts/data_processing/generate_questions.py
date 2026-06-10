import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

BATCH_SIZE = 10000


def truncate_examples(example):
    return example["text"][:4000]


def format_prompt(drug_name, section_text, section_title=""):
    """
    Creates a single prompt for a given drug and section.
    MODIFIED: Now takes text and an optional title directly.
    """
    # Use a generic title if none is provided
    title_info = f'Section title: "{section_title}"' if section_title else ""

    # Note: The double curly braces {{ and }} are intentional to escape them in the f-string.
    return f"""You are an expert in creating training data for medical AI. Your task is to generate a single, clear question that a patient might ask, based on the provided drug information.

Follow the style of the examples provided below. The question must be answerable by the provided text.
Your response MUST be a valid JSON object containing a single key: "question".

---
EXAMPLE 1
Drug name: "ABSIMKY"
Text: "ABSIMKY is intended for use under the guidance and supervision of a doctor experienced in the diagnosis and treatment of Crohn’s disease or ulcerative colitis. ABSIMKY 130 mg concentrate for solution for infusion will be given to you by your doctor, through a drip in the vein of your arm (intravenous infusion) over at least one hour."
Question: {{"question": "How to use the medicine ABSIMKY?"}}

EXAMPLE 2
Drug name: "Jubbonti"
Text: "swollen, red area of skin, most commonly in the lower leg that feels hot and tender, possibly with symptoms of fever, pain in the mouth and/or jaw, swelling or non-healing of sores in the mouth or jaw, discharge, numbness or a feeling of heaviness in the jaw, loosening of a tooth, low calcium levels in the blood (hypocalcaemia), spasms, twitches, or cramps in your muscles, numbness or tingling in your fingers, toes or around your mouth, seizures, confusion, loss of consciousness, unusual fractures of the thigh bone, allergic reactions including swelling of the face, lips, tongue, throat or other parts of the body, rash, itching or hives on the skin, wheezing or difficulty breathing, bone, joint, and/or muscle pain which is sometimes severe, arm or leg pain (pain in extremity), painful urination, frequent urination, blood in the urine, inability to hold your urine, upper respiratory tract infection, pain, tingling or numbness that moves down your leg (sciatica), constipation, abdominal discomfort, rash, skin condition with itching, redness and/or dryness (eczema), hair loss (alopecia), fever, vomiting and abdominal pain or discomfort (diverticulitis), ear infection, rash that may occur on the skin or sores in the mouth (lichenoid drug eruptions), allergic reaction that can damage blood vessels mainly in the skin (e.g. purple or brownish-red spots, hives or skin sores) (hypersensitivity vasculitis), ear pain, discharge from the ear and/or an ear infection"
Question: {{"question": "What are the side effects I have to consider when taking Jubbonti?"}}

EXAMPLE 3
Drug name: "N/A"
Text: "if you are allergic to aflibercept or any of the other ingredients of this medicine (listed in section 6), if you have an active or suspected infection in or around the eye (ocular or periocular infection), if you have severe inflammation of the eye (indicated by pain or redness)"
Question: {{"question": "What are the contraindications for using aflibercept?"}}
---

---
CONTEXT
Drug name: "{drug_name}"
{title_info}
Text: "{section_text}"
---

JSON_OUTPUT:"""


def process_and_save_batch(batch_tasks, llm, sampling_params, output_file):
    """Generates questions for a batch of tasks and appends them to the output file."""
    if not batch_tasks:
        return 0

    prompts_to_generate = [task["prompt"] for task in batch_tasks]
    outputs = llm.generate(prompts_to_generate, sampling_params)

    saved_count = 0
    with open(output_file, "a") as f_out:
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            original_context = batch_tasks[idx]["original_context"]

            try:
                if generated_text.startswith("```json"):
                    generated_text = generated_text.replace("```json", "").replace("```", "").strip()

                generated_json = json.loads(generated_text)
                question = generated_json.get("question")

                if question:
                    final_data_point = {
                        "question": question,
                        "answer": original_context["answer"],
                        "metadata": {
                            "drug_name": original_context["drug_name"],
                            "section_title": original_context["title"],
                        },
                    }
                    f_out.write(json.dumps(final_data_point) + "\n")
                    saved_count += 1
            except Exception as e:
                print(f"--- WARNING: Could not process an item. Error: {e}. Text: {generated_text} ---")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Generate questions from medical text data.")
    parser.add_argument(
        "--model_name", type=str, default="google/gemma-3-12b-it", help="Name of the VLLM model to use."
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for VLLM.")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input data file (.jsonl or .parquet)."
    )
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .jsonl file.")
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["jsonl", "parquet"],
        help="Type of the input data ('jsonl' or 'parquet').",
    )
    args = parser.parse_args()

    llm = LLM(model=args.model_name, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=200)

    batch_tasks = []
    total_saved = 0

    # Clear the output file before starting
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    open(args.output_file, "w").close()

    # MODIFIED: Logic to handle different data types
    data_iterator = None

    if args.data_type == "jsonl":
        data_iterator = open(args.input_file)
        desc = "Processing JSONL"
    elif args.data_type == "parquet":
        # Use streaming to avoid loading the entire dataset into memory at once
        data_iterator = load_dataset("parquet", data_files=args.input_file, split="train", streaming=True)
        desc = "Processing Parquet"

    with tqdm(data_iterator, desc=desc) as pbar:
        for item in pbar:
            if args.data_type == "jsonl":
                if not item.strip():
                    continue

                data_point = json.loads(item)
                drug_name = data_point["drug_name"]
                for section in data_point["sections"]:
                    prompt = format_prompt(drug_name, section["text"], section["title"])
                    batch_tasks.append(
                        {
                            "prompt": prompt,
                            "original_context": {
                                "drug_name": drug_name,
                                "title": section["title"],
                                "answer": section["text"],
                            },
                        }
                    )

            elif args.data_type == "parquet":
                # Adapt the parquet data structure to fit the processing logic
                text_content = item["text"]
                text_content = text_content[:4000]
                # Since parquet has no drug/section name, we use placeholders
                drug_name = "N/A"
                section_title = "Full Text"

                prompt = format_prompt(drug_name, text_content, section_title)
                batch_tasks.append(
                    {
                        "prompt": prompt,
                        "original_context": {"drug_name": drug_name, "title": section_title, "answer": text_content},
                    }
                )

            # Batch processing logic is now shared
            if len(batch_tasks) >= BATCH_SIZE:
                print(f"\nProcessing batch of {len(batch_tasks)} prompts...")
                saved_in_batch = process_and_save_batch(batch_tasks, llm, sampling_params, args.output_file)
                total_saved += saved_in_batch
                print(f"Saved {saved_in_batch} items. Total saved: {total_saved}")
                batch_tasks = []

    # Process any remaining tasks after the loop
    if batch_tasks:
        print(f"\nProcessing final batch of {len(batch_tasks)} prompts...")
        saved_in_batch = process_and_save_batch(batch_tasks, llm, sampling_params, args.output_file)
        total_saved += saved_in_batch
        print(f"Saved {saved_in_batch} items. Total saved: {total_saved}")

    # Close the file if it was opened for jsonl
    if args.data_type == "jsonl":
        data_iterator.close()

    print(f"\n\n✅ Processing complete. Total {total_saved} pairs saved to {args.output_file}")


if __name__ == "__main__":
    main()
