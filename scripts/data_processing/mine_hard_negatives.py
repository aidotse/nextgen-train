from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
DATASET_PATH = "/data/combined_questions_answers"  # Path to the dataset containing "text1" and "text2" columns
OUTPUT_PATH = "/data/processed/mined_negatives"  # Path to save the dataset with mined hard negatives

model = SentenceTransformer(MODEL_NAME)

dataset = load_from_disk(DATASET_PATH)

dataset = mine_hard_negatives(
    dataset=dataset,
    model=model,
    relative_margin=0.05,  # 0.05 means that the negative is at most 95% as similar to the anchor as the positive
    num_negatives=3,  # 10 or less is recommended
    sampling_strategy="top",  # "top" means that we sample the top candidates as negatives
    batch_size=256,  # Adjust as neede
    use_faiss=True,  # Optional: Use faiss/faiss-gpu for faster similarity search
    # use_multi_process=["cuda:5", "cuda:0", "cuda:1", "cuda:2"]
    range_max=200,
)

dataset.save_to_disk(OUTPUT_PATH)
