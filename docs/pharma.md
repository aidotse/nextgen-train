# Pharmaceutical Question-Answering Model Training

This repository contains scripts to train / fine-tune sentence-transformer models on various pharmaceutical QA datasets. The idea is to fine-tune a specialist embedding model to embed pharmaceutical pamphlet data, and to leverage this as part of a wider RAG setting for retrieval tasks. We wish to evaluate the degree to which specializing the embedding model itself can lead to improved performance of a Q/A system (i.e. chatbot) for downstream users.

For more details on the use case, the data, training, evaluation, results, contrastive learning, etc. see [the official documentation here.](https://aidotse.github.io/nextgen-report/#use-case-rag-for-medical-leaflets-astrazeneca)
