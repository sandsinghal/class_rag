# Efficient Query Classsification for Adaptive RAG Systems
** DISCLAIMER : We do not claim Adaptive RAG is our work. Our work builds on top of it and tries to improve the classfier part of the work. The original work can be found here ["Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"](https://arxiv.org/pdf/2403.14403.pdf).

You can use our finetuned models available here. [Models](https://drive.google.com/drive/folders/1keXs-QDbTe1Q1DznngIvItBpBkRkHv5t?usp=sharing)

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge retrieval, improving factual accuracy and reducing hallucinations. However, existing RAG systems struggle with fixed retrieval strategies, leading to inefficiencies in handling queries of varying complexity. To address this, we propose a distillation-based classifier for adaptive prompt tuning in RAG systems. Our approach involves fine-tuning a lightweight model, TinyBERT, for query classification. This classifier dynamically selects the optimal retrieval strategy—no retrieval, single retrieval, or multi-retrieval—based on query complexity, ensuring a balance between efficiency and accuracy. We build upon Adaptive RAG methodology, improving classification performance while reducing computational overhead. This report details our methodology, data preparation, fine-tuning and how Tiny-BERT is able to achieve a comparable performance to Adaptive-RAG while reducing the classifier parameter count by 50 times.

## Installation
The first step (to run our Adaptive-RAG) is to create a conda environment as follows:
```bash
$ conda create -n adaptiverag python=3.8
$ conda activate adaptiverag
$ pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install -r requirements.txt
```

## Prepare Retriever Server
After installing the conda environment, you should setup the retriever server as follows:
```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
bash run_elastic.sh
```

Start the elasticsearch server on port 9200 (default), and then start the retriever server as shown below.
```bash
bash run_retriever.sh
```

## Datasets
* You can download multi-hop datasets (MuSiQue, HotpotQA, and 2WikiMultiHopQA) from https://github.com/StonyBrookNLP/ircot.
```bash
bash ./download/raw_data.sh
```
We provide the preprocessed datasets in [`processed_data.tar.gz`](./processed_data.tar.gz).

## Prepare LLM Server
After indexing for retrieval is done, you can verify the number of indexed documents in each of the four indices by executing the following command in your terminal: `curl localhost:9200/_cat/indices`. You should have 4 indices and expect to see the following sizes: HotpotQA (5,233,329), 2WikiMultihopQA (430,225), MuSiQue (139,416), and Wiki (21,015,324).

Next, if you want to use FLAN-T5 series models, start the llm_server (for flan-t5-xl and xxl) by running:
```bash
MODEL_NAME={model_name} uvicorn serve:app --port 8010 --app-dir llm_server # model_name: flan-t5-xxl, flan-t5-xl
```

We provide all the results for three different retrieval strategies in [`predictions.tar.gz`](./predictions.tar.gz).

## Classified Data for the process
We provide the datasets for the classifier in [`data.tar.gz`](./data.tar.gz).

Now, you are ready with the training dataset for the classifier. 
You can train any encoder model available on hugging face by editing train_tiny_bert.sh.
```
MODEL=intfloat/e5-small-v2 # Model name on HuggingFace
MODEL_NAME=e5_small_v2 # Used to name the output folder
LLM_NAME=flan_t5_xl   # Simply used to identify the inference LLM does not have any effect on training
```
```bash
# Train the classifiers!
bash run_training.sh

# Create the file containing the test set queries, each labeled with its classified query complexity.
# Additionally, this outputs the step efficiency for each dataset.
cd ..
python ./classifier/postprocess/predict_complexity_on_classification_results.py flan_t5_xl
```

Finally, you are able to evaluate the QA performance of our Adaptive-RAG (based on the identified query complexity results) with the following code!
```bash
python ./evaluate_final_acc.py
```

## Acknowledgement
We refer to the repository of [Adaptive RAG](https://github.com/starsuzi/Adaptive-RAG) as a skeleton code.

## Citation
If you found the provided code with our paper useful, we kindly request that you cite our work.
```BibTex
@inproceedings{jeong2024adaptiverag,
  author       = {Soyeong Jeong and
                  Jinheon Baek and
                  Sukmin Cho and
                  Sung Ju Hwang and
                  Jong Park},
  title        = {Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity},
  booktitle={NAACL},
  year={2024},
  url={https://arxiv.org/abs/2403.14403}
}
```
