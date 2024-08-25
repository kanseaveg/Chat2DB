# Chat2DB
## Pay Attention
### Live demo
You can access the live demo version of Chat2DB by visiting [8.138.157.109](http://8.138.157.109/).
Please be aware that due to the high cost of GPU, the live website currently only supports engines based on deepseek for LLM and t5-base for PLM. If you wish to experience the full version, we have provided **a comprehensive multi-engine Chat2DB framework code** in our repository. If you wish to experience the full version's features, please prepare an A800 GPU locally and deploy it according to the following instructions.


### Video demo
You can watch the live video of Chat2DB by visiting [Chat2DB-demonstration](https://vimeo.com/995253448)

### Preview
![image](https://github.com/user-attachments/assets/3af7283d-e5e5-48c7-bd26-fa5fa05d3910)


## Overview
### Description
Code implementation for the paper Chat2DB: Chatting to the Database with Interactive Agent Assisted Language Models.


### Prerequisites
Create a virtual anaconda environment and install the required modules and tools:
```shell
conda create -n chat2db python=3.10
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
python nltk_downloader.py
```
Create several folders:
```shell
mkdir checkpoints
mkdir data
mkdir logs
```


### Prepare Data
Download the data [here](https://drive.google.com/drive/folders/1zIk7fSMDb8aARBz_wOttxTRBiUTBEwjI?usp=sharing) and unzip them in the data/ folder as follows:
```
data/
├── spider/
│   ├── database/
│   ├── ...
|   └── tables.json
└── usr/
    └── database/
```


### Prepare Model
1. Download the [Embeddings Models](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) and put it into the checkpoint folders.
2. Download the [Schema Filter Models](https://drive.google.com/file/d/1zHAhECq1uGPR9Rt1EDsTai1LbRx0jYIo/view) and put it into the checkpoint folders.
3. For the parsers, due to our company's privacy protection policies, we have made some publicly available parsers available for download below. We also welcome you to use your own trained parsers.

    | Parsers | Checkpoint Links |
    | ------  | ---------------- |
    | t5-3b   | [text2sql-t5-3b](https://drive.google.com/file/d/1M-zVeB6TKrvcIzaH8vHBIKeWqPn95i11/view) |
    | t5-large | [text2sql-t5-large](https://drive.google.com/file/d/1-xwtKwfJZSrmJrU-_Xdkx1kPuZao7r7e/view) |
    | t5-base | [text2sql-t5-base](https://drive.google.com/file/d/1M-zVeB6TKrvcIzaH8vHBIKeWqPn95i11/view) |
    | deepseek-6.7b | [deepsql-6.7b](https://huggingface.co/MrezaPRZ/DeepSQL_BIRD) |


After download these checkpoints, you need to change the `settings.json` and `modelend/utils/args.py` to adapt these parsers. Please pay special attention to the folder names of the checkpoint which **must match** those in the configuration file.


### Deploy & Serve
Grant the execution permission to the script.
```shell
chmod +x ./run_all.sh
```

Starting the `Chat2DB` Service.
```shell
./run_all.sh start
```
At this moment, the backend server, model side, and frontend visualization interface of Chat2DB will each start in the background and listen on ports `8299`, `8091`, and `8080` respectively. Of course, you can modify these in `settings.py` and `run_all.sh`.

You will find three service output logs in the `logs` folder, which is organized as follows. You can monitor your services by observing the log outputs.
```
logs/
├── backend.log
├── frontend.log
└── modelend.log

```

Stopping all the `Chat2DB` Service:
```shell
./run_all.sh stop
```

### Acknowledgements
We would thanks to RESDSQL(paper, code), DTS-SQL(paper, code), Spider(paper, dataset) for their interesting work and open-sourced code and dataset.
