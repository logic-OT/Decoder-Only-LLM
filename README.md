# Decoder-Only-LLM With Pytorch ✨

This repository features a custom-built decoder-only language model (LLM) with 8 decoders and a total of 37 million parameters. I developed this from scratch as a personal project to explore NLP techniques ***which you can now pretrain to fit whatever task***

![image](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff6133c18-bfaf-4578-8c5a-e5ac7809f65b_1632x784.png)

## 📖 Notebook Experiments
If you wish to run the code sequentially to see how everything works check out the [Decoder-Only transformer](https://github.com/logic-OT/Decoder-Only-LLM/blob/main/Decoder-only%20transformer.ipynb) notebook in the repo where I train the model to ask a question from a given context.

# 🔧 Setup
- Clone repository
```
git clone git@github.com:logic-ot/decoder-only-llm.git
cd decoder-only-llm
```
- **Configuration file**
  - Configure certain variables from the [config.py]() file such as dataset path, etc. <br><br>
  **Structure of config.py:**<br><br>

  ```python
  
  config = {
      "data_path":"sample_data.json", #path to dataset
      "tokenizer": AutoTokenizer.from_pretrained('bert-base-uncased'),
      "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
      "model_weights": None, #path to model weights
      "learning_rate":0.0001,
      "num_batches":10,
  }

- **Training**
  - After configuration, run the training script like so: <br><br>
  ```
  python training.py
  ```
  The model should automatically begin training

- **Data Structure**
  - Your data for **training** must be in the following format:<br><br>
    ```python
    ["conversation":[{"from":"human","value":"{some text}"},{"from":"gpt","value":"{some text}"}]
    ```
    See an example in [sample_data.json](https://github.com/logic-OT/Decoder-Only-LLM/blob/main/sample_data.json) in the repo
    
- **Inferencing**
  - The inference script takes in a text and the queries the model for a response. To inference, run:<br><br>
  
  ```
  python inference.py
  ```
## 🪄 Tips
- I have a [what-to-expect.pdf](https://github.com/logic-OT/Decoder-Only-LLM/blob/main/what-to-expect.pdf)  file in the repo that details some observations i made while training the model for the question asking task. This information applies to any other model you decide to pretrain.
- Here is an excellent video that explains tranformers: [3 blue blue 1 brown](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=1046s)

