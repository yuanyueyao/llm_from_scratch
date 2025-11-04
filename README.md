# LLM_from_scratch

Build a Large Language Model (LLM) completely from scratch! 😄


## Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/yuanyueyao/llm_from_scratch
    cd llm_from_scratch
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Look at the scripts folder and run your desired script. For example, to train the model:
    ```bash
    python scripts/train.py
    ```
    
## ✅ Todo

### Tokenizer
- [x] Implement a Byte Pair Encoding (BPE) tokenizer  
  - [x] Train the tokenizer on a corpus  
  - [x] Save and load the tokenizer  
  - [x] Encode and decode text  
  - [x] Validate the tokenizer  

### Model architecture
- [x] Build a Transformer model  
  - [x] Implement the soft_max
  - [x] Implement the attention mechanism  
  - [x] Implement the feed-forward network  
  - [x] Implement positional encoding  
  - [x] Implement layer normalization  

### Train
- [x] Train the model on a dataset 
  - [x] Implement the cross_entropy loss function
  - [x] Implement the Adam optimizer
  - [x] Set up the training loop
  - [x] Monitor training progress

### Test
- [ ] Evaluate the model's performance


