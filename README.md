# NLP_Assignment_Sentiment_Analysis
NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis
## Team member
- Xiaoyan Hong, xiaoyan.hong@student-cs.fr
- Felipe Garavano, felipe.garavano@student-cs.fr
- Ruining Ma, ruining.ma@student-cs.fr
- Victor Hong, victor.hong@student-cs.fr

## Pre-required version
- pytorch = 1.13.1
- pytorch-lightning = 1.8.1 (not used)
- transformers = 4.22.2
- datasets = 2.9.0 (just the library ‘datasets’, no labelled data)
- sentencepiece = 0.1.97 (not used)
- scikit-learn = 1.2.0
- numpy = 1.23.5
- pandas = 1.5.3
- nltk = 3.8.1
- stanza = 1.4.2 (not used)

- GPU is strongly recommended to run the codes

## Data Preprocessing
In the beginning, we tried several classic preprocessing method for NLP tasks such as lowering case, removing the stop words or lemmatization, but in fact it performed better without any preprocessing for the text input for the transformaer based model, like BERT. The model itself wad designed to handle original texts, the traditional proprocessing methods may reduce the contexts which can be useful.

We used label encoding for the polarity labels: **0: 'positive, 1: 'neutral', 2: 'negative'**.

For the BERT pretrained model 'bert-base-uncased', we designed the input as **'[CLS] ' + aspect_category + ' [SEP] ' + sentence_left + ' [SEP] ' + target + ' [SEP] ' + sentence_right + ' [SEP] '**. The idea was inspired by some recent research paper[2] and is quite intuitive too. We separate the aspect term from the sentence to stress it for the aspect based sentiment analysis task. In addition, we also transformed the category into understandable questions to make it more semantic[5].

For the RoBERTa pretrained model 'roberta-base', the text input is similar to BERT, except that we removed the category questions as it didn't improve the model results.

## Training Efficiency:
To further enhance training efficiency and reduce the training time while maintaining strong performance, we implemented **mixed precision training and gradient accumulation techniques**. Mixed precision training leverages both single-precision (FP32) and half-precision (FP16) formats during the training process. Do note that we are using autocast which automatically selects the best data type for each operation, meaning it only uses half-precision when it is safe to do so. Also the weight updates, however, are still performed in single-precision to maintain model accuracy and stability. For devices without mixed precision support (e.g., CPU), a dummy autocast context manager is defined to maintain compatibility.

Gradient accumulation is another technique used to enable training with larger batch sizes. This method involves accumulating gradients from several mini-batches before performing a single weight update. By doing so, we can effectively utilize larger batch sizes without exhausting GPU memory, leading to more stable training and faster convergence.

Implementing mixed precision training and gradient accumulation, in combination with our existing strategies such as pre-trained models, adaptive batch size, and linear scheduler, further improves training efficiency. This approach allows us to maintain high performance and complete model training within **5 minutes per run**, striking a balance between speed and accuracy.

## Model Selection
After reading some research paper, we found the transformer based deep learning model significantly aced compared to classic machine learning model or statistical model. Therefore we selected BERT('bert-base-uncased') as the baseline and found RoBERTa('roberta-base') which was an improved version of BERT afterwards.

For the model training, we run 15 epochs for each model and select the best model according to the devdata accuracy (if not available, use validation set instead). We also defined a function to calculate the maximum length for tokens. The batch size was set as 32 and we used a linear scheduler to adjust the learning rate automatically. As it is shown in the EDA phase, we found the distributions of polarity labels in both the traindata and devdata are **imbalanced** (positive:0.70, negative:0.26, neutral:0.04). Therefore, we also set the **inverse ratio of [1.43, 25.00, 3.85]** as class weights and passed them as a parameter to the CrossEntropyLoss loss function to alleviate the influence.

To use these Large Language Models(LLMs), we imported their respective tokenizers directly. Then taking RoBERTa as an example, we defined the customized classifier by importing the RobertaModel as base layer and then add a classifier layer which take the architecture of the predefined classifier RobertaForSequenceClassification as a reference. 

Besides BERT base and RoBERTa base, we also tried larger LLMs like RoBERTa large which is trained on a much larger corpus of text data. According to huggingface, the Roberta base has 125 million parameters, while Roberta large has 355 million parameters. With larger LLMs, we were able to reach a higher accuracy 89 on average, sometimes even 91. However RoBERTa large also required more computing resources and about 3 times slower then RoBERTa. Therefore, we decided to choose RoBERTa base in the end considering all the pros and cons.

## Final Performance
The performance metric is accuracy, and the model training process should be able to finished around 5 min per run.

**On devdata(on average):**
- BERT(base): 86.97 
- **RoBERTa(base): 88.62 (selected)**
- RoBERTa(large): 89.09 (sometimes 91)

## References
- [1] Devlin, Jacob, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” ArXiv abs/1810.04805 (2019): n. pag.
- [2] Zhuang Chen and Tieyun Qian. 2019. Transfer Capsule Network for Aspect Level Sentiment Classification. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 547–556, Florence, Italy. Association for Computational Linguistics.
- [3] Tang, Duyu, Bing Qin, Xiaocheng Feng and Ting Liu. “Effective LSTMs for Target-Dependent Sentiment Classification.” International Conference on Computational Linguistics (2015).
- [4] Liu, Yinhan, Ott, Myle, Goyal, Naman, Du, Jingfei, Joshi, Mandar, Chen, Danqi, Levy, Omer, Lewis, Mike, Zettlemoyer, Luke and Stoyanov, Veselin RoBERTa: A Robustly Optimized BERT Pretraining Approach. (2019). , cite arxiv:1907.11692 .
- [5] Sun, Chi, Luyao Huang and Xipeng Qiu. “Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence.” North American Chapter of the Association for Computational Linguistics (2019).
