---
layout:     post
title:      "Langchain Integrations for BigDL"
subtitle:   " \"Hello World, Hello Blog\""
date:       2023-08-27 21:00:00
author:     "Potter"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - LLM
    - Tutorial
---

# Langchain Integrations 

[LangChain](https://python.langchain.com/docs/get_started/introduction.html) is a popular library for developing applications powered by language models. You can use LangChain with LLMs to build various interesting applications such as [Chatbot](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/langchain/transformers_int4/chat.py), [Document Q&A](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/langchain/transformers_int4/docqa.py), [voice assistant](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/langchain/transformers_int4/voiceassistant.py). BigDL-LLM provides LangChain integrations (i.e. LLM wrappers and embeddings) and you can use them the same way as [other LLM wrappers in LangChain](https://python.langchain.com/docs/integrations/llms/). 

This notebook goes over how to use langchain to interact with BigDL-LLM.

## Installation

First of all, install BigDL-LLM in your prepared environment. For best practices of environment setup, refer to [Chapter 2]() in this tutorial.


```python
!pip install bigdl-llm[all]
```

Then install LangChain.


```python
!pip install -U langchain==0.0.248
```

> **Note**
> 
> We recommend to use `langchain==0.0.248`, which is verified in our tutorial.

## LLM Wrapper

BigDL-LLM provides `TransformersLLM` and `TransformersPipelineLLM`, which implement the standard interface of LLM wrapper of LangChain.

`TransformerLLM` can be instantiated using `TransformerLLM.from_model_id` from a huggingface model_id or path. Model generation related parameters (e.g. `temperature`, `max_length`) can be passed in as a dictionary in `model_kwargs`. Let's use [`vicuna-7b-v1.5`](https://huggingface.co/lmsys/vicuna-7b-v1.5) model as an example to instatiate `TransformerLLM`.



```python
from bigdl.llm.langchain.llms import TransformersLLM

llm = TransformersLLM.from_model_id(
        model_id="lmsys/vicuna-7b-v1.5",
        model_kwargs={"temperature": 0, "max_length": 1024, "trust_remote_code": True},
    )
```

> **Note**
>
> `TransformersPipelineLLM` can be instantiated in similar way as `TransformersLLM` from a huggingface model_id or path, `model_kwargs` and `pipeline_kwargs`. Besides, there's an extra `task` parameter which specifies the type of task to perform.

Use a prompt template to format the prompt and simply call `llm` to test generation.

> **Note**
>
> `max_new_tokens` parameter defines the maximum number of tokens to generate.


```python
prompt = "What is AI?"
VICUNA_PROMPT_TEMPLATE = "USER: {prompt}\nASSISTANT:"
result = llm(prompt=VICUNA_PROMPT_TEMPLATE.format(prompt=prompt), max_new_tokens=128)
```

    AI stands for "Artificial Intelligence." It refers to the development of computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI can be achieved through a combination of techniques such as machine learning, natural language processing, computer vision, and robotics. The ultimate goal of AI research is to create machines that can think and learn like humans, and can even exceed human capabilities in certain areas.


You can also use `generate` on LLM to get batch results.


```python
llm_result = llm.generate([VICUNA_PROMPT_TEMPLATE.format(prompt="Tell me a joke"), VICUNA_PROMPT_TEMPLATE.format(prompt="Tell me a poem")]*3)
```


```python
print("-"*20+"number of generations"+"-"*20)
print(len(llm_result.generations))
print("-"*20+"the first generation"+"-"*20)
print(llm_result.generations[0][0].text)
```

    --------------------number of generations--------------------
    6
    --------------------the first generation--------------------
    USER: Tell me a joke
    ASSISTANT: Why did the tomato turn red?
    
    Because it saw the salad dressing!


## Using Chains

Now let's begin using LLM wrappers and embeddings in [Chains](https://docs.langchain.com/docs/components/chains/).

>**Note**
> Chain is an important component in LangChain, which combines a sequence of modular components (even other chains) to achieve a particular purpose. The compoents in chain may be propmt templates, models, memory buffers, etc. 

### LLMChain

Let's first try use a simple chain `LLMChain`. 

Create a simple prompt template as below. 


```python
from langchain import PromptTemplate

template ="USER: {question}\nASSISTANT:"
prompt = PromptTemplate(template=template, input_variables=["question"])
```

Now use the `llm` we created in previous section and the prompt tempate we just created to instantiate a `LLMChain`. 


```python
from langchain import LLMChain

llm_chain = LLMChain(prompt=prompt, llm=llm)
```

Now let's ask the llm a question and get the response by calling `run` on `LLMChain`.


```python
question = "What is AI?"
result = llm_chain.run(question)
```

    AI stands for "Artificial Intelligence." It refers to the development of computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI can be achieved through a combination of techniques such as machine learning, natural language processing, computer vision, and robotics. The ultimate goal of AI research is to create machines that can think and learn like humans, and can even exceed human capabilities in certain areas.


### Conversation Chain

To build a chat application, we can use a more complex chain with memory buffers to remember the chat history. This is useful to enable multi-turn chat experience.


```python
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

template = "The following is a friendly conversation between a human and an AI.\
    The AI is talkative and provides lots of specific details from its context.\
    If the AI does not know the answer to a question, it truthfully says it does not know.\
    \nCurrent conversation:\n{history}\nHuman: {input}\nAI Asistant:"
prompt = PromptTemplate(template=template, input_variables=["history", "input"])
conversation_chain = ConversationChain(
    verbose=True,
    prompt=prompt,
    llm=llm,
    memory=ConversationBufferMemory(),
    llm_kwargs={"max_new_tokens": 256},
)
```


```python
query ="Good morning AI!" 
result = conversation_chain.run(query)
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI.    The AI is talkative and provides lots of specific details from its context.    If the AI does not know the answer to a question, it truthfully says it does not know.    
    Current conversation:
    
    Human: Good morning AI!
    AI Asistant:[0m
    Good morning! How can I assist you today?
    
    [1m> Finished chain.[0m



```python
query ="Tell me about Intel." 
result = conversation_chain.run(query)
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI.    The AI is talkative and provides lots of specific details from its context.    If the AI does not know the answer to a question, it truthfully says it does not know.    
    Current conversation:
    Human: Good morning AI!
    AI: The following is a friendly conversation between a human and an AI.    The AI is talkative and provides lots of specific details from its context.    If the AI does not know the answer to a question, it truthfully says it does not know.    
    Current conversation:
    
    Human: Good morning AI!
    AI Asistant: Good morning! How can I assist you today?
    Human: Tell me about Intel.
    AI Asistant:[0m
    Intel is a multinational technology company that specializes in the development and manufacturing of computer processors and related technologies. It was founded in 1976 by Robert Noyce and Gordon Moore, and is headquartered in Santa Clara, California. Intel's processors are used in a wide range of devices, including personal computers, servers, smartphones, and other electronic devices. The company is also involved in the development of new technologies, such as artificial intelligence and autonomous driving.
    
    [1m> Finished chain.[0m


### MathChain

Let's try use LLM solve some math problem, using `MathChain`.

> **Note** 
> MathChain usually need LLMs to be instantiated with larger `max_length`, e.g. 1024



```python
from langchain.chains import LLMMathChain

MATH_CHAIN_TEMPLATE ="Question: {question}\nAnswer:"
prompt = PromptTemplate(template=MATH_CHAIN_TEMPLATE, input_variables=["question"])
llm_math = LLMMathChain.from_llm(prompt=prompt, llm=llm, verbose=True)
```


```python
question = "What is 13 raised to the 2 power"
llm_math.run(question)
```

    
    
    [1m> Entering new LLMMathChain chain...[0m
    What is 13 raised to the 2 power13 raised to the 2 power is equal to 13 \* 13, which is 169.
    [32;1m[1;3mQuestion: What is 13 raised to the 2 power
    Answer: 13 raised to the 2 power is equal to 13 \* 13, which is 169.[0m
    [1m> Finished chain.[0m





    'Answer:  13 raised to the 2 power is equal to 13 \\* 13, which is 169.'



### Question Answering over Docs
Suppose you have some text documents (PDF, blog, Notion pages, etc.) and want to ask questions related to the contents of those documents. LLMs, given their proficiency in understanding text, are a great tool for this.

#### Installation


```python
pip install -U faiss-cpu
```

#### Load document
For convienence, here we use a text string as a loaded document.


```python
input_doc = "\
    BigDL: fast, distributed, secure AI for Big Data\
    BigDL seamlessly scales your data analytics & AI applications from laptop to cloud, with the following libraries:\
        Orca: Distributed Big Data & AI (TF & PyTorch) Pipeline on Spark and Ray\
        Nano: Transparent Acceleration of Tensorflow & PyTorch Programs on XPU\
        DLlib: “Equivalent of Spark MLlib” for Deep Learning\
        Chronos: Scalable Time Series Analysis using AutoML\
        Friesian: End-to-End Recommendation Systems\
        PPML: Secure Big Data and AI (with SGX Hardware Security)\
        LLM: A library for running large language models with very low latency using low-precision techniques on Intel platforms\
    "
```

#### Split texts of input document


```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(input_doc)
```

#### Create embeddings and store into vectordb

BigDL-LLM provides `TransformersEmbeddings`, which allows you to obtain embeddings from text input using LLM.

`TransformersEmbeddings` can be instantiated the similar way as `TransformersLLM`


```python
from bigdl.llm.langchain.embeddings import TransformersEmbeddings

embeddings = TransformersEmbeddings.from_model_id(model_id="lmsys/vicuna-7b-v1.5")
```

After introducing `TransformersEmbeddings`, let's create embeddings and store into vectordb


```python
from langchain.vectorstores import FAISS

docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
```

#### Get relavant texts


```python
query = "What is BigDL?"
docs = docsearch.get_relevant_documents(query)
print("-"*20+"number of relevant documents"+"-"*20)
print(len(docs))
```

    --------------------number of relevant documents--------------------
    1


#### Prepare chain


```python
from langchain.chains.chat_vector_db.prompts import QA_PROMPT
from langchain.chains.question_answering import load_qa_chain

doc_chain = load_qa_chain(
    llm, chain_type="stuff", prompt=QA_PROMPT
)
```

#### Generate


```python
result = doc_chain.run(input_documents=docs, question=query)
```

    BigDL is a fast, distributed, and secure AI library for Big Data. It enables the seamless scaling of data analytics and AI applications from laptops to the cloud. BigDL provides a range of libraries, including Orca, Nano, DLlib, Chronos, Friesian, PPML, and LLM, to support various AI and data analysis tasks.

