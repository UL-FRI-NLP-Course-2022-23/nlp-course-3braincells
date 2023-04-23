# Natural language processing course 2022/23: Literacy situation models knowledge base creation

# Team members:
 * Radoslav Atanasoski, 63190355, ra9902@student.uni-lj.si
 * Mila Marinković, 63170369, mm9136@student.uni-lj.si
 * Ilina Kirovska, 63170366, ik8739@student.uni-lj.si
 
<!-- Group public acronym/name: burek
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues. -->
 
# Prerequisites

 ```conda create -n nlp python=3.7.4```
 
 ```conda activate nlp```
 
 ```git clone https://github.com/UL-FRI-NLP-Course-2022-23/nlp-course-3braincells.git```
 
 ```cd nlp-course-3braincells```
 
 ```pip install -r requirements.txt  ```
 
 ##### download spacy:
 ```python -m spacy download en_core_web_sm```
 
 ##### run stanza and ntlk:
 ```python run_models.py```
 
 ##### StanfordCoreNlp for coreference resolution:
 
  1. download StanfordCoreNlp from site inside of the project: 
  
   ```https://stanfordnlp.github.io/CoreNLP/download.html?fbclid=IwAR2QLGJV0LcmgG-IEH7lebtiwNVlRnxR9FJdq8EY9oE71l3oFj_ha_lPzwM ```
  
  2. install package: ```unzip stanford-corenlp-4.5.4.zip```
  
  3. start server: 
  
  ```cd stanford-corenlp-4.5.4```

  ```java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 75000```
 
 4. add python package ```pycorenlp``` in requirements.txt and reinstall requirements.txt
