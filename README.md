# Natural language processing course 2022/23: Literacy situation models knowledge base creation

Short stories are powerful mediums that in spite of their length often convey complex themes and ideas. Understanding them demands a thorough grasp of the context in which they were written. The purpose of this project is to help the readers in bettering their understanding and knowledge of the stories by building a knowledge base. Using NLP techniques such as named-entity extraction and sentiment analysis we managed to identify the stories' character relationships and their sentiments as well as identify the protagonists and antagonists. Finally, we have made graphs to analyze the relations and sentiments of communicators from selected stories.

<!--
# Team members:
 * Radoslav Atanasoski, 63190355, ra9902@student.uni-lj.si
 * Mila Marinković, 63170369, mm9136@student.uni-lj.si
 * Ilina Kirovska, 63170366, ik8739@student.uni-lj.si -->
 
<!-- Group public acronym/name: burek
 > This value will be used for publishing marks/scores. It will be known only to you and not your colleagues. -->
 
## Dataset

Data includes stories, that we acquired from the Project Gutenberg website - a digital library containing a large number of eBooks. More precisely, we decided for [Grimm's Fairy Tales](https://www.gutenberg.org/ebooks/2591) stories collection.  These fairy tales and folklore stories typically involve magic, enchantments, and fictitious or mythical creatures. Firstly we collected a dozen stories, that we manually annotated and saved this stories including their annotations in `data/our_data/`.

The annotation file of each story counts three fields:
* Characters - contains the story characters
* Relationships - contains the relationships between related characters and sentiment evaluation of each relationship marked as -1, 0, and 1
* Protagonist and antagonist - the story's protagonist and antagonist, if there is one


## Prerequisites

1. Create and activate the Anaconda environment. 

 ```
 conda create -name=nlp python=3.7.4
 ```
 ```
 conda activate name=nlp
 ```
 2. Clone this project's git repository and navigate to the code directory.
 ```
 git clone https://github.com/UL-FRI-NLP-Course-2022-23/nlp-course-3braincells.git
 ```
 ```
 cd nlp-course-3braincells
 ```
 3. Install necessary dependencies.
 ```
 pip install -r requirements.txt 
 ```
 
 4. Download Spacy model.
 
 ```
 python -m spacy download en_core_web_sm
 ```
 
5. Run script to install Stanza model and ntlk toolkit.
 
 ```
 python run_models.py
 ```

## Running the code

When running `Pipeline.py` script, located in `src`, we are performing the whole project pipeline that consists of the
following stages: character extraction, coreference resolution, sentiment analysis, and visualization.

In order to run the script, you need to provide a path to the story (txt file) as an argument.

Example:

```
 python Pipeline.py --path "data/Cinderella.txt"
 ```

There is an option inside the script to choose whether you would like to calculate sentiment with coreference resolution being done first or not. Furthermore, you can manually pass the function which model (affin or stanza) would you like to use in order to evaluate sentiment as well as the offset - the range of sentences you would like to consider. 

A more detailed project description and results analysis can be found in provided [PDF](report/NLP_Report_Submission2.pdf).



