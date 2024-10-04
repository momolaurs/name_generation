# Generating diachronic Spanish names



This project is structured in two steps: obtaining the data and training the character-based language model on it.
The script for obtaining and processing the dataset of old names can be found in `for_name_extraction` directory, while the language model is in `main`. 

If you already have a `.txt` file with all the names from an older period, the first step can be skipped.

**Installing the requirements** 



1.To run the `name_extraction.py` scripts while extracting the data, you will need the following:



```

*spacy==3.7.4

*spaCy trained pipeline: es_core_news_lg

*tqdm==4.66.2

*pandas==2.2.1

```

2.To run the language model script, you will need the following:


```

*gdown==5.1.0

*matplotlib==3.8.3

*torch==2.2.1

*pandas==2.2.1

```

3.If working locally, you can simply install the `0_requirements.txt` file using the command `pip install -r 0_requirements.txt` in your terminal. There are two separate `0_requirements.txt` files depending on which step you want to implement.

### A note on running `name_extraction.py`

A few output files are created throughout the execution of the script. To ensure that the unnecessary ones are cleaned up at the end of the process, name all intermediate outputs with the `temp_` prefix. The only output file we will need for training the language model is the output of the `old_names_final` function.



### Code and data attribution

The code used in this project is based on Karpathy's [makemore](https://github.com/karpathy/makemore).

The modern Spanish names database used in our code was retrieved from the [INE](https://www.ine.es/uc/nDER2igi) (names with frequency of 20 or higher).

The corpus we used is thanks to [Calvo Tello's novel compilation](https://github.com/cligs/textbox/tree/master/spanish/novela-espanola).

Likewise, the list of words in the Diccionario de la Lengua Española dictionary was made available by [Dueñas Lerín](https://github.com/JorgeDuenasLerin/diccionario-espanol-txt/blob/master/0_palabras_todas.txt). 



### Contact

If you encounter any issues, you can contact us at angelagomezzuluaga@gmail.com and momolaurs@gmail.com. Have fun! 

