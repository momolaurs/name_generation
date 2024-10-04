"""
NLP Assignment 2: Character-based LM for name generation - Extracting names

Authors: Ángela María Gómez Zuluaga and Laura Moset Estruch

Original file is located at
    https://colab.research.google.com/drive/1JTfJjInX0QTWc2vadwJ5tQRu5D0AvjLs

"""

#import required libraries
from tqdm import tqdm
import spacy
import os
import pandas as pd

def extract_names_ner(input_file, output_file, chunk_size=1000000):

    """
    Extracts Spanish person names from a corpus file using spaCy.

    Parameters:
    - input_file (str): Path to the input corpus file.
    - output_file (str): Path to the output file to save the extracted names.
    - chunk_size (int): Size of chunks to process the input file (spaCy limit is set to 1,000,000).
    """

    # Load the Spanish language model and disable unnecessary components
    nlp = spacy.load("es_core_news_lg")
    all_names = []

    # Read the content from the input file
    with open(input_file, 'r', encoding='utf-8') as file:

        print("Extracting names...\n")
        #Progress bar
        total_size = os.path.getsize(input_file)  #Obtain total size
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing", position = 0) #Initialize

        #Read the file in chunks
        chunk = file.read(chunk_size)
        while chunk:
            doc = nlp(chunk) #Process chunk with spaCy
            names = [ent.text for ent in doc.ents if ent.label_ == "PER"] # Extract person names using spaCy NER

            # Add names from the current chunk to the list
            all_names.extend(names)

            # Update the progress bar
            progress_bar.update(len(chunk))

            # Read the next chunk
            chunk = file.read(chunk_size)

    # Close the progress bar
    progress_bar.close()
    print("\nDone!")

    # Print a sample of extracted names for verifying correct extraction
    print("\nSample of extracted names:")
    for name in all_names[-5:]:
        print(name)

    # Save names to the output file
    with open(output_file, 'w') as output_file:
        for name in all_names:
            output_file.write(name + '\n') #add newlines between each extracted name

extract_names_ner('old_corpus.txt', 'temp_old_names.txt')

all_names = []

#Make second pass with the txt file generated in the previous step

def ner_for_persons(input_file, output_file):
    print("\nSecond pass...")
    # Load the Spanish language model
    nlp = spacy.load("es_core_news_lg", disable=["tagger", "parser"])  # Disable unnecessary components for speed

    # Read the content from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Process with the spaCy pipeline
    doc = nlp(text)

    # Extract persons (PER) using NER
    persons = [ent.text for ent in doc.ents if ent.label_ == "PER"]
    all_names.extend(persons)

    # Save the names to the output file
    with open(output_file, 'w') as output_file:
        for name in all_names:
            output_file.write(name + '\n')

ner_for_persons('temp_old_names.txt', 'temp_old_names.txt')

def clean_names(input_file, output_file):
    print("\nCreating a cleaner set of names...\n")

    # Read the list of names from the file
    with open(input_file, 'r', encoding='utf-8') as file:
        names = [line.strip() for line in file if line.strip()]
    # Remove duplicates
    unique_names = list(set(names))

    # Write the names back to the output file
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for name in unique_names:
            output_file.write(name + '\n')

    return print(f"Extracted names have been saved in your directory as '{output_file.name}'")

clean_names('temp_old_names.txt', 'temp_old_names.txt')

def read_names(sheet_name):
    """
    Reads names from an Excel file

    Parameters:
    - sheet_name (str): The sheet name to read from the Excel file.

    Returns:
    - list: A list of names in lowercase.
    """
    #Read in Excel file
    df = pd.read_excel('nombres_por_edad_media.xls', sheet_name=sheet_name, skiprows=6)

    #Convert to lowercase and return as list
    return [str(name).lower() for name in df['Nombre'].tolist()]

names_modern = read_names('Hombres') + read_names('Mujeres')

def filter_names(input_file, output_file, filter_list):
    """
    Filters lines from an input file based on a list of names.

    Parameters:
    - input_file (str): Path to the input text file to filter.
    - output_file (str): Path to the output file to save the filtered lines.
    - filter (list): List of names/words to use for filtering.

    """
    # Read the lines from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]

    total_lines = len(lines)

    print("\nFiltering dictionary words to remove names...\n")

    # Initialize the progress bar
    progress_bar = tqdm(total=total_lines, unit='line', desc="Filtering", position = 0)

    # Exclude lines that match the names from filter file (names_modern)
    filtered_lines = []
    for idx, line in enumerate(lines):
        if line.lower() not in filter_list: #check whether the line is not present in input file
            filtered_lines.append(line)

        # Update the progress bar every 10%
        if (idx + 1) % (total_lines // 10) == 0:
          progress_bar.update(total_lines // 10)

    # Close the progress bar
    progress_bar.close()
    print("\nDone!")

    # Write the filtered lines back to the output file
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for line in filtered_lines:
            output_file.write(line + '\n')

    #Return path to output file
    return print(f"Filtered output has been saved in your directory as '{output_file.name}'")

filter_names('palabras_DLE.txt', 'temp_filtered_dictionary.txt', names_modern)

def filter_through_dictionary(input_file, filter, output_file):

    """
    Filters lines from an input file based on a dictionary or list of filter words.

    Parameters:
    - input_file (str): Path to the input text file to filter.
    - filter (str): Path to the file containing filter words.
    - output_file (str): Path to the output file to save the filtered lines.

    """
    # Read the lines from input file and filter
    with open(input_file, 'r', encoding='utf-8') as input, open(filter, 'r', encoding='utf-8') as filter:
        lines_input = [line.strip().lower() for line in input]
        lines_filter = [line.strip().lower() for line in filter]

    print("\nFiltering names through filtered dictionary words...\n")
    # Remove lines from input that are also in filter and create final set
    filtered_lines = [line for line in lines_input if line not in lines_filter]

    # Write the filtered lines back to the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        for line in filtered_lines:
            output.write(line + '\n')

filter_through_dictionary('temp_old_names.txt', 'temp_filtered_dictionary.txt', 'temp_old_names_final.txt')

def old_names_final(input_file, output_file):
    # Read the list of names from the file
    print("Retrieving unique names...\n")

    with open(input_file, 'r', encoding='utf-8') as file:
        names = [line.strip() for line in file if line.strip()]

    # Remove duplicates
    unique_names = list(set(names))

    # Write the cleaned names back to the output file
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for name in unique_names:
            output_file.write(name + '\n')

    print('Almost done...\n')


    # Cleanup - Remove all files that begin with "temp"
    temp_files = []
    for temp_file in os.listdir():
        if temp_file.startswith("temp_"):
            temp_files.append(temp_file)
            os.remove(temp_file)

    if temp_files:
        print("Removed temporary files.")
    else:
        print("No temporary files found.")


    print(f"Final name list has been saved in your directory as '{output_file.name}'! Ready for training. :) ")

old_names_final('temp_old_names_final.txt', '0_old_names_final.txt')

# Data attribution:
# [1] Instituto Nacional de Estadística. (2023). Todos los nombres con frecuencia igual o mayor a 20 personas. Last accessed on March 1, 2024. https://www.ine.es/uc/nDER2igi
# [2] José Calvo Tello (Comp.) (2017). Corpus de novelas de la Edad de Plata Würzburg: CLiGS, 2017. Last accessed on February 29, 2024. https://github.com/cligs/textbox/tree/master/spanish/novela-espanola
# [3] Jorge Dueñas Lerín (2022). List of all Spanish words in RAE. Last accessed on February 29, 2024. \url{https://github.com/JorgeDuenasLerin/diccionario-espanol-txt/blob/master/0_palabras_todas.txt}.
