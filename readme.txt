Comments about the code: 

Our loadPairs() functions works in a different way than yours. We pass tokenized text contained in an array file as arguments, which creates the pairs. That's why you will notice the changes in the test part of your main function. We first pass a txt.file into text2sentences to generate the array of tokens. Then we use the loadPairs() function to convert it into pairs. Please, when testing the algorithm, try to pass a text or modify our loadPairs() function if you wish to pass an excel file.
