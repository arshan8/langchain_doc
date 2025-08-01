#heirarychial:
# first split by paragraphs, if chunksize > allowed chunk size then by sentences,.. then my words
# then by characters

#if very small, it can merge , upwards heirarcy

#recurive in nature.  


#all these like para, svcentacne,, word can made into list like allowed { \n \n, \n, " "}\

from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)