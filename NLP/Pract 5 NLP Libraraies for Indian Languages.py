#Practical No. 5: NLP Libraries for Indian Languages

#5(a) Hindi Word Tokenization

!pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
!pip install inltk
!pip install tornado==4.5.3

from inltk.inltk import setup, tokenize
setup('hi')
hindi_text = """प्राकृतिक भाषा सीखना बहुत दिलचस्प है।"""
print(tokenize(hindi_text, "hi"))






#5(b) Similar Hindi Sentences

!pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
!pip install inltk
!pip install tornado==4.5.3

from inltk.inltk import setup, get_similar_sentences
setup('hi')
output = get_similar_sentences('मैं आज बहुत खुश हूं', 5, 'hi')
print(output)






#5(c) Language Identification

!pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
!pip install inltk
!pip install tornado==4.5.3

from inltk.inltk import setup, identify_language
setup('gu')
print(identify_language('जीना श्रपडिया'))
